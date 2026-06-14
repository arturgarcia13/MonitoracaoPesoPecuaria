import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score
import time
import os

# ─── CONFIGURAÇÃO DE PLOT ─────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# ─── PARÂMETROS DA SIMULAÇÃO ──────────────────────────────────────────────────
N_SIMULACOES  = 1000
N_ANIMAIS     = 10000
LAMBDA_ALERTA = 30.0   
K_DECAIMENTO  = 0.35    # Usamos 0.5 para o modelo ser funcional e dinâmico

# Equação 1: Peso ao Nascer
BETA_0              =  4.10    
BETA_GEMEO          = -0.65    
BETA_TRIGEMEO       = -1.40    
BETA_FEMEA          = -0.30    
BETA_PRIMIPARA      = -0.35    
SIGMA_NASCIMENTO    =  0.66 # Aumentado para 0.66 para gerar mais variabilidade  

# Equação 3: Risco Logístico Basal
ALPHA_0_MACHO       = -2.5     
ALPHA_0_FEMEA       = -3.0     
ALPHA_1             =  1.2     
P_OPT_SIMPLES       =  3.96    
P_OPT_GEMEO         =  3.63    
P_OPT_TRIGEMEO      =  3.44    

# Equação 2: GMD e Trajetória de Peso
GMD_BASE                = 0.252    
PENALIDADE_MULTIPLO_GMD = -0.025   
GAMMA                   = 0.02     
SIGMA_GMD               = 0.015    

LAMBDA                  = 1.5      
T_MAX                   = 90       

# Etapa 6: Proporções de Morbidade e Multiplicadores
PROP_SAUDAVEL   = 0.70
PROP_SUBNUTRIDO = 0.15
PROP_DOENTE     = 0.10
PROP_CRITICO    = 0.05

MULT_GMD_SAUDAVEL   = 1.0
MULT_GMD_SUBNUTRIDO = 0.7
MULT_GMD_DOENTE     = 0.5
MULT_GMD_CRITICO    = 0.2

DIA_QUEDA_DOENTE = 20
SIGMA_BALANCA = 0.10

# Proporções da População
PROP_SIMPLES    = 0.65
PROP_GEMEO      = 0.30
PROP_TRIGEMEO   = 0.05
PROP_SEXO       = 0.50  
PROP_PRIMIPARA  = 0.20

DIAS_PESAGEM = np.array([0, 15, 30, 45, 60])

# ═══════════════════════════════════════════════════════════════════════════════
# MOTOR E COLETA DE DADOS
# ═══════════════════════════════════════════════════════════════════════════════
def gerar_graficos_monte_carlo():
    print(f"Executando {N_SIMULACOES} simulações de {N_ANIMAIS} animais para os gráficos...")
    start_time = time.time()
    
    rng = np.random.default_rng(42)
    
    # Arrays de armazenamento para métricas
    sensibilidades = np.zeros(N_SIMULACOES)
    especificidades = np.zeros(N_SIMULACOES)
    
    # Para ROC
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    
    # Para Curva de Calibração (Agrupamento Global)
    bins_calib = np.linspace(0, 100, 11)  # Decis de 0 a 100
    calib_sum_pred = np.zeros(10)
    calib_sum_true = np.zeros(10)
    calib_counts = np.zeros(10)
    
    # Para Detecção Cumulativa
    cum_det_fractions = np.zeros((N_SIMULACOES, len(DIAS_PESAGEM)))
    
    # Armazenar estado da última simulação para Gráfico 4
    ultima_sim_dados = {}

    for sim in range(N_SIMULACOES):
        sexo = rng.binomial(1, PROP_SEXO, size=N_ANIMAIS)  
        primipara = rng.binomial(1, PROP_PRIMIPARA, size=N_ANIMAIS)  
        tipo_parto = rng.choice([0, 1, 2], size=N_ANIMAIS, p=[PROP_SIMPLES, PROP_GEMEO, PROP_TRIGEMEO])
        
        eta = rng.normal(0, SIGMA_NASCIMENTO, N_ANIMAIS)
        beta_parto_vec = np.choose(tipo_parto, [0.0, BETA_GEMEO, BETA_TRIGEMEO])
        p0 = BETA_0 + beta_parto_vec + (sexo * BETA_FEMEA) + (primipara * BETA_PRIMIPARA) + eta
        p0 = np.clip(p0, 0.5, 8.0)
        
        p_opt_bio = np.choose(tipo_parto, [P_OPT_SIMPLES, P_OPT_GEMEO, P_OPT_TRIGEMEO])
        alpha_0_vec = np.where(sexo == 1, ALPHA_0_FEMEA, ALPHA_0_MACHO)
        z_morte = alpha_0_vec + ALPHA_1 * (p0 - p_opt_bio)**2
        p_morte = 1.0 / (1.0 + np.exp(-z_morte))
        
        eps = rng.normal(0, SIGMA_GMD, N_ANIMAIS)
        penalidade_parto_gmd = np.choose(tipo_parto, [0.0, PENALIDADE_MULTIPLO_GMD, PENALIDADE_MULTIPLO_GMD])
        gmd_base = GMD_BASE + penalidade_parto_gmd + GAMMA * (p0 - p_opt_bio) + eps
        
        grupos = rng.choice([0, 1, 2, 3], size=N_ANIMAIS, p=[PROP_SAUDAVEL, PROP_SUBNUTRIDO, PROP_DOENTE, PROP_CRITICO])
        is_problematico = grupos > 0
        
        P_t_real = np.zeros((N_ANIMAIS, len(DIAS_PESAGEM)))
        P_t_alvo = np.zeros((N_ANIMAIS, len(DIAS_PESAGEM)))
        
        for j, t in enumerate(DIAS_PESAGEM):
            P_t_alvo[:, j] = p0 + gmd_base * t
            
            gmd_efetivo = np.copy(gmd_base)
            gmd_efetivo[grupos == 1] *= MULT_GMD_SUBNUTRIDO
            gmd_efetivo[grupos == 3] *= MULT_GMD_CRITICO
            
            if t <= DIA_QUEDA_DOENTE:
                peso_t = p0 + gmd_efetivo * t
            else:
                gmd_efetivo[grupos == 2] *= MULT_GMD_DOENTE
                peso_t = np.where(
                    grupos == 2,
                    p0 + gmd_base * DIA_QUEDA_DOENTE + gmd_efetivo * (t - DIA_QUEDA_DOENTE),
                    p0 + gmd_efetivo * t
                )
            
            peso_t += rng.normal(0, SIGMA_BALANCA, N_ANIMAIS)
            P_t_real[:, j] = peso_t
            
        sigma_t = SIGMA_NASCIMENTO + LAMBDA * (DIAS_PESAGEM / T_MAX)**2
        Z_atual = (P_t_real - P_t_alvo) / sigma_t
        
        # [C2] R(t)
        R_t = (p_morte[:, None] * 100.0) * np.exp(K_DECAIMENTO * np.maximum(0.0, -Z_atual))
        R_t = np.clip(R_t, 0.0, 100.0)
        
        Alertas = R_t > LAMBDA_ALERTA
        alertou_algum_dia = Alertas.any(axis=1)
        
        # Sensibilidade / Especificidade
        VP = np.sum(is_problematico & alertou_algum_dia)
        FN = np.sum(is_problematico & ~alertou_algum_dia)
        VN = np.sum(~is_problematico & ~alertou_algum_dia)
        FP = np.sum(~is_problematico & alertou_algum_dia)
        
        sensibilidades[sim] = VP / (VP + FN) if (VP + FN) > 0 else 0
        especificidades[sim] = VN / (VN + FP) if (VN + FP) > 0 else 0
        
        # Curva ROC
        max_risk = np.max(R_t, axis=1)
        fpr, tpr, _ = roc_curve(is_problematico, max_risk / 100.0)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc_score(is_problematico, max_risk))
        
        # Calibração
        bin_indices = np.digitize(max_risk, bins_calib) - 1
        # Correção para o valor exato 100.0 que cai no bin 10
        bin_indices[bin_indices == 10] = 9 
        
        for b in range(10):
            mask = (bin_indices == b)
            calib_counts[b] += np.sum(mask)
            calib_sum_pred[b] += np.sum(max_risk[mask])
            calib_sum_true[b] += np.sum(is_problematico[mask])
            
        # Detecção Cumulativa
        total_problematicos = np.sum(is_problematico)
        if total_problematicos > 0:
            vp_mask = is_problematico & alertou_algum_dia
            # Primeiro dia em que Alertas é True para os VP
            # np.argmax pega o primeiro índice de True
            primeiro_alerta_idx = np.argmax(Alertas[vp_mask], axis=1)
            
            # Contagem cumulativa por dia
            for j in range(len(DIAS_PESAGEM)):
                det_ate_agora = np.sum(primeiro_alerta_idx <= j)
                cum_det_fractions[sim, j] = det_ate_agora / total_problematicos
                
        # Salva a última simulação para os trajetos temporais
        if sim == N_SIMULACOES - 1:
            ultima_sim_dados = {
                'grupos': grupos,
                'P_t_real': P_t_real,
                'P_t_alvo': P_t_alvo,
                'Alertas': Alertas,
                'max_risk': max_risk
            }

    print(f"Simulações concluídas em {time.time() - start_time:.2f} segundos. Gerando gráficos...")

    # ═══════════════════════════════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # 1. Histogramas (Sensibilidade e Especificidade)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(sensibilidades * 100, kde=True, ax=axes[0], color='#3498db', bins=30)
    axes[0].set_title('Distribuição da Sensibilidade (1.000 runs)')
    axes[0].set_xlabel('Sensibilidade (%)')
    axes[0].set_ylabel('Frequência')
    
    sns.histplot(especificidades * 100, kde=True, ax=axes[1], color='#2ecc71', bins=30)
    axes[1].set_title('Distribuição da Especificidade (1.000 runs)')
    axes[1].set_xlabel('Especificidade (%)')
    axes[1].set_ylabel('Frequência')
    
    plt.tight_layout()
    plt.savefig('grafico_1_histogramas.png')
    plt.close()
    
    # 2. Curva ROC
    fig, ax = plt.subplots(figsize=(8, 6))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Média ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)
            
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 Desvio Padrão')
                    
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Sorte', alpha=.8)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title='Curva ROC: Capacidade Discriminatória (1.000 runs)')
    ax.set_xlabel('Taxa de Falsos Positivos (1 - Especificidade)')
    ax.set_ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)')
    ax.legend(loc="lower right")
    plt.savefig('grafico_2_roc_curve.png')
    plt.close()

    # 3. Curva de Calibração
    fig, ax = plt.subplots(figsize=(8, 6))
    mean_pred = calib_sum_pred / np.maximum(calib_counts, 1)
    mean_true = (calib_sum_true / np.maximum(calib_counts, 1)) * 100.0 # em porcentagem
    
    # Filtrar bins vazios
    valid_bins = calib_counts > 0
    
    ax.plot(mean_pred[valid_bins], mean_true[valid_bins], "s-", color='#9b59b6', lw=2, markersize=8, label='Modelo Dinâmico $R_t$')
    ax.plot([0, 100], [0, 100], "k:", label='Calibração Perfeita')
    ax.set_title('Curva de Calibração: Risco Previsto vs Ocorrência Real')
    ax.set_xlabel('Risco Previsto Máximo (% no Decil)')
    ax.set_ylabel('Proporção Real de Animais Problemáticos (%)')
    ax.legend(loc='upper left')
    plt.savefig('grafico_3_calibracao.png')
    plt.close()

    # 4. Trajetória Temporal de Peso
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Encontrar índices representativos
    grupos = ultima_sim_dados['grupos']
    alertas = ultima_sim_dados['Alertas']
    p_t_real = ultima_sim_dados['P_t_real']
    p_t_alvo = ultima_sim_dados['P_t_alvo']
    
    idx_saudavel = np.where((grupos == 0) & (~alertas.any(axis=1)))[0][0]
    idx_subnutrido = np.where((grupos == 1) & (alertas.any(axis=1)))[0][0]
    idx_critico = np.where((grupos == 3) & (alertas.any(axis=1)))[0][0]
    
    # Plota a referência (usando o alvo do saudável como guia)
    ax.plot(DIAS_PESAGEM, p_t_alvo[idx_saudavel], 'k--', lw=2, label='Curva Esperada (Referência)')
    
    perfis = [
        (idx_saudavel, 'Saudável', '#2ecc71'),
        (idx_subnutrido, 'Subnutrido', '#f39c12'),
        (idx_critico, 'Crítico', '#e74c3c')
    ]
    
    for idx, label, color in perfis:
        # Plot da trajetória
        ax.plot(DIAS_PESAGEM, p_t_real[idx], marker='o', color=color, lw=2, label=label)
        
        # Ponto de alerta
        alert_days = np.where(alertas[idx])[0]
        if len(alert_days) > 0:
            first_alert_idx = alert_days[0]
            ax.scatter(DIAS_PESAGEM[first_alert_idx], p_t_real[idx, first_alert_idx], 
                       s=200, facecolors='none', edgecolors='red', lw=2, zorder=5)
            # Add label for the first circle only to show in legend if needed
            if label == 'Subnutrido': # just arbitrary to get it in the legend once
                ax.scatter([], [], s=150, facecolors='none', edgecolors='red', lw=2, label='Disparo do Alerta')

    ax.set_title('Trajetória Temporal do Peso e Disparo de Alertas')
    ax.set_xlabel('Dias de Vida')
    ax.set_ylabel('Peso Observado (kg)')
    ax.legend(loc='upper left')
    plt.savefig('grafico_4_trajetoria_peso.png')
    plt.close()

    # 5. Curva de Detecção Acumulada
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_cum_det = np.mean(cum_det_fractions, axis=0) * 100.0 # Em percentual
    std_cum_det = np.std(cum_det_fractions, axis=0) * 100.0
    
    ax.step(DIAS_PESAGEM, mean_cum_det, where='post', color='#e67e22', lw=3, label='Média Detecção')
    
    # Fill between for step functions requires explicitly repeating elements to match the step geometry
    # We use step for visualization but we can just fill_between standardly as Seaborn does
    ax.fill_between(DIAS_PESAGEM, mean_cum_det - std_cum_det, mean_cum_det + std_cum_det, 
                    step='post', color='#e67e22', alpha=0.3, label=r'$\pm$ 1 Desvio Padrão')

    ax.set_title('Curva de Detecção Acumulada (Step Function)')
    ax.set_xlabel('Idade na Pesagem (dias)')
    ax.set_ylabel('Animais em Risco Detectados (%)')
    ax.set_xticks(DIAS_PESAGEM)
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right')
    plt.savefig('grafico_5_deteccao_acumulada.png')
    plt.close()

    print("Todos os gráficos foram gerados e salvos em disco (grafico_1_*.png a grafico_5_*.png).")

if __name__ == "__main__":
    gerar_graficos_monte_carlo()
