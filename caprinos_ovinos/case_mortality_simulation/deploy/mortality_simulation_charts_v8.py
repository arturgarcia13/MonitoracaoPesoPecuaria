import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score
import time
import os

import mortality_simulation_view_v8 as v8

# ─── CONFIGURAÇÃO DE PLOT ─────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def gerar_graficos_monte_carlo_v8():
    p = v8.SimParams()
    print(f"Executando {p.n_simulacoes} simulações de {p.n_animais} animais para os gráficos (v8)...")
    start_time = time.time()
    
    rng = np.random.default_rng(p.seed)
    
    sensibilidades = np.zeros(p.n_simulacoes)
    especificidades = np.zeros(p.n_simulacoes)
    
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    
    bins_calib = np.linspace(0, 100, 11)  
    calib_sum_pred = np.zeros(10)
    calib_sum_true = np.zeros(10)
    calib_counts = np.zeros(10)
    
    cum_det_fractions = np.zeros((p.n_simulacoes, len(p.dias_pesagem)))
    
    ultima_sim_dados = {}

    for sim in range(p.n_simulacoes):
        rebanho = v8.gerar_rebanho(p.n_animais, rng, p)
        P_t_real, P_t_alvo = v8.gerar_trajetoria(rebanho, rng, p)
        Z_atual, R_t, alertas = v8.calcular_risco_dinamico(rebanho, P_t_real, P_t_alvo, p)
        
        grupos = rebanho["grupos"]
        is_problematico = grupos > v8.GrupoMorbidade.SAUDAVEL
        alertou = alertas.any(axis=1)
        
        vp = np.sum(is_problematico & alertou)
        fn = np.sum(is_problematico & ~alertou)
        vn = np.sum(~is_problematico & ~alertou)
        fp = np.sum(~is_problematico & alertou)
        
        sensibilidades[sim] = vp / (vp + fn) if (vp + fn) > 0 else 0
        especificidades[sim] = vn / (vn + fp) if (vn + fp) > 0 else 0
        
        max_risk = R_t.max(axis=1)
        fpr, tpr, _ = roc_curve(is_problematico, max_risk / 100.0)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc_score(is_problematico, max_risk))
        
        bin_indices = np.digitize(max_risk, bins_calib) - 1
        bin_indices[bin_indices == 10] = 9 
        
        for b in range(10):
            mask = (bin_indices == b)
            calib_counts[b] += np.sum(mask)
            calib_sum_pred[b] += np.sum(max_risk[mask])
            calib_sum_true[b] += np.sum(is_problematico[mask])
            
        total_problematicos = np.sum(is_problematico)
        if total_problematicos > 0:
            vp_mask = is_problematico & alertou
            if vp_mask.any():
                primeiro_alerta_idx = np.argmax(alertas[vp_mask], axis=1)
                for j in range(len(p.dias_pesagem)):
                    det_ate_agora = np.sum(primeiro_alerta_idx <= j)
                    cum_det_fractions[sim, j] = det_ate_agora / total_problematicos
                
        if sim == p.n_simulacoes - 1:
            ultima_sim_dados = {
                'grupos': grupos,
                'P_t_real': P_t_real,
                'P_t_alvo': P_t_alvo,
                'Alertas': alertas,
                'max_risk': max_risk
            }

    print(f"Simulações concluídas em {time.time() - start_time:.2f} segundos. Gerando gráficos...")

    # ═══════════════════════════════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # 1. Histogramas (Sensibilidade e Especificidade)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(sensibilidades * 100, kde=True, ax=axes[0], color='#3498db', bins=30)
    axes[0].set_title('v8: Distribuição da Sensibilidade (1.000 runs)')
    axes[0].set_xlabel('Sensibilidade (%)')
    axes[0].set_ylabel('Frequência')
    
    sns.histplot(especificidades * 100, kde=True, ax=axes[1], color='#2ecc71', bins=30)
    axes[1].set_title('v8: Distribuição da Especificidade (1.000 runs)')
    axes[1].set_xlabel('Especificidade (%)')
    axes[1].set_ylabel('Frequência')
    
    plt.tight_layout()
    plt.savefig('v8_grafico_1_histogramas.png')
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
           title='v8: Curva ROC: Capacidade Discriminatória (1.000 runs)')
    ax.set_xlabel('Taxa de Falsos Positivos (1 - Especificidade)')
    ax.set_ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)')
    ax.legend(loc="lower right")
    plt.savefig('v8_grafico_2_roc_curve.png')
    plt.close()

    # 3. Curva de Calibração
    fig, ax = plt.subplots(figsize=(8, 6))
    mean_pred = calib_sum_pred / np.maximum(calib_counts, 1)
    mean_true = (calib_sum_true / np.maximum(calib_counts, 1)) * 100.0
    
    valid_bins = calib_counts > 0
    
    ax.plot(mean_pred[valid_bins], mean_true[valid_bins], "s-", color='#9b59b6', lw=2, markersize=8, label='Modelo Dinâmico $R_t$')
    ax.plot([0, 100], [0, 100], "k:", label='Calibração Perfeita')
    ax.set_title('v8: Curva de Calibração: Risco Previsto vs Ocorrência Real')
    ax.set_xlabel('Risco Previsto Máximo (% no Decil)')
    ax.set_ylabel('Proporção Real de Animais Problemáticos (%)')
    ax.legend(loc='upper left')
    plt.savefig('v8_grafico_3_calibracao.png')
    plt.close()

    # 4. Trajetória Temporal de Peso
    fig, ax = plt.subplots(figsize=(10, 6))
    
    grupos = ultima_sim_dados['grupos']
    alertas = ultima_sim_dados['Alertas']
    p_t_real = ultima_sim_dados['P_t_real']
    p_t_alvo = ultima_sim_dados['P_t_alvo']
    
    dias_pesagem = p.dias_pesagem
    
    idx_saudavel = np.where((grupos == v8.GrupoMorbidade.SAUDAVEL) & (~alertas.any(axis=1)))[0]
    idx_subnutrido = np.where((grupos == v8.GrupoMorbidade.SUBNUTRIDO) & (alertas.any(axis=1)))[0]
    idx_critico = np.where((grupos == v8.GrupoMorbidade.CRITICO) & (alertas.any(axis=1)))[0]
    
    if len(idx_saudavel) > 0:
        idx_saudavel = idx_saudavel[0]
        ax.plot(dias_pesagem, p_t_alvo[idx_saudavel], 'k--', lw=2, label='Curva Esperada (Referência)')
        ax.plot(dias_pesagem, p_t_real[idx_saudavel], marker='o', color='#2ecc71', lw=2, label='Saudável')
        
    if len(idx_subnutrido) > 0:
        idx_subnutrido = idx_subnutrido[0]
        ax.plot(dias_pesagem, p_t_real[idx_subnutrido], marker='o', color='#f39c12', lw=2, label='Subnutrido')
        alert_days = np.where(alertas[idx_subnutrido])[0]
        if len(alert_days) > 0:
            first_alert_idx = alert_days[0]
            ax.scatter(dias_pesagem[first_alert_idx], p_t_real[idx_subnutrido, first_alert_idx], 
                       s=200, facecolors='none', edgecolors='red', lw=2, zorder=5, label='Disparo do Alerta')
                       
    if len(idx_critico) > 0:
        idx_critico = idx_critico[0]
        ax.plot(dias_pesagem, p_t_real[idx_critico], marker='o', color='#e74c3c', lw=2, label='Crítico')
        alert_days = np.where(alertas[idx_critico])[0]
        if len(alert_days) > 0:
            first_alert_idx = alert_days[0]
            ax.scatter(dias_pesagem[first_alert_idx], p_t_real[idx_critico, first_alert_idx], 
                       s=200, facecolors='none', edgecolors='red', lw=2, zorder=5)

    ax.set_title('v8: Trajetória Temporal do Peso e Disparo de Alertas')
    ax.set_xlabel('Dias de Vida')
    ax.set_ylabel('Peso Observado (kg)')
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left')
    
    plt.savefig('v8_grafico_4_trajetoria_peso.png')
    plt.close()

    # 5. Curva de Detecção Acumulada
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_cum_det = np.mean(cum_det_fractions, axis=0) * 100.0 
    std_cum_det = np.std(cum_det_fractions, axis=0) * 100.0
    
    ax.step(dias_pesagem, mean_cum_det, where='post', color='#e67e22', lw=3, label='Média Detecção')
    
    ax.fill_between(dias_pesagem, mean_cum_det - std_cum_det, mean_cum_det + std_cum_det, 
                    step='post', color='#e67e22', alpha=0.3, label=r'$\pm$ 1 Desvio Padrão')

    ax.set_title('v8: Curva de Detecção Acumulada (Step Function)')
    ax.set_xlabel('Idade na Pesagem (dias)')
    ax.set_ylabel('Animais em Risco Detectados (%)')
    ax.set_xticks(dias_pesagem)
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right')
    plt.savefig('v8_grafico_5_deteccao_acumulada.png')
    plt.close()

    print("Todos os gráficos foram gerados e salvos em disco (v8_grafico_1_*.png a v8_grafico_5_*.png).")

if __name__ == "__main__":
    gerar_graficos_monte_carlo_v8()
