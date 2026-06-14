import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import time

# ─── PARÂMETROS DA SIMULAÇÃO MONTE CARLO ──────────────────────────────────────
N_SIMULACOES  = 1000
N_ANIMAIS     = 10000
LAMBDA_ALERTA = 30.0   # Limiar de risco R(t) em % para disparar o alerta
K_DECAIMENTO  = 0.4    

# ─── CONSTANTES BIOLÓGICAS (Exatamente iguais à v5) ───────────────────────────

# Equação 1: Peso ao Nascer
BETA_0              =  4.10    # Intercepto (kg)
BETA_GEMEO          = -0.65    # Efeito gêmeo (kg)
BETA_TRIGEMEO       = -1.40    # Efeito trigêmeo (kg)
BETA_FEMEA          = -0.30    # Efeito fêmea (kg)
BETA_PRIMIPARA      = -0.35    # Efeito primípara (kg)
SIGMA_NASCIMENTO    =  0.35    # Erro aleatório eta (kg)

# Equação 3: Risco Logístico Basal
ALPHA_0_MACHO       = -2.5     # Intercepto machos
ALPHA_0_FEMEA       = -3.0     # Intercepto fêmeas
ALPHA_1             =  1.2     # Coeficiente quadrático
P_OPT_SIMPLES       =  3.96    # kg
P_OPT_GEMEO         =  3.63    # kg
P_OPT_TRIGEMEO      =  3.44    # kg

# Equação 2: GMD e Trajetória de Peso
GMD_BASE                = 0.252    # GMD base (kg/dia)
PENALIDADE_MULTIPLO_GMD = -0.025   # Penalização no GMD por parto múltiplo
GAMMA                   = 0.02     # Sensibilidade do GMD ao peso
P_REF_GMD               = 4.0
SIGMA_GMD               = 0.015    # Erro no desenvolvimento (kg/dia) - ruído do Monte Carlo

LAMBDA                  = 1.5      # Fator de escala biométrica (Banda adaptativa)
T_MAX                   = 90       # Dias até desmama

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

# Proporções da População (Exatamente iguais à v5)
PROP_SIMPLES    = 0.65
PROP_GEMEO      = 0.30
PROP_TRIGEMEO   = 0.05
PROP_SEXO       = 0.50  # Corresponde a PROP_FEMEAS
PROP_PRIMIPARA  = 0.20

# Períodos de Avaliação
DIAS_PESAGEM = np.array([0, 15, 30, 45, 60])

# ═══════════════════════════════════════════════════════════════════════════════
# MOTOR DE SIMULAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════
def rodar_monte_carlo():
    print(f"Iniciando Monte Carlo: {N_SIMULACOES} simulações de {N_ANIMAIS} animais...")
    start_time = time.time()
    
    rng = np.random.default_rng(42)
    
    metricas = {
        'sensibilidade': np.zeros(N_SIMULACOES),
        'especificidade': np.zeros(N_SIMULACOES),
        'tempo_deteccao': np.zeros(N_SIMULACOES),
        'auc': np.zeros(N_SIMULACOES)
    }

    for sim in range(N_SIMULACOES):
        # ── Etapa 1: Fatores Iniciais
        sexo = rng.binomial(1, PROP_SEXO, size=N_ANIMAIS)  
        primipara = rng.binomial(1, PROP_PRIMIPARA, size=N_ANIMAIS)  
        tipo_parto = rng.choice([0, 1, 2], size=N_ANIMAIS, p=[PROP_SIMPLES, PROP_GEMEO, PROP_TRIGEMEO])
        
        # ── Etapa 2: Peso ao Nascer
        eta = rng.normal(0, SIGMA_NASCIMENTO, N_ANIMAIS)
        beta_parto_vec = np.choose(tipo_parto, [0.0, BETA_GEMEO, BETA_TRIGEMEO])
        p0 = BETA_0 + beta_parto_vec + (sexo * BETA_FEMEA) + (primipara * BETA_PRIMIPARA) + eta
        # A v5 limitava o peso para evitar extremos irreais
        p0 = np.clip(p0, 0.5, 8.0)
        
        # P_opt biológico por tipo de parto
        p_opt_bio = np.choose(tipo_parto, [P_OPT_SIMPLES, P_OPT_GEMEO, P_OPT_TRIGEMEO])
        
        # ── Etapa 3: Risco Verdadeiro (P(Morte))
        alpha_0_vec = np.where(sexo == 1, ALPHA_0_FEMEA, ALPHA_0_MACHO)
        z_morte = alpha_0_vec + ALPHA_1 * (p0 - p_opt_bio)**2
        p_morte = 1.0 / (1.0 + np.exp(-z_morte))
        
        # ── Etapa 4: GMD Esperado
        eps = rng.normal(0, SIGMA_GMD, N_ANIMAIS)
        penalidade_parto_gmd = np.choose(tipo_parto, [0.0, PENALIDADE_MULTIPLO_GMD, PENALIDADE_MULTIPLO_GMD])
        gmd_base = GMD_BASE + penalidade_parto_gmd + GAMMA * (p0 - p_opt_bio) + eps
        
        # ── Etapa 6: Grupos de Morbidade (Problemas Reais)
        grupos = rng.choice([0, 1, 2, 3], size=N_ANIMAIS, p=[PROP_SAUDAVEL, PROP_SUBNUTRIDO, PROP_DOENTE, PROP_CRITICO])
        is_problematico = grupos > 0
        
        # ── Etapas 5 e 6 combinadas: Construindo P_real
        P_t_real = np.zeros((N_ANIMAIS, len(DIAS_PESAGEM)))
        P_t_alvo = np.zeros((N_ANIMAIS, len(DIAS_PESAGEM)))
        
        for j, t in enumerate(DIAS_PESAGEM):
            # Target Ideal do Sistema
            P_t_alvo[:, j] = p0 + gmd_base * t
            
            # GMD Real
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
            
            # Ruído da Balança
            ruido = rng.normal(0, SIGMA_BALANCA, N_ANIMAIS)
            peso_t += ruido
            
            P_t_real[:, j] = peso_t
            
        # ── Etapa 8: Banda Adaptativa (Sigma_t)
        sigma_t = SIGMA_NASCIMENTO + LAMBDA * (DIAS_PESAGEM / T_MAX)**2
        
        # ── Etapa 7: Calcular Z Atual
        Z_atual = (P_t_real - P_t_alvo) / sigma_t
        
        # ── Etapa 9: Disparar Alerta
        # [C2] R(t) assimétrico: só penaliza queda abaixo do alvo
        R_t = (p_morte[:, None] * 100.0) * np.exp(K_DECAIMENTO * np.maximum(0.0, -Z_atual))
        R_t = np.clip(R_t, 0.0, 100.0)
        
        Alertas = R_t > LAMBDA_ALERTA
        alertou_algum_dia = Alertas.any(axis=1)
        
        # ── Métricas de Avaliação
        VP = np.sum(is_problematico & alertou_algum_dia)
        FN = np.sum(is_problematico & ~alertou_algum_dia)
        VN = np.sum(~is_problematico & ~alertou_algum_dia)
        FP = np.sum(~is_problematico & alertou_algum_dia)
        
        metricas['sensibilidade'][sim] = VP / (VP + FN) if (VP + FN) > 0 else 0
        metricas['especificidade'][sim] = VN / (VN + FP) if (VN + FP) > 0 else 0
        
        # Tempo Médio de Detecção
        vp_mask = is_problematico & alertou_algum_dia
        if np.any(vp_mask):
            primeiro_alerta_idx = np.argmax(Alertas[vp_mask], axis=1)
            dias_primeiro_alerta = DIAS_PESAGEM[primeiro_alerta_idx]
            dias_antecedencia = DIAS_PESAGEM[-1] - dias_primeiro_alerta
            metricas['tempo_deteccao'][sim] = np.mean(dias_antecedencia)
        else:
            metricas['tempo_deteccao'][sim] = np.nan
            
        # Curva ROC (AUC)
        max_risk = np.max(R_t, axis=1)
        # Se R_t é constante (K_DECAIMENTO = 0), a curva ROC ainda será calculada com base no risco de nascimento
        metricas['auc'][sim] = roc_auc_score(is_problematico, max_risk)

    # ─── RELATÓRIO FINAL ────────────────────────────────────────────────────────
    end_time = time.time()
    
    print(f"\n{'-' * 60}")
    print(f" RESULTADOS DA SIMULAÇÃO MONTE CARLO ({N_SIMULACOES} runs)")
    print(f" Tempo de execução: {end_time - start_time:.2f} segundos")
    print(f"{'-' * 60}")
    
    sens_mean = np.mean(metricas['sensibilidade']) * 100
    sens_std = np.std(metricas['sensibilidade']) * 100
    spec_mean = np.mean(metricas['especificidade']) * 100
    spec_std = np.std(metricas['especificidade']) * 100
    tempo_mean = np.nanmean(metricas['tempo_deteccao'])
    auc_mean = np.mean(metricas['auc'])
    
    print(" 1. Sensibilidade (Recuperação de animais em risco):")
    print(f"    Média: {sens_mean:.2f}% (± {sens_std:.2f}%)")
    
    print("\n 2. Especificidade (Silêncio em Animais Saudáveis):")
    print(f"    Média: {spec_mean:.2f}% (± {spec_std:.2f}%)")
    
    print("\n 3. Tempo Médio de Detecção:")
    print(f"    O alerta soa, em média, {tempo_mean:.1f} dias antes do fechamento (dia 60).")
    
    print("\n 4. Curva ROC (AUC):")
    print(f"    AUC médio: {auc_mean:.4f} (Ideal > 0.80)")
    
    print(f"{'-' * 60}\n")

if __name__ == "__main__":
    rodar_monte_carlo()
