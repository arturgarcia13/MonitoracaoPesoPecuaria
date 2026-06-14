"""
Motor de Validação Monte Carlo — Zootecnia de Precisão
Validação de Robustez do Risco Longitudinal R(t)

Implementação das Etapas 6 a 9:
- Injeção de 4 grupos de morbidade (Saudável, Subnutrido, Doente, Crítico).
- Simulação vetorizada de 10.000 animais x 1.000 iterações.
- Cálculo de Sensibilidade, Especificidade e Tempo Médio de Detecção.
"""

import numpy as np
import time

# ─── PARÂMETROS DA SIMULAÇÃO MONTE CARLO ──────────────────────────────────────
N_SIMULACOES  = 1000
N_ANIMAIS     = 10000
LAMBDA_ALERTA = 30.0   # Limiar de risco R(t) em % para disparar o alerta

# ─── CONSTANTES BIOLÓGICAS (Mantidas da v5) ───────────────────────────────────
BETA_0           =  4.10
BETA_GEMEO       = -0.65
BETA_TRIGEMEO    = -1.40
BETA_FEMEA       = -0.30
BETA_PRIMIPARA   = -0.35
SIGMA_NASCIMENTO =  0.35

GMD_BASE                = 0.252
GAMMA                   = 0.02
PENALIDADE_MULTIPLO_GMD = -0.025

ALPHA_0_MACHO   = -2.5
ALPHA_0_FEMEA   = -3.0
ALPHA_1         =  1.2
P_OPT_SIMPLES   =  3.96
P_OPT_GEMEO     =  3.63
P_OPT_TRIGEMEO  =  3.44

LAMBDA_CONE  = 1.5
T_MAX        = 90
K_DECAIMENTO = 0.5

# Proporções Populacionais
PROP_SIMPLES   = 0.65
PROP_GEMEO     = 0.30
PROP_TRIGEMEO  = 0.05
PROP_FEMEAS    = 0.50
PROP_PRIMIPARA = 0.20

# Proporções de Morbidade (Etapa 6)
# 0: Saudável (1.0x GMD), 1: Subnutrido (0.7x), 2: Doente (0.5x), 3: Crítico (0.2x)
PROP_SAUDAVEL   = 0.70
PROP_SUBNUTRIDO = 0.15
PROP_DOENTE     = 0.10
PROP_CRITICO    = 0.05
MULT_GMD_MORB   = np.array([1.0, 0.7, 0.5, 0.2])

DIAS_PESAGEM = np.array([15, 30, 45, 60, 75, 90])

# ═══════════════════════════════════════════════════════════════════════════════
# MOTOR VETORIZADO MONTE CARLO
# ═══════════════════════════════════════════════════════════════════════════════
def rodar_monte_carlo():
    print(f"Iniciando Monte Carlo: {N_SIMULACOES} simulações de {N_ANIMAIS} animais...")
    start_time = time.time()
    
    rng = np.random.default_rng(42)
    
    # Acumuladores de métricas
    metricas = {
        'sensibilidade': np.zeros(N_SIMULACOES),
        'especificidade': np.zeros(N_SIMULACOES),
        'tempo_deteccao': np.zeros(N_SIMULACOES)
    }

    for sim in range(N_SIMULACOES):
        # ── Etapa 1: Gerar os fatores iniciais
        tipos_parto = rng.choice([0, 1, 2], size=N_ANIMAIS, p=[PROP_SIMPLES, PROP_GEMEO, PROP_TRIGEMEO]) # 0:S, 1:G, 2:T
        sexos = rng.binomial(1, PROP_FEMEAS, size=N_ANIMAIS) # 0:M, 1:F
        paridades = rng.binomial(1, PROP_PRIMIPARA, size=N_ANIMAIS) # 0:Multi, 1:Primi
        
        # ── Etapa 6: Atribuir Grupo de Morbidade (Problemas Reais)
        grupos_morb = rng.choice(
            [0, 1, 2, 3], size=N_ANIMAIS, 
            p=[PROP_SAUDAVEL, PROP_SUBNUTRIDO, PROP_DOENTE, PROP_CRITICO]
        )
        is_problematico = grupos_morb > 0  # Máscara booleana para VP e FN

        # ── Etapa 2: Gerar Peso ao Nascer
        beta_parto = np.choose(tipos_parto, [0.0, BETA_GEMEO, BETA_TRIGEMEO])
        beta_sexo = sexos * BETA_FEMEA
        beta_matriz = paridades * BETA_PRIMIPARA
        
        p0 = BETA_0 + beta_parto + beta_sexo + beta_matriz + rng.normal(0, SIGMA_NASCIMENTO, N_ANIMAIS)
        p0 = np.clip(p0, 0.5, 8.0)

        # ── Oculto Biológico vs Sistema (Simplificado: assumimos sistema ideal calibrado)
        # O sistema prevê o risco inicial (R_nasc) usando seus betas calibrados.
        p_opt_bio = np.choose(tipos_parto, [P_OPT_SIMPLES, P_OPT_GEMEO, P_OPT_TRIGEMEO])
        alpha0_bio = np.where(sexos == 1, ALPHA_0_FEMEA, ALPHA_0_MACHO)
        
        z_nasc = alpha0_bio + ALPHA_1 * (p0 - p_opt_bio)**2
        r_nasc = (1.0 / (1.0 + np.exp(-z_nasc))) * 100.0

        # ── Etapa 4: Gerar o GMD e aplicar efeito da Morbidade
        gmd_base_indiv = GMD_BASE + GAMMA * (p0 - p_opt_bio)
        gmd_base_indiv += np.where(tipos_parto > 0, PENALIDADE_MULTIPLO_GMD, 0.0)
        
        # Multiplicador do grupo (Etapa 6) + ruído de desenvolvimento
        gmd_real = (gmd_base_indiv * MULT_GMD_MORB[grupos_morb]) + rng.normal(0, 0.015, N_ANIMAIS)
        
        # ── Etapas 5, 7, 8 e 9: Simular Pesagens e Disparar Alerta (Vetorizado para todo o tempo T)
        # Formatos: p0 (N,), gmd_real (N,), DIAS_PESAGEM (T,) -> Broadcasting para matriz (N, T)
        P_t_real = p0[:, None] + gmd_real[:, None] * DIAS_PESAGEM
        P_t_real += rng.normal(0, 0.15, size=(N_ANIMAIS, len(DIAS_PESAGEM))) # Ruído de balança
        
        # Alvo esperado pelo sistema (assume que o animal seria saudável, mult=1.0)
        P_t_alvo = p0[:, None] + gmd_base_indiv[:, None] * DIAS_PESAGEM
        Sigma_t = SIGMA_NASCIMENTO * (1 + LAMBDA_CONE * (DIAS_PESAGEM / T_MAX)**2)
        
        # Z atual matricial (N_animais, T_dias)
        Z_atual = (P_t_real - P_t_alvo) / Sigma_t
        
        # Risco longitudinal (Clamp Max de 100%)
        # np.maximum(0, -Z_atual) penaliza apenas Z negativo
        R_t = r_nasc[:, None] * np.exp(K_DECAIMENTO * np.maximum(0.0, -Z_atual))
        R_t = np.clip(R_t, 0.0, 100.0)

        # Matriz booleana de alertas: True onde R_t > Lambda
        Alertas = R_t > LAMBDA_ALERTA
        
        # Reduzindo a dimensão T: o animal alertou em algum momento?
        alertou_algum_dia = Alertas.any(axis=1)

        # ── Cálculo das Métricas (Sensibilidade, Especificidade)
        # VP: Problemático E Alertou
        # FN: Problemático E NÃO Alertou
        VP = np.sum(is_problematico & alertou_algum_dia)
        FN = np.sum(is_problematico & ~alertou_algum_dia)
        
        # VN: Saudável E NÃO Alertou
        # FP: Saudável E Alertou
        VN = np.sum(~is_problematico & ~alertou_algum_dia)
        FP = np.sum(~is_problematico & alertou_algum_dia)
        
        sensibilidade = VP / (VP + FN) if (VP + FN) > 0 else 0
        especificidade = VN / (VN + FP) if (VN + FP) > 0 else 0
        
        # Tempo Médio de Detecção (Apenas para os VP)
        # argmax no booleano retorna o primeiro índice onde é True
        dias_primeiro_alerta = DIAS_PESAGEM[np.argmax(Alertas[is_problematico & alertou_algum_dia], axis=1)]
        tempo_medio = np.mean(dias_primeiro_alerta) if len(dias_primeiro_alerta) > 0 else np.nan

        metricas['sensibilidade'][sim] = sensibilidade
        metricas['especificidade'][sim] = especificidade
        metricas['tempo_deteccao'][sim] = tempo_medio

    # ─── RELATÓRIO FINAL ────────────────────────────────────────────────────────
    end_time = time.time()
    
    print(f"\n{'─' * 60}")
    print(f" RESULTADOS DA SIMULAÇÃO MONTE CARLO")
    print(f" Tempo de execução: {end_time - start_time:.2f} segundos")
    print(f"{'─' * 60}")
    
    sens_mean = np.mean(metricas['sensibilidade']) * 100
    sens_std = np.std(metricas['sensibilidade']) * 100
    spec_mean = np.mean(metricas['especificidade']) * 100
    spec_std = np.std(metricas['especificidade']) * 100
    tempo_mean = np.nanmean(metricas['tempo_deteccao'])
    
    print(f" 1. Sensibilidade (Recuperação de Doentes/Críticos):")
    print(f"    Média: {sens_mean:.2f}% (± {sens_std:.2f}%)")
    print(f"    O sistema detecta ~{sens_mean/100:.2%} dos animais fora do padrão.")
    
    print(f"\n 2. Especificidade (Silêncio em Animais Saudáveis):")
    print(f"    Média: {spec_mean:.2f}% (± {spec_std:.2f}%)")
    print(f"    O sistema evita alarmes falsos em ~{spec_mean/100:.2%} dos casos.")
    
    print(f"\n 3. Tempo Médio de Detecção (KPI Operacional):")
    print(f"    Média: {tempo_mean:.1f} dias de vida.")
    print(f"{'─' * 60}\n")

if __name__ == "__main__":
    rodar_monte_carlo()