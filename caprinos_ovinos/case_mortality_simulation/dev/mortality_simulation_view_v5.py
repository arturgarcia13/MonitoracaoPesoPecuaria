"""
Simulação Preditiva — Zootecnia de Precisão e Risco Longitudinal
Espécie: Ovinos (Morada Nova / Santa Inês / Dorper)

Histórico de correções (v4 → v5):
  [C1] sigma_t revertida para Sarmento et al. (2010):
       σ(t) = 0.35 + 1.5·(t/90)²   [v4 usava forma multiplicativa incorreta]
  [C2] R(t) corrigida para assimetria biológica:
       R(t) = R_nasc · exp(k · max(0, −z))
       Z positivo (acima do alvo) mantém R_nasc — sobrepeso não é prêmio
       Z negativo (abaixo do alvo) eleva risco exponencialmente
       Clamp bilateral: max(0, min(100, R(t)))
  [C3] Sistema preditivo não acessa mais Alpha0_Biologico (verdade oculta);
       alpha0 é computado internamente a partir da coluna observável 'Sexo'
  [C4] GMD do cone usa p_opt do sistema calibrado, não P_Opt_Biologico;
       alinha gráficos 4 e 5 na mesma perspectiva (do sistema, não da biologia)
  [C5] Métricas sklearn agora são calculadas e impressas no __main__
  [C6] Dia 0 removido das pesagens; série começa em t=15 (primeira pesagem real)

Fundamentação teórica:
  Freitas et al. (1980)  — peso médio Morada Nova: 3.1 kg
  McMillan et al. (1983) — faixa ótima de sobrevivência: 3.3–4.1 kg (NZ)
  Everts et al. (1985)   — efeito sexo: fêmeas −0.19 kg
  Gardner et al.         — efeitos parto (gêmeo −0.692, trigêmeo −1.40),
                           sexo (−0.363), paridade (primípara −0.351 kg)
  Hatcher (2009)         — P_opt por tipo de parto: simples 4.00 kg, gêmeo 3.35 kg
  Sarmento et al. (2010) — variância heterogênea crescente com a idade (σ_t)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score

# ─── CONTROLE DE EXECUÇÃO ────────────────────────────────────────────────────
SALVAR_FIGURA   = True
ARQUIVO_SAIDA   = "zootecnia_precisao_v5.png"
ID_ANIMAL_CONE  = None   # None → último animal; altere para qualquer ID_Parto

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONSTANTES BIOLÓGICAS
# ═══════════════════════════════════════════════════════════════════════════════
ESPECIE = "Ovinos"

# ── Equação 1: Peso ao Nascer ─────────────────────────────────────────────────
# P0 = BETA_0 + β_parto + β_sexo + β_matriz + η,  η ~ N(0, SIGMA_NASCIMENTO²)
# β0 = 4.10 kg → Macho, Parto Simples, Mãe Multípara (regressao.md seção 1)
BETA_0              =  4.10    # kg — Hatcher (2009): parto simples = 4.00 kg
BETA_GEMEO          = -0.65    # kg — Hatcher (2009): 4.00 vs 3.35 kg
BETA_TRIGEMEO       = -1.40    # kg — Gardner et al.: decréscimo progressivo
BETA_FEMEA          = -0.30    # kg — média consensual Everts / Gardner / Medeiros
BETA_PRIMIPARA      = -0.35    # kg — Gardner et al.: maior incremento 1ª→2ª gestação
SIGMA_NASCIMENTO    =  0.35    # kg — desvio padrão do erro aleatório (regressao.md 3.2)

# ── Equação 2: GMD e Trajetória de Peso ──────────────────────────────────────
# Pt = P0 + GMD · t
# GMD = GMD_BASE + GAMMA · (P0 − P_REF_GMD)
GMD_BASE                = 0.252    # kg/dia — ovinos corte (regressao.md)
GAMMA                   = 0.02     # sensibilidade GMD ao P0 (regressao.md 3.2)
P_REF_GMD               = 4.0      # kg — ponto neutro do ajuste GMD (regressao.md)
PENALIDADE_MULTIPLO_GMD = -0.025   # kg/dia — penalização para partos não-simples

# ── Equação 3: Risco Logístico Basal (Verdade Oculta Biológica) ───────────────
# z = α0 + ALPHA_1 · (P0 − P_opt)²,  P(Y=1) = sigmoid(z)
# P_opt e α0 diferenciados por sexo e tipo de parto — mais fiel à biologia
ALPHA_0_MACHO   = -2.5     # intercepto machos (regressao.md Cenário B)
ALPHA_0_FEMEA   = -3.0     # fêmeas têm vantagem de sobrevivência
ALPHA_1         =  1.2     # coeficiente quadrático (regressao.md Cenário B)
P_OPT_SIMPLES   =  3.96    # kg — Hatcher (2009): média parto simples ≈ 4.00 kg
P_OPT_GEMEO     =  3.63    # kg — Hatcher (2009): média gêmeos
P_OPT_TRIGEMEO  =  3.44    # kg — extrapolação progressiva Gardner et al.

# ── Heterogeneidade Residual — Sarmento et al. (2010) ─────────────────────────
# σ(t) = SIGMA_NASCIMENTO + LAMBDA · (t / T_MAX)²
# [C1] Forma aditiva corrigida — v4 usava forma multiplicativa (metade do cone)
LAMBDA  = 1.5    # fator de escala biométrica (regressao.md seção 3.2)
T_MAX   = 90     # dias até desmama

# ── R(t): Dinâmica Longitudinal de Risco ──────────────────────────────────────
# [C2] R(t) = R_nasc · exp(k · max(0, −z)),  clamp ∈ [0, 100]
# Z negativo (abaixo do alvo) → risco cresce exponencialmente
# Z positivo (acima do alvo)  → risco mantém R_nasc (sobrepeso não é prêmio)
K_DECAIMENTO = 0  # sensibilidade ao desvio negativo (empírico; calibrar com dados locais)

# ── Sistema Preditivo (Crença Inicial) ────────────────────────────────────────
# McMillan et al. (1983): faixa de mínima mortalidade estimada em NZ
P_OPT_MIN_INICIAL = 3.3    # kg
P_OPT_MAX_INICIAL = 4.1    # kg

# ── Parâmetros de Simulação ───────────────────────────────────────────────────
N_PARTOS        = 200
GATILHO_RETREINO = 50

# Proporções populacionais do rebanho
PROP_SIMPLES    = 0.65
PROP_GEMEO      = 0.30
PROP_TRIGEMEO   = 0.05
PROP_FEMEAS     = 0.50
PROP_PRIMIPARA  = 0.20

# ═══════════════════════════════════════════════════════════════════════════════
# 2. GERAÇÃO DO MUNDO BIOLÓGICO
# ═══════════════════════════════════════════════════════════════════════════════
def gerar_rebanho_biologico(n_partos: int, rng: np.random.Generator) -> pd.DataFrame:
    """Gera animais com parâmetros biológicos reais (verdade oculta).
    As colunas P_Opt_Biologico e Alpha0_Biologico são usadas APENAS
    para calcular a mortalidade simulada — não são acessadas pelo sistema preditivo.
    """
    tipos_parto = rng.choice(
        ['Simples', 'Gemeo', 'Trigemeo'], size=n_partos,
        p=[PROP_SIMPLES, PROP_GEMEO, PROP_TRIGEMEO]
    )
    sexos     = rng.choice(['Macho', 'Femea'], size=n_partos, p=[1 - PROP_FEMEAS, PROP_FEMEAS])
    paridades = rng.choice(['Multipara', 'Primipara'], size=n_partos, p=[1 - PROP_PRIMIPARA, PROP_PRIMIPARA])

    cond_parto     = [tipos_parto == 'Simples', tipos_parto == 'Gemeo', tipos_parto == 'Trigemeo']
    beta_parto_vec = np.select(cond_parto, [0.0, BETA_GEMEO, BETA_TRIGEMEO])
    beta_sexo_vec  = np.where(sexos == 'Femea', BETA_FEMEA, 0.0)
    beta_mat_vec   = np.where(paridades == 'Primipara', BETA_PRIMIPARA, 0.0)

    pesos_nascer = (BETA_0 + beta_parto_vec + beta_sexo_vec + beta_mat_vec
                    + rng.normal(0, SIGMA_NASCIMENTO, size=n_partos))
    pesos_nascer = np.round(np.clip(pesos_nascer, 0.5, 8.0), 2)

    # Parâmetros ocultos individuais (biologia)
    p_opt_bio  = np.select(cond_parto, [P_OPT_SIMPLES, P_OPT_GEMEO, P_OPT_TRIGEMEO])
    alpha0_bio = np.where(sexos == 'Femea', ALPHA_0_FEMEA, ALPHA_0_MACHO)

    # GMD real
    gmd_real = GMD_BASE + GAMMA * (pesos_nascer - p_opt_bio)
    gmd_real = np.where(tipos_parto != 'Simples', gmd_real + PENALIDADE_MULTIPLO_GMD, gmd_real)

    # Desfecho real de mortalidade (estocástico)
    z_real               = alpha0_bio + ALPHA_1 * (pesos_nascer - p_opt_bio) ** 2
    prob_mortalidade_real = 1.0 / (1.0 + np.exp(-z_real))
    mortalidade_observada = rng.binomial(1, prob_mortalidade_real)

    return pd.DataFrame({
        'ID_Parto'              : range(1, n_partos + 1),
        'Tipo_Parto'            : tipos_parto,
        'Sexo'                  : sexos,
        'Paridade_Mae'          : paridades,
        'Peso_Nascer_Kg'        : pesos_nascer,
        'P_Opt_Biologico'       : p_opt_bio,      # referência analítica — não usar no sistema
        'Alpha0_Biologico'      : alpha0_bio,     # referência analítica — não usar no sistema
        'GMD_Real'              : np.round(gmd_real, 4),
        'Prob_Mortalidade_Real' : prob_mortalidade_real,
        'Risco_Real_%'          : prob_mortalidade_real * 100,
        'Mortalidade_Observada' : mortalidade_observada,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SISTEMA PREDITIVO
# ═══════════════════════════════════════════════════════════════════════════════
def simular_sistema_preditivo(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """Simula o sistema que opera em campo: só acessa colunas observáveis.

    [C3] alpha0 é computado internamente a partir da coluna 'Sexo'
         — sem acesso a Alpha0_Biologico (verdade oculta).

    Retorna o DataFrame enriquecido e os limites finais do sistema.
    """
    df_sim             = df.copy()
    p_opt_min_sistema  = P_OPT_MIN_INICIAL
    p_opt_max_sistema  = P_OPT_MAX_INICIAL
    retreino_feito     = False

    fases, prob_sistema_lista, risco_sistema_lista = [], [], []

    for index, linha in df_sim.iterrows():

        # Retreino automático ao atingir o gatilho
        if index >= GATILHO_RETREINO and not retreino_feito:
            amostra = df_sim.loc[0:(index - 1), 'Peso_Nascer_Kg']
            if not amostra.empty:
                p_opt_min_sistema = np.round(amostra.quantile(0.25), 2)
                p_opt_max_sistema = np.round(amostra.quantile(0.75), 2)
            retreino_feito = True

        fases.append('Cold Start' if not retreino_feito else 'Calibrado')

        # [C3] alpha0 calculado a partir do campo observável 'Sexo'
        alpha0_sistema = ALPHA_0_FEMEA if linha['Sexo'] == 'Femea' else ALPHA_0_MACHO

        p_opt_sistema  = (p_opt_min_sistema + p_opt_max_sistema) / 2.0
        z_sistema      = alpha0_sistema + ALPHA_1 * (linha['Peso_Nascer_Kg'] - p_opt_sistema) ** 2
        prob_s         = 1.0 / (1.0 + np.exp(-z_sistema))

        prob_sistema_lista.append(prob_s)
        risco_sistema_lista.append(prob_s * 100)

    df_sim['Fase']           = fases
    df_sim['Prob_Sistema']   = prob_sistema_lista
    df_sim['Risco_Sistema_%'] = risco_sistema_lista
    df_sim['Erro_Predicao']  = df_sim['Risco_Sistema_%'] - df_sim['Risco_Real_%']

    return df_sim, p_opt_min_sistema, p_opt_max_sistema


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MONITORAMENTO LONGITUDINAL
# ═══════════════════════════════════════════════════════════════════════════════
def gerar_monitoramento_longitudinal(
    animal: pd.Series,
    p_opt_sistema: float,
    rng: np.random.Generator
) -> dict:
    """Gera o cone de crescimento e R(t) da perspectiva do sistema.

    [C4] gmd_esperado usa p_opt_sistema (aprendido), não P_Opt_Biologico.
         Alinha gráficos 4 e 5 na mesma perspectiva.
    [C6] Pesagens começam em t=15; t=0 removido (Z sempre 0, sem informação).
    [C2] R(t) = R_nasc · exp(k · max(0, −z)), clamp ∈ [0, 100].
    """
    peso_nascer  = animal['Peso_Nascer_Kg']
    gmd_real     = animal['GMD_Real']
    risco_nasc   = animal['Risco_Sistema_%']

    # [C4] Trajetória alvo: perspectiva do sistema
    gmd_esperado = GMD_BASE + GAMMA * (peso_nascer - p_opt_sistema)
    if animal['Tipo_Parto'] != 'Simples':
        gmd_esperado += PENALIDADE_MULTIPLO_GMD

    vetor_t  = np.arange(0, T_MAX + 1)
    traj_alvo = peso_nascer + gmd_esperado * vetor_t

    # [C1] Sigma corrigido — forma aditiva (Sarmento et al. 2010)
    sigma_t = SIGMA_NASCIMENTO + LAMBDA * (vetor_t / T_MAX) ** 2

    # [C6] Pesagens: série começa em t=15
    dias_pesagem = np.array([15, 30, 45, 60, 75, 90])
    ruido_sanitario = np.array([
        rng.normal(0.0,  0.15),   # dia 15 — normal
        rng.normal(0.0,  0.20),   # dia 30 — normal
        rng.normal(-0.3, 0.15),   # dia 45 — leve queda (~0.5σ)
        rng.normal(-0.8, 0.20),   # dia 60 — evento sanitário (~0.7σ)
        rng.normal(-0.4, 0.20),   # dia 75 — recuperação parcial
        rng.normal(-0.1, 0.20),   # dia 90 — retorno ao normal
    ])
    pesagens_reais = (peso_nascer + gmd_real * dias_pesagem) + ruido_sanitario

    z_scores, riscos_t, cores_alerta = [], [], []

    for t, peso_medido in zip(dias_pesagem, pesagens_reais):
        peso_alvo_dia = peso_nascer + gmd_esperado * t
        sigma_dia     = SIGMA_NASCIMENTO + LAMBDA * (t / T_MAX) ** 2
        z_atual       = (peso_medido - peso_alvo_dia) / sigma_dia
        z_scores.append(z_atual)

        # [C2] R(t) assimétrico: só penaliza queda abaixo do alvo
        r_t = risco_nasc * np.exp(K_DECAIMENTO * max(0.0, -z_atual))
        r_t = max(0.0, min(100.0, r_t))
        riscos_t.append(r_t)

        if   z_atual >= -1.0: cores_alerta.append('#2ecc71')
        elif z_atual >= -2.0: cores_alerta.append('#f1c40f')
        else:                 cores_alerta.append('#e74c3c')

    return {
        'vetor_t'       : vetor_t,
        'traj_alvo'     : traj_alvo,
        'sigma_t'       : sigma_t,
        'dias_pesagem'  : dias_pesagem,
        'pesagens_reais': pesagens_reais,
        'cores_alerta'  : cores_alerta,
        'z_scores'      : z_scores,
        'riscos_t'      : riscos_t,
        'gmd_esperado'  : gmd_esperado,
        'risco_nasc'    : risco_nasc,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. VISUALIZAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════
def plotar_resultados(df: pd.DataFrame, dados_long: dict, animal: pd.Series) -> None:
    sns.set_theme(style="whitegrid")
    PALETTE_FASES = {'Cold Start': '#e74c3c', 'Calibrado': '#2ecc71'}

    fig = plt.figure(figsize=(16, 16))
    gs  = fig.add_gridspec(3, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])

    fig.suptitle(
        f'Zootecnia de Precisão: Risco Dinâmico — {ESPECIE}',
        fontsize=18, fontweight='bold', y=0.98
    )

    # Gráfico 1: Evolução do Erro de Predição
    ax1.axvline(x=GATILHO_RETREINO, color='black', linestyle='--', lw=1.2,
                label=f'Retreino (n={GATILHO_RETREINO})')
    sns.lineplot(data=df, x='ID_Parto', y='Erro_Predicao',
                 hue='Fase', palette=PALETTE_FASES, ax=ax1)
    ax1.set_title('Evolução do Erro de Predição (Adaptação à Distribuição Local)')
    ax1.set_ylabel('Erro (pp)')
    ax1.legend()

    # Gráfico 2: Distribuição de Peso e Desfechos por Sexo
    df_plot = df.copy()
    df_plot['Mortalidade_Observada'] = df_plot['Mortalidade_Observada'].map({0: 'Sobreviveu', 1: 'Óbito'})
    sns.boxplot(data=df_plot, x='Sexo', y='Peso_Nascer_Kg',
                hue='Mortalidade_Observada', ax=ax2, palette=['#3498db', '#e74c3c'])
    ax2.set_title('Peso ao Nascer e Desfechos por Sexo')
    ax2.set_ylabel('Peso ao Nascer (kg)')

    # Gráfico 3: Curvas U Biológicas (análise, não predição)
    pesos_t = np.linspace(1.5, 6.5, 300)
    risco_macho_simples = (1 / (1 + np.exp(-(ALPHA_0_MACHO + ALPHA_1 * (pesos_t - P_OPT_SIMPLES) ** 2)))) * 100
    risco_femea_gemeo   = (1 / (1 + np.exp(-(ALPHA_0_FEMEA + ALPHA_1 * (pesos_t - P_OPT_GEMEO) ** 2)))) * 100
    ax3.plot(pesos_t, risco_macho_simples, color='#2980b9', lw=2,
             label=f'Macho/Simples (P_opt={P_OPT_SIMPLES} kg)')
    ax3.plot(pesos_t, risco_femea_gemeo,   color='#8e44ad', lw=2, linestyle='--',
             label=f'Fêmea/Gêmeo (P_opt={P_OPT_GEMEO} kg)')
    sns.scatterplot(data=df, x='Peso_Nascer_Kg', y='Risco_Sistema_%',
                    alpha=0.3, ax=ax3, color='gray', label='População')
    ax3.set_title('Curvas U Biológicas por Sexo e Tipo de Parto')
    ax3.set_ylabel('Risco de Mortalidade (%)')
    ax3.legend(fontsize=8)

    # Gráfico 4: Cone de Crescimento Individual
    vetor_t, traj_alvo, sigma_t = dados_long['vetor_t'], dados_long['traj_alvo'], dados_long['sigma_t']
    ax4.plot(vetor_t, traj_alvo, '#2c3e50', lw=2,
             label=f"Alvo sistema (GMD={dados_long['gmd_esperado']:.3f} kg/d)")
    ax4.fill_between(vetor_t, traj_alvo, traj_alvo - 1.0 * sigma_t,
                     color='#2ecc71', alpha=0.20, label='Normal (Z ≥ −1.0)')
    ax4.fill_between(vetor_t, traj_alvo - 1.0 * sigma_t, traj_alvo - 2.0 * sigma_t,
                     color='#f1c40f', alpha=0.30, label='Atenção (−2.0 < Z < −1.0)')
    ax4.fill_between(vetor_t, traj_alvo - 2.0 * sigma_t, traj_alvo - 3.0 * sigma_t,
                     color='#e74c3c', alpha=0.20, label='Crítico (Z ≤ −2.0)')
    ax4.scatter(dados_long['dias_pesagem'], dados_long['pesagens_reais'],
                c=dados_long['cores_alerta'], s=80, edgecolors='black', zorder=5,
                label='Pesagens reais')
    ax4.set_title(
        f"Cone — Animal #{animal['ID_Parto']} "
        f"({animal['Sexo']}, {animal['Tipo_Parto']}, {animal['Paridade_Mae']})\n"
        f"P₀={animal['Peso_Nascer_Kg']:.2f} kg | "
        f"GMD real={animal['GMD_Real']:.3f} | GMD sistema={dados_long['gmd_esperado']:.3f} kg/d",
        fontsize=9, fontweight='bold'
    )
    ax4.set_ylabel('Peso Vivo (kg)')
    ax4.legend(loc='upper left', fontsize=8)

    # Gráfico 5: R(t) — Risco Longitudinal Dinâmico
    ax5.plot(dados_long['dias_pesagem'], dados_long['riscos_t'],
             color='#e74c3c', marker='o', lw=3, markersize=10,
             label='R(t) atualizado')
    ax5.axhline(dados_long['risco_nasc'], color='#7f8c8d', linestyle='--', lw=2,
                label=f"Risco no nascimento: {dados_long['risco_nasc']:.1f}%")
    for x, y, z in zip(dados_long['dias_pesagem'], dados_long['riscos_t'], dados_long['z_scores']):
        ax5.annotate(
            f"Z={z:.2f}", (x, y),
            textcoords="offset points", xytext=(0, 14),
            ha='center', fontsize=10, fontweight='bold'
        )
    ax5.set_title(
        f"R(t) = R_nasc · exp(k · max(0, −Z))   [k={K_DECAIMENTO}]  "
        f"— Z negativo eleva risco; Z positivo mantém baseline",
        fontsize=12
    )
    ax5.set_xlabel('Dias de Vida')
    ax5.set_ylabel('Risco Atualizado de Mortalidade (%)')
    ax5.set_ylim(0, max(max(dados_long['riscos_t']) * 1.25, dados_long['risco_nasc'] * 2))
    ax5.legend()

    plt.tight_layout()
    if SALVAR_FIGURA:
        plt.savefig(ARQUIVO_SAIDA, dpi=150, bbox_inches='tight')
        print(f"Figura salva: {ARQUIVO_SAIDA}")
    else:
        plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUÇÃO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Geração do rebanho e predição do sistema
    df_bio = gerar_rebanho_biologico(N_PARTOS, rng)
    df_res, p_opt_min_final, p_opt_max_final = simular_sistema_preditivo(df_bio)
    p_opt_sistema_final = (p_opt_min_final + p_opt_max_final) / 2.0

    # Animal para o cone (parâmetrizável)
    if ID_ANIMAL_CONE is None:
        animal_alvo = df_res.iloc[-1]
    else:
        animal_alvo = df_res[df_res['ID_Parto'] == ID_ANIMAL_CONE].iloc[0]

    dados_monitoramento = gerar_monitoramento_longitudinal(
        animal_alvo, p_opt_sistema_final, rng
    )

    #plotar_resultados(df_res, dados_monitoramento, animal_alvo)

    # ── [C5] Avaliação do modelo com métricas sklearn ────────────────────────
    auc   = roc_auc_score(df_res['Mortalidade_Observada'], df_res['Prob_Sistema'])
    brier = brier_score_loss(df_res['Mortalidade_Observada'], df_res['Prob_Sistema'])
    # Threshold: prevalência observada como ponto de corte natural
    thresh  = df_res['Mortalidade_Observada'].mean()
    pred_bin = (df_res['Prob_Sistema'] >= thresh).astype(int)
    f1    = f1_score(df_res['Mortalidade_Observada'], pred_bin, zero_division=0)

    print(f"\n{'─' * 60}")
    print(f"  ESPÉCIE          : {ESPECIE}")
    print(f"  Animais          : {N_PARTOS}")
    print(f"  Mortalidade real : {df_res['Mortalidade_Observada'].sum()} "
          f"({df_res['Mortalidade_Observada'].mean() * 100:.1f}%)")
    print(f"\n  Erro médio Cold Start : {df_res.loc[df_res['Fase'] == 'Cold Start', 'Erro_Predicao'].mean():.2f} pp")
    print(f"  Erro médio Calibrado  : {df_res.loc[df_res['Fase'] == 'Calibrado',   'Erro_Predicao'].mean():.2f} pp")
    print(f"\n  ── Avaliação do Modelo ──────────────────────────────")
    print(f"  AUC-ROC     : {auc:.4f}  (>0.75 = bom discriminador)")
    print(f"  Brier Score : {brier:.4f}  (0=perfeito, 0.25=aleatório)")
    print(f"  F1 @ {thresh:.0%}   : {f1:.4f}  (threshold = prevalência observada)")
    print(f"\n  P_opt sistema final: [{p_opt_min_final:.2f}, {p_opt_max_final:.2f}] kg")
    print(f"  Animal cone — ID #{animal_alvo['ID_Parto']}")
    print(f"    {animal_alvo['Sexo']} / {animal_alvo['Tipo_Parto']} / {animal_alvo['Paridade_Mae']}")
    print(f"    P₀={animal_alvo['Peso_Nascer_Kg']:.2f} kg | GMD real={animal_alvo['GMD_Real']:.4f}")
    print(f"    R(t): {[f'{r:.1f}%' for r in dados_monitoramento['riscos_t']]}")
    print(f"{'─' * 60}\n")
