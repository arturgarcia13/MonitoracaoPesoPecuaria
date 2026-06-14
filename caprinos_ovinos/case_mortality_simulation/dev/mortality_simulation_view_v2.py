"""
Simulação Preditiva — Desenvolvimento Ponderal e Risco Neonatal
Espécie: Ovinos (Morada Nova / Santa Inês / Dorper)

Fundamentação teórica:
  Freitas et al. (1980)  — peso médio Morada Nova: 3.1 kg
  McMillan et al. (1983) — faixa ótima de sobrevivência: 3.3–4.1 kg (NZ)
  Everts et al. (1985)   — efeito sexo: fêmeas −0.19 kg
  Gardner et al.         — efeitos parto (gêmeo −0.692, trigêmeo −1.40),
                           sexo (−0.363), paridade (primípara −0.351 kg)
  Hatcher (2009)         — parto simples 4.00 kg, gêmeo 3.35 kg
  Sarmento et al. (2010) — variância heterogênea crescente com a idade
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ─── CONTROLE DE EXECUÇÃO ────────────────────────────────────────────────────
SALVAR_FIGURA = False          # False → plt.show() interativo
ARQUIVO_SAIDA = "output_v2.png"

sns.set_theme(style="whitegrid")
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONSTANTES DO MODELO — todas rastreadas à literatura
# ═══════════════════════════════════════════════════════════════════════════════
# Espécie e contexto
ESPECIE = "Ovinos"          # Dorper / Santa Inês / Morada Nova

# ── Equação 1: Peso ao Nascer Esperado (P0) ──────────────────────────────────
# P0 = BETA_0 + beta_parto + beta_sexo + beta_matriz + η
# β0 = 4.10 kg → referência: Macho, Parto Simples, Mãe Multípara
# Hatcher (2009): simples = 4.00 kg; regressao.md seção 1 fixa 4.10 kg
BETA_0 = 4.10

# Efeito do tipo de parto — Hatcher (2009) e Gardner et al.
BETA_GEMEO    = -0.65   # kg — Hatcher: 4.00 vs 3.35 kg
BETA_TRIGEMEO = -1.40   # kg — Gardner et al.: decréscimo progressivo

# Efeito de sexo — média consensual Everts (-0.19), Gardner (-0.363), Medeiros (-0.32)
BETA_FEMEA = -0.30      # kg

# Efeito da paridade da matriz — Gardner et al.: maior incremento 1ª→2ª gestação
BETA_PRIMIPARA = -0.35  # kg

# Desvio padrão do erro aleatório no nascimento (regressao.md seção 3.2)
SIGMA_NASCIMENTO = 0.35 # kg

# ── Equação 2: GMD e Trajetória de Peso ──────────────────────────────────────
# Pt = P0 + GMD * t
# GMD = GMD_BASE + γ * (P0 - P_REF_GMD)
# Ovinos corte (Dorper/Santa Inês): 0.252–0.292 kg/dia (regressao.md)
GMD_BASE     = 0.252    # kg/dia — limite inferior para Morada Nova
GAMMA        = 0.02     # sensibilidade GMD ao P0 (regressao.md seção 3.2)
P_REF_GMD    = 4.0      # kg — ponto neutro do ajuste GMD (regressao.md, corrigido)
PENALIDADE_MULTIPLO_GMD = -0.025  # kg/dia — regressao.md seção 3.2

# ── Equação 3: Risco Logístico de Mortalidade Neonatal ───────────────────────
# z = ALPHA_0 + ALPHA_1 * (P0 - P_OPT)²
# P(Y=1) = 1 / (1 + e^-z)
# Curva U com mínimo em P_OPT — regressao.md seção 3 / Cenário B
ALPHA_0  = -2.5         # intercepto logístico (regressao.md Cenário B)
ALPHA_1  =  1.2         # coeficiente quadrático (regressao.md Cenário B)
P_OPT    =  4.0         # kg — peso de mínima mortalidade (McMillan et al. 1983)

# ── Heterogeneidade Residual do Cone (Sarmento et al. 2010) ──────────────────
# σ(t) = SIGMA_NASCIMENTO + LAMBDA * (t / T_MAX)²
# Simplificação da abordagem FL5 de Legendre, válida para prototipagem
LAMBDA = 1.5            # fator de escala biométrica (regressao.md seção 3.2)
T_MAX  = 90             # dias (desmama) — regressao.md seção 3.2

# ── Crença Inicial do Sistema (antes do retreino) ────────────────────────────
# McMillan et al. (1983): faixa de mínima mortalidade estimada em NZ
# Representa o prior de um sistema sem dados locais
P_OPT_MIN_INICIAL = 3.3  # kg — McMillan et al. 1983
P_OPT_MAX_INICIAL = 4.1  # kg — McMillan et al. 1983

# ── Parâmetros de Simulação e Retreino ───────────────────────────────────────
N_PARTOS        = 150
GATILHO_RETREINO = 50   # retreino ao atingir 50 registros acumulados
PESO_MIN_PLAUSIVEL = 1.5  # kg — filtro de outliers na amostra de retreino
PESO_MAX_PLAUSIVEL = 6.0  # kg

# ── Proporções populacionais (para geração realista do rebanho) ───────────────
PROP_PARTO_SIMPLES = 0.70
PROP_FEMEAS        = 0.50
PROP_PRIMIPARA     = 0.20  # matrizes de primeiro parto no rebanho

# ═══════════════════════════════════════════════════════════════════════════════
# 2. GERAÇÃO DO REBANHO SIMULADO (Verdade Biológica)
# ═══════════════════════════════════════════════════════════════════════════════
n = N_PARTOS

# Variáveis independentes
tipos_parto = np.random.choice(
    ['Simples', 'Gemeo'],
    size=n,
    p=[PROP_PARTO_SIMPLES, 1 - PROP_PARTO_SIMPLES]
)
sexos = np.random.choice(
    ['Macho', 'Femea'],
    size=n,
    p=[1 - PROP_FEMEAS, PROP_FEMEAS]
)
paridades = np.random.choice(
    ['Multipara', 'Primipara'],
    size=n,
    p=[1 - PROP_PRIMIPARA, PROP_PRIMIPARA]
)
dias_vida = np.random.randint(0, T_MAX + 1, size=n)

# ── Equação 1: P0 com efeitos fixos + ruído aleatório ─────────────────────────
beta_parto_vec  = np.where(tipos_parto == 'Gemeo', BETA_GEMEO, 0.0)
beta_sexo_vec   = np.where(sexos == 'Femea', BETA_FEMEA, 0.0)
beta_matriz_vec = np.where(paridades == 'Primipara', BETA_PRIMIPARA, 0.0)
ruido_nascimento = np.random.normal(0, SIGMA_NASCIMENTO, size=n)

pesos_nascer = (
    BETA_0
    + beta_parto_vec
    + beta_sexo_vec
    + beta_matriz_vec
    + ruido_nascimento
)
pesos_nascer = np.round(np.clip(pesos_nascer, 0.5, 8.0), 2)

# ── Equação 2: GMD real (biologia gravada uma única vez) ─────────────────────
gmd_base_ajustado = GMD_BASE + GAMMA * (pesos_nascer - P_REF_GMD)
gmd_real = np.where(
    tipos_parto == 'Gemeo',
    gmd_base_ajustado + PENALIDADE_MULTIPLO_GMD,
    gmd_base_ajustado
)

# ── DataFrame do rebanho ─────────────────────────────────────────────────────
df = pd.DataFrame({
    'ID_Parto'      : range(1, n + 1),
    'Tipo_Parto'    : tipos_parto,
    'Sexo'          : sexos,
    'Paridade_Mae'  : paridades,
    'Peso_Nascer_Kg': pesos_nascer,
    'Idade_Dias'    : dias_vida,
    'GMD_Real'      : np.round(gmd_real, 4),
    'Peso_Atual'    : np.round(pesos_nascer + gmd_real * dias_vida, 2),
})

# ═══════════════════════════════════════════════════════════════════════════════
# 3. LOOP DE PROCESSAMENTO COM RETREINO AUTOMÁTICO
# ═══════════════════════════════════════════════════════════════════════════════
p_opt_min_sistema = P_OPT_MIN_INICIAL
p_opt_max_sistema = P_OPT_MAX_INICIAL
retreino_feito    = False  # garante retreino único e robusto

fases               = []
risco_sistema_lista = []
p_opt_min_memoria   = []
p_opt_max_memoria   = []

for index, linha in df.iterrows():

    # ── Retreino automático ao atingir o gatilho ──────────────────────────────
    # Usa '>=' para garantir que qualquer refatoração de índice não silencia o retreino
    if index >= GATILHO_RETREINO and not retreino_feito:
        amostra_historica = df.loc[0:(index - 1), 'Peso_Nascer_Kg']
        amostra_limpa = amostra_historica[
            (amostra_historica >= PESO_MIN_PLAUSIVEL) &
            (amostra_historica <= PESO_MAX_PLAUSIVEL)
        ]
        if not amostra_limpa.empty:
            p_opt_min_sistema = np.round(amostra_limpa.quantile(0.25), 2)
            p_opt_max_sistema = np.round(amostra_limpa.quantile(0.75), 2)
        retreino_feito = True

    fases.append('Cold Start' if not retreino_feito else 'Calibrado')
    p_opt_min_memoria.append(p_opt_min_sistema)
    p_opt_max_memoria.append(p_opt_max_sistema)

    # ── Equação 3: Risco logístico (usando crença atual do sistema) ───────────
    # O sistema usa seu P_OPT aprendido (centro do platô calibrado)
    # como substituto do P_OPT biológico de 4.0 kg
    p_opt_sistema = (p_opt_min_sistema + p_opt_max_sistema) / 2.0
    z_sistema = ALPHA_0 + ALPHA_1 * (linha['Peso_Nascer_Kg'] - p_opt_sistema) ** 2
    risco_sistema_lista.append((1 / (1 + np.exp(-z_sistema))) * 100)

df['Fase']              = fases
df['P_Opt_Min_Sistema'] = p_opt_min_memoria
df['P_Opt_Max_Sistema'] = p_opt_max_memoria
df['Risco_Sistema_%']   = risco_sistema_lista

# ── Risco real (curva U do documento, Popt biológico = 4.0 kg) ───────────────
z_real = ALPHA_0 + ALPHA_1 * (df['Peso_Nascer_Kg'] - P_OPT) ** 2
df['Risco_Real_%']   = (1 / (1 + np.exp(-z_real))) * 100
df['Erro_Predicao']  = df['Risco_Sistema_%'] - df['Risco_Real_%']

# ═══════════════════════════════════════════════════════════════════════════════
# 4. CONE DE CRESCIMENTO — animal parametrizável
# ═══════════════════════════════════════════════════════════════════════════════
# Escolha do animal a visualizar (último por padrão; altere o ID aqui)
ID_ANIMAL_CONE = df['ID_Parto'].iloc[-1]
animal = df[df['ID_Parto'] == ID_ANIMAL_CONE].iloc[0]

peso_nascer_indiv = animal['Peso_Nascer_Kg']
gmd_real_indiv    = animal['GMD_Real']

# GMD esperado pelo sistema (usa P_OPT do sistema calibrado como referência)
p_opt_sistema_final = (p_opt_min_sistema + p_opt_max_sistema) / 2.0
gmd_esperado = GMD_BASE + GAMMA * (peso_nascer_indiv - p_opt_sistema_final)
if animal['Tipo_Parto'] == 'Gemeo':
    gmd_esperado += PENALIDADE_MULTIPLO_GMD

# Trajetória alvo e bandas de confiança
vetor_t       = np.arange(0, T_MAX + 1)
traj_alvo     = peso_nascer_indiv + gmd_esperado * vetor_t
sigma_t       = SIGMA_NASCIMENTO + LAMBDA * (vetor_t / T_MAX) ** 2  # Sarmento et al. 2010

limite_atencao = traj_alvo - 1.0 * sigma_t  # Z < -1.0
limite_critico = traj_alvo - 2.0 * sigma_t  # Z < -2.0
limite_alarme  = traj_alvo - 3.0 * sigma_t  # Z < -3.0 (zona visual)

# ── Pesagens no curral: biologia real + ruído sanitário plausível ────────────
# Ruído modelado como gaussiano com média zero; dias 45–60 simulam
# evento sanitário (queda de ~0.5–0.8 kg, dentro de 1–2 desvios típicos)
dias_pesagem  = np.array([15, 30, 45, 60, 75, 90])
pesos_bio_dias = peso_nascer_indiv + gmd_real_indiv * dias_pesagem
sigma_nos_dias = SIGMA_NASCIMENTO + LAMBDA * (dias_pesagem / T_MAX) ** 2

# Ruído totalmente estocástico; evento sanitário codificado via desvio controlado
np.random.seed(42)   # seed local para reprodutibilidade das pesagens
ruido_sanitario = np.array([
    np.random.normal(0.0,  0.15),   # dia 15 — normal
    np.random.normal(0.0,  0.20),   # dia 30 — normal
    np.random.normal(-0.3, 0.15),   # dia 45 — leve queda (~0.5σ)
    np.random.normal(-0.7, 0.15),   # dia 60 — evento sanitário (~0.7σ)
    np.random.normal(-0.2, 0.15),   # dia 75 — recuperação parcial
    np.random.normal(0.0,  0.20),   # dia 90 — normal
])
pesagens_reais = pesos_bio_dias + ruido_sanitario

# Z-score de cada pesagem e cor de alerta correspondente
PALETTE_ALERTA = {
    'normal'   : '#2ecc71',
    'atencao'  : '#f1c40f',
    'critico'  : '#e74c3c',
}
cores_alerta = []
for t, peso_medido in zip(dias_pesagem, pesagens_reais):
    peso_alvo_dia = peso_nascer_indiv + gmd_esperado * t
    sigma_dia     = SIGMA_NASCIMENTO + LAMBDA * (t / T_MAX) ** 2
    z_score       = (peso_medido - peso_alvo_dia) / sigma_dia
    if z_score >= -1.0:
        cores_alerta.append(PALETTE_ALERTA['normal'])
    elif z_score >= -2.0:
        cores_alerta.append(PALETTE_ALERTA['atencao'])
    else:
        cores_alerta.append(PALETTE_ALERTA['critico'])

# ═══════════════════════════════════════════════════════════════════════════════
# 5. GERAÇÃO DOS GRÁFICOS
# ═══════════════════════════════════════════════════════════════════════════════
PALETTE_FASES = {'Cold Start': '#e74c3c', 'Calibrado': '#2ecc71'}

fig, axs = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    f'Análise Preditiva e Retreino Automático — {ESPECIE}',
    fontsize=18, fontweight='bold', y=0.98
)

# ── Gráfico 1: Queda do Erro de Predição ─────────────────────────────────────
ax1 = axs[0, 0]
ax1.axvline(x=GATILHO_RETREINO, color='black', linestyle='--', lw=1.2,
            label=f'Retreino (n={GATILHO_RETREINO})')
sns.lineplot(
    data=df, x='ID_Parto', y='Erro_Predicao',
    hue='Fase', palette=PALETTE_FASES, ax=ax1
)
ax1.set_title('Queda do Erro de Predição após Retreino')
ax1.set_xlabel('ID Parto')
ax1.set_ylabel('Erro (pp)')
ax1.legend()

# ── Gráfico 2: Ajuste da Faixa Ótima à Realidade Local ───────────────────────
ax2 = axs[0, 1]
ax2.axvline(x=GATILHO_RETREINO, color='black', linestyle='--', lw=1.2)
ax2.plot(df['ID_Parto'], df['P_Opt_Min_Sistema'], color='#3498db',
         lw=1.8, label='Limites aprendidos pelo sistema')
ax2.plot(df['ID_Parto'], df['P_Opt_Max_Sistema'], color='#3498db', lw=1.8)
# Faixa de mínima mortalidade do documento (McMillan et al. 1983)
ax2.axhline(P_OPT, color='#e67e22', linestyle=':', lw=1.5,
            label=f'P_opt biológico = {P_OPT} kg (McMillan 1983)')
ax2.set_title('Convergência da Crença do Sistema à Realidade Local')
ax2.set_xlabel('ID Parto')
ax2.set_ylabel('Peso (kg)')
ax2.legend(fontsize=8)

# ── Gráfico 3: Curva Logística U (alinhada ao documento) ─────────────────────
ax3 = axs[1, 0]
pesos_t  = np.linspace(1.5, 6.0, 300)
z_curva  = ALPHA_0 + ALPHA_1 * (pesos_t - P_OPT) ** 2
risco_t  = (1 / (1 + np.exp(-z_curva))) * 100

sns.scatterplot(
    data=df, x='Peso_Nascer_Kg', y='Risco_Sistema_%',
    hue='Fase', palette=PALETTE_FASES, alpha=0.6, ax=ax3
)
ax3.plot(pesos_t, risco_t, 'k:', lw=2, label=f'Curva U (P_opt={P_OPT} kg)')
ax3.axvline(P_OPT, color='#e67e22', linestyle='--', lw=1, alpha=0.6)
ax3.set_title(f'Risco Epidemiológico Neonatal — Curva U (α₀={ALPHA_0}, α₁={ALPHA_1})')
ax3.set_xlabel('Peso ao Nascer (kg)')
ax3.set_ylabel('Risco de Mortalidade (%)')
ax3.legend(fontsize=8)

# ── Gráfico 4: Cone de Crescimento Individual ─────────────────────────────────
ax4 = axs[1, 1]
ax4.plot(vetor_t, traj_alvo, '#2c3e50', lw=2,
         label=f'Alvo sistema (GMD={gmd_esperado:.3f} kg/d)')
ax4.fill_between(vetor_t, traj_alvo, limite_atencao,
                 color='#2ecc71', alpha=0.20, label='Normal (Z ≥ −1.0)')
ax4.fill_between(vetor_t, limite_atencao, limite_critico,
                 color='#f1c40f', alpha=0.30, label='Atenção (−2.0 < Z < −1.0)')
ax4.fill_between(vetor_t, limite_critico, limite_alarme,
                 color='#e74c3c', alpha=0.20, label='Crítico (Z ≤ −2.0)')  # 3σ
ax4.scatter(dias_pesagem, pesagens_reais,
            c=cores_alerta, s=80, zorder=5, edgecolors='black',
            label='Pesagens reais')
ax4.set_title(
    f'Trajetória — Animal #{ID_ANIMAL_CONE} | '
    f'{animal["Sexo"]}, {animal["Tipo_Parto"]}, {animal["Paridade_Mae"]}\n'
    f'P₀={peso_nascer_indiv:.2f} kg | GMD real={gmd_real_indiv:.3f} | '
    f'GMD esperado={gmd_esperado:.3f} kg/d',
    fontsize=10, fontweight='bold'
)
ax4.set_xlabel('Dias de Vida')
ax4.set_ylabel('Peso Vivo (kg)')
ax4.legend(loc='upper left', fontsize=9)

plt.tight_layout()

if SALVAR_FIGURA:
    plt.savefig(ARQUIVO_SAIDA, dpi=150, bbox_inches='tight')
    print(f"Figura salva em: {ARQUIVO_SAIDA}")
else:
    plt.show()

plt.close()

# ── Resumo diagnóstico ────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"  ESPÉCIE           : {ESPECIE}")
print(f"  Animais simulados : {N_PARTOS}")
print(f"  Retreino em       : animal #{GATILHO_RETREINO}")
print(f"  P_Opt_Min após retreino: {p_opt_min_sistema:.2f} kg")
print(f"  P_Opt_Max após retreino: {p_opt_max_sistema:.2f} kg")
print(f"\n  Erro médio (Cold Start): "
      f"{df.loc[df['Fase']=='Cold Start','Erro_Predicao'].mean():.2f} pp")
print(f"  Erro médio (Calibrado) : "
      f"{df.loc[df['Fase']=='Calibrado','Erro_Predicao'].mean():.2f} pp")
print(f"\n  Animal cone — ID #{ID_ANIMAL_CONE}")
print(f"    Sexo/Parto/Paridade: {animal['Sexo']} / "
      f"{animal['Tipo_Parto']} / {animal['Paridade_Mae']}")
print(f"    P0={peso_nascer_indiv:.2f} kg | "
      f"GMD real={gmd_real_indiv:.4f} | GMD esperado={gmd_esperado:.4f}")
print(f"    Alertas nas pesagens: {cores_alerta}")
print(f"{'─'*60}\n")