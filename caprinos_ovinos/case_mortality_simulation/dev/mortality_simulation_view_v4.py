"""
Simulação Preditiva — Desenvolvimento Ponderal e Risco Neonatal
Espécie: Ovinos (Morada Nova / Santa Inês / Dorper)
Abordagem: Funcional com Validação Estatística
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, brier_score_loss
)

# ─── CONTROLE DE EXECUÇÃO E CONSTANTES BIOLÓGICAS ────────────────────────────
SALVAR_FIGURA = False
ARQUIVO_SAIDA = "output_v3_funcional.png"

# Espécie e contexto
ESPECIE = "Ovinos"

# Parâmetros Biológicos (Equação 1: Peso ao Nascer)
BETA_0 = 4.10
BETA_GEMEO = -0.65
BETA_TRIGEMEO = -1.40
BETA_FEMEA = -0.30
BETA_PRIMIPARA = -0.35
SIGMA_NASCIMENTO = 0.35

# Parâmetros de Crescimento (Equação 2: GMD)
GMD_BASE = 0.252 # tornar em função do sexo.
GAMMA = 0.02
P_REF_GMD = 4.0 # tornar em função do sexo.
PENALIDADE_MULTIPLO_GMD = -0.025

# Risco Epidemiológico Real (Equação 3)
ALPHA_0 = -2.5
ALPHA_1 = 1.2
P_OPT = 3.7 # mesma coisa que o p_ref_gmd, mas para a curva de risco. Tornar em função do sexo.

# Heterogeneidade do Cone
LAMBDA = 1.5
T_MAX = 90

# Crença Inicial do Sistema (Prior)
P_OPT_MIN_INICIAL = 3.0
P_OPT_MAX_INICIAL = 4.5

# Configurações da Simulação
N_PARTOS = 500
GATILHO_RETREINO = 50
PESO_MIN_PLAUSIVEL = 1.5
PESO_MAX_PLAUSIVEL = 6.0

# Proporções Populacionais !!procurar dados reais para ajustar!!
PROP_PARTO_SIMPLES = 0.70
PROP_FEMEAS = 0.50
PROP_PRIMIPARA = 0.20

# ═══════════════════════════════════════════════════════════════════════════════
# 1. GERAÇÃO DO MUNDO BIOLÓGICO
# ═══════════════════════════════════════════════════════════════════════════════
def gerar_rebanho_biologico(n_partos: int, rng: np.random.Generator) -> pd.DataFrame:
    """Gera a verdade oculta do rebanho, incluindo o desfecho real de mortalidade."""
    
    tipos_parto = rng.choice(
        ['Simples', 'Gemeo'], size=n_partos, 
        p=[PROP_PARTO_SIMPLES, 1 - PROP_PARTO_SIMPLES]
    )
    sexos = rng.choice(
        ['Macho', 'Femea'], size=n_partos, 
        p=[1 - PROP_FEMEAS, PROP_FEMEAS]
    )
    paridades = rng.choice(
        ['Multipara', 'Primipara'], size=n_partos, 
        p=[1 - PROP_PRIMIPARA, PROP_PRIMIPARA]
    )
    dias_vida = rng.integers(0, T_MAX + 1, size=n_partos)

    # Cálculo do Peso ao Nascer
    beta_parto_vec = np.where(tipos_parto == 'Gemeo', BETA_GEMEO, 0.0) # adicionar beta trigemeos
    beta_sexo_vec = np.where(sexos == 'Femea', BETA_FEMEA, 0.0)
    beta_matriz_vec = np.where(paridades == 'Primipara', BETA_PRIMIPARA, 0.0)
    ruido_nascimento = rng.normal(0, SIGMA_NASCIMENTO, size=n_partos)

    pesos_nascer = BETA_0 + beta_parto_vec + beta_sexo_vec + beta_matriz_vec + ruido_nascimento
    pesos_nascer = np.round(np.clip(pesos_nascer, 0.5, 8.0), 2)

    # Cálculo do GMD Real
    gmd_base_ajustado = GMD_BASE + GAMMA * (pesos_nascer - P_REF_GMD)
    gmd_real = np.where(
        tipos_parto == 'Gemeo',
        gmd_base_ajustado + PENALIDADE_MULTIPLO_GMD,
        gmd_base_ajustado
    ) # aplicar penalidade de femea: -0,023kg. deixar implicita na formula do gmd_base_ajustado

    # Risco Real e Mortalidade Observada (Amostragem Binomial)
    z_nasc = ALPHA_0 + ALPHA_1 * (pesos_nascer - P_OPT) ** 2
    prob_mortalidade_real = 1 / (1 + np.exp(-z_nasc))
    mortalidade_observada = rng.binomial(1, prob_mortalidade_real)

    peso_atual = pesos_nascer + gmd_real * dias_vida

    return pd.DataFrame({
        'ID_Parto': range(1, n_partos + 1),
        'Tipo_Parto': tipos_parto,
        'Sexo': sexos,
        'Paridade_Mae': paridades,
        'Peso_Nascer_Kg': pesos_nascer,
        'Idade_Dias': dias_vida,
        'GMD_Real': np.round(gmd_real, 4),
        'Peso_Atual': np.round(peso_atual, 2),
        'Prob_Mortalidade_Real': prob_mortalidade_real,
        'Risco_Real_%': prob_mortalidade_real * 100,
        'Mortalidade_Observada': mortalidade_observada # Desfecho 0 ou 1
    })

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SIMULAÇÃO DO SISTEMA PREDITIVO
# ═══════════════════════════════════════════════════════════════════════════════
def simular_sistema_preditivo(df: pd.DataFrame) -> pd.DataFrame:
    """Passa os dados pelo sistema preditivo, simulando a chegada temporal e o retreino."""
    df_sim = df.copy()
    
    p_opt_min_sistema = P_OPT_MIN_INICIAL
    p_opt_max_sistema = P_OPT_MAX_INICIAL
    retreino_feito = False

    fases, risco_sistema_lista, prob_sistema_lista = [], [], []
    p_opt_min_memoria, p_opt_max_memoria = [], []

    for index, linha in df_sim.iterrows():
        # Retreino automático
        if index >= GATILHO_RETREINO and not retreino_feito:
            amostra_historica = df_sim.loc[0:(index - 1), 'Peso_Nascer_Kg']
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

        # Previsão do sistema
        p_opt_sistema = (p_opt_min_sistema + p_opt_max_sistema) / 2.0
        z_sistema = ALPHA_0 + ALPHA_1 * (linha['Peso_Nascer_Kg'] - p_opt_sistema) ** 2
        prob_sistema = 1 / (1 + np.exp(-z_sistema))
        
        prob_sistema_lista.append(prob_sistema)
        risco_sistema_lista.append(prob_sistema * 100)

    df_sim['Fase'] = fases
    df_sim['P_Opt_Min_Sistema'] = p_opt_min_memoria
    df_sim['P_Opt_Max_Sistema'] = p_opt_max_memoria
    df_sim['Prob_Sistema'] = prob_sistema_lista
    df_sim['Risco_Sistema_%'] = risco_sistema_lista
    df_sim['Erro_Predicao'] = df_sim['Risco_Sistema_%'] - df_sim['Risco_Real_%']
    
    return df_sim

# ═══════════════════════════════════════════════════════════════════════════════
# 3. VALIDAÇÃO ESTATÍSTICA
# ═══════════════════════════════════════════════════════════════════════════════
def calcular_metricas(df: pd.DataFrame) -> dict:
    """Calcula as métricas de validação do modelo comparando previsão vs realidade."""
    y_true = df['Mortalidade_Observada']
    y_prob = df['Prob_Sistema']
    y_pred = (y_prob >= 0.5).astype(int) # Threshold padrão de 50%
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan # Caso só exista uma classe na amostra gerada
        
    metricas = {
        'total_animais': len(df),
        'obitos_observados': y_true.sum(),
        'obitos_previstos': y_pred.sum(),
        'matriz_confusao': confusion_matrix(y_true, y_pred),
        'acuracia': accuracy_score(y_true, y_pred),
        'precisao': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': auc,
        'brier_score': brier_score_loss(y_true, y_prob),
        'erro_medio_cold_start': df.loc[df['Fase'] == 'Cold Start', 'Erro_Predicao'].mean(),
        'erro_medio_calibrado': df.loc[df['Fase'] == 'Calibrado', 'Erro_Predicao'].mean()
    }
    return metricas

# ═══════════════════════════════════════════════════════════════════════════════
# 4. GERAÇÃO DO CONE DE CRESCIMENTO (Para um animal específico)
# ═══════════════════════════════════════════════════════════════════════════════
def gerar_cone(animal: pd.Series, p_opt_final: float, rng: np.random.Generator) -> dict:
    """Simula a trajetória de um animal específico e suas pesagens ruidosas."""
    peso_nascer = animal['Peso_Nascer_Kg']
    gmd_real = animal['GMD_Real']

    # Expectativa do sistema
    gmd_esperado = GMD_BASE + GAMMA * (peso_nascer - p_opt_final)
    if animal['Tipo_Parto'] == 'Gemeo':
        gmd_esperado += PENALIDADE_MULTIPLO_GMD

    vetor_t = np.arange(0, T_MAX + 1)
    traj_alvo = peso_nascer + gmd_esperado * vetor_t
    sigma_t = SIGMA_NASCIMENTO + LAMBDA * (vetor_t / T_MAX) ** 2

    # Pesagens ruidosas (Evento sanitário no dia 60)
    dias_pesagem = np.array([15, 30, 45, 60, 75, 90])
    pesos_bio_dias = peso_nascer + gmd_real * dias_pesagem
    
    ruido_sanitario = np.array([
        rng.normal(0.0,  0.15),  # dia 15
        rng.normal(0.0,  0.20),  # dia 30
        rng.normal(-0.3, 0.15),  # dia 45 (leve queda)
        rng.normal(-0.7, 0.15),  # dia 60 (evento)
        rng.normal(-0.2, 0.15),  # dia 75 (recuperação)
        rng.normal(0.0,  0.20),  # dia 90
    ])
    pesagens_reais = pesos_bio_dias + ruido_sanitario

    cores_alerta = []
    for t, peso_medido in zip(dias_pesagem, pesagens_reais):
        peso_alvo_dia = peso_nascer + gmd_esperado * t
        sigma_dia = SIGMA_NASCIMENTO + LAMBDA * (t / T_MAX) ** 2
        z_score = (peso_medido - peso_alvo_dia) / sigma_dia
        if z_score >= -1.0: cores_alerta.append('#2ecc71')   # Normal
        elif z_score >= -2.0: cores_alerta.append('#f1c40f') # Atenção
        else: cores_alerta.append('#e74c3c')                 # Crítico

    return {
        'vetor_t': vetor_t, 'traj_alvo': traj_alvo, 'sigma_t': sigma_t,
        'dias_pesagem': dias_pesagem, 'pesagens_reais': pesagens_reais,
        'cores_alerta': cores_alerta, 'gmd_esperado': gmd_esperado
    }

# ═══════════════════════════════════════════════════════════════════════════════
# 5. GERAÇÃO DOS GRÁFICOS
# ═══════════════════════════════════════════════════════════════════════════════
def plotar_resultados(df: pd.DataFrame, dados_cone: dict, animal: pd.Series):
    sns.set_theme(style="whitegrid")
    PALETTE_FASES = {'Cold Start': '#e74c3c', 'Calibrado': '#2ecc71'}

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Análise Preditiva e Retreino Automático — {ESPECIE}', fontsize=18, fontweight='bold', y=0.98)

    # Gráfico 1: Erro de Predição
    ax1 = axs[0, 0]
    ax1.axvline(x=GATILHO_RETREINO, color='black', linestyle='--', lw=1.2, label=f'Retreino (n={GATILHO_RETREINO})')
    sns.lineplot(data=df, x='ID_Parto', y='Erro_Predicao', hue='Fase', palette=PALETTE_FASES, ax=ax1)
    ax1.set_title('Queda do Erro de Predição após Retreino')
    ax1.set_xlabel('ID Parto')
    ax1.set_ylabel('Erro (pp)')
    ax1.legend()

    # Gráfico 2: Ajuste da Faixa Ótima
    ax2 = axs[0, 1]
    ax2.axvline(x=GATILHO_RETREINO, color='black', linestyle='--', lw=1.2)
    ax2.plot(df['ID_Parto'], df['P_Opt_Min_Sistema'], color='#3498db', lw=1.8, label='Limites do sistema')
    ax2.plot(df['ID_Parto'], df['P_Opt_Max_Sistema'], color='#3498db', lw=1.8)
    ax2.axhline(P_OPT, color='#e67e22', linestyle=':', lw=1.5, label=f'P_opt biológico = {P_OPT} kg')
    ax2.set_title('Convergência da Crença do Sistema')
    ax2.set_xlabel('ID Parto')
    ax2.set_ylabel('Peso (kg)')
    ax2.legend(fontsize=8)

    # Gráfico 3: Curva Logística e Desfechos
    ax3 = axs[1, 0]
    pesos_t = np.linspace(1.5, 6.0, 300)
    risco_t = (1 / (1 + np.exp(-(ALPHA_0 + ALPHA_1 * (pesos_t - P_OPT) ** 2)))) * 100
    sns.scatterplot(data=df, x='Peso_Nascer_Kg', y='Risco_Sistema_%', hue='Fase', palette=PALETTE_FASES, alpha=0.6, ax=ax3)
    ax3.plot(pesos_t, risco_t, 'k:', lw=2, label=f'Curva Biológica Real')
    ax3.axvline(P_OPT, color='#e67e22', linestyle='--', lw=1, alpha=0.6)
    ax3.set_title('Risco Epidemiológico e Probabilidades Atribuídas')
    ax3.set_xlabel('Peso ao Nascer (kg)')
    ax3.set_ylabel('Risco Previsto pelo Sistema (%)')
    ax3.legend(fontsize=8)

    # Gráfico 4: Cone de Crescimento
    ax4 = axs[1, 1]
    vetor_t, traj_alvo, sigma_t = dados_cone['vetor_t'], dados_cone['traj_alvo'], dados_cone['sigma_t']
    
    ax4.plot(vetor_t, traj_alvo, '#2c3e50', lw=2, label=f"Alvo sistema (GMD={dados_cone['gmd_esperado']:.3f})")
    ax4.fill_between(vetor_t, traj_alvo, traj_alvo - 1.0*sigma_t, color='#2ecc71', alpha=0.20, label='Normal')
    ax4.fill_between(vetor_t, traj_alvo - 1.0*sigma_t, traj_alvo - 2.0*sigma_t, color='#f1c40f', alpha=0.30, label='Atenção')
    ax4.fill_between(vetor_t, traj_alvo - 2.0*sigma_t, traj_alvo - 3.0*sigma_t, color='#e74c3c', alpha=0.20, label='Crítico')
    ax4.scatter(dados_cone['dias_pesagem'], dados_cone['pesagens_reais'], c=dados_cone['cores_alerta'], s=80, edgecolors='black', zorder=5)
    
    ax4.set_title(f"Trajetória — Animal #{animal['ID_Parto']} | {animal['Sexo']}\nP₀={animal['Peso_Nascer_Kg']}kg | GMD Real={animal['GMD_Real']:.3f}", fontsize=10, fontweight='bold')
    ax4.set_xlabel('Dias de Vida')
    ax4.set_ylabel('Peso Vivo (kg)')
    ax4.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    if SALVAR_FIGURA:
        plt.savefig(ARQUIVO_SAIDA, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 6. EXECUÇÃO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Instanciando o gerador global unificado
    gerador_rng = np.random.default_rng(42)

    # Pipeline Funcional
    df_bio = gerar_rebanho_biologico(N_PARTOS, gerador_rng)
    df_resultado = simular_sistema_preditivo(df_bio)
    metricas = calcular_metricas(df_resultado)

    # Selecionando o último animal para o cone
    animal_alvo = df_resultado.iloc[-1]
    p_opt_final = (df_resultado['P_Opt_Min_Sistema'].iloc[-1] + df_resultado['P_Opt_Max_Sistema'].iloc[-1]) / 2.0
    cone_res = gerar_cone(animal_alvo, p_opt_final, gerador_rng)

    # Resumo Estatístico Final
    print(f"\n{'─'*60}")
    print(f" RESUMO ESTATÍSTICO E VALIDAÇÃO — {ESPECIE}")
    print(f"{'─'*60}")
    print(f" Animais simulados     : {metricas['total_animais']}")
    print(f" Mortalidade Simulada  : {metricas['obitos_observados']} óbitos (Realidade)")
    print(f" Mortalidade Prevista  : {metricas['obitos_previstos']} óbitos (Sistema >= 50%)")
    print("\n MÉTRICAS DE AVALIAÇÃO:")
    print(f" ROC-AUC               : {metricas['roc_auc']:.4f}")
    print(f" Brier Score           : {metricas['brier_score']:.4f}")
    print(f" Acurácia              : {metricas['acuracia']:.4f}")
    print(f" F1-Score              : {metricas['f1_score']:.4f}")
    
    print("\n MATRIZ DE CONFUSÃO:")
    print(f" Verdadeiros Negativos : {metricas['matriz_confusao'][0][0]}")
    print(f" Falsos Positivos      : {metricas['matriz_confusao'][0][1]}")
    print(f" Falsos Negativos      : {metricas['matriz_confusao'][1][0]}")
    print(f" Verdadeiros Positivos : {metricas['matriz_confusao'][1][1]}")

    print("\n EVOLUÇÃO DO ERRO DO SISTEMA:")
    print(f" Erro médio antes do retreino (Cold Start) : {metricas['erro_medio_cold_start']:+.2f} pp")
    print(f" Erro médio após retreino (Calibrado)      : {metricas['erro_medio_calibrado']:+.2f} pp")
    print(f"{'─'*60}\n")

    # Plot
    plotar_resultados(df_resultado, cone_res, animal_alvo)