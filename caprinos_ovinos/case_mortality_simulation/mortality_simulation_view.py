import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração de estilo
sns.set_theme(style="whitegrid")

# ==========================================
# 1. MOTOR DE SIMULAÇÃO E GERAÇÃO DA VERDADE BIOLÓGICA
# ==========================================
np.random.seed(42)
n_partos = 150
gmd_base = 0.252 # Base unificada no topo do código

# A Verdade Biológica (Território)
peso_medio_real = 3.1 
p_opt_min_bio = 2.7
p_opt_max_bio = 3.5
centro_plato_bio = (p_opt_min_bio + p_opt_max_bio) / 2

# A Crença Inicial do Sistema (Mapa)
p_opt_min_sistema = 3.3
p_opt_max_sistema = 4.1
gatilho_retreino = 50

# Geração dos animais
pesos_aferidos = np.random.normal(loc=peso_medio_real, scale=0.4, size=n_partos)
tipos_parto = np.random.choice(['Simples', 'Múltiplo'], size=n_partos, p=[0.7, 0.3])
dias_vida = np.random.randint(0, 91, size=n_partos)

# Cálculo do GMD REAL (Biologia) - Feito apenas UMA vez para gerar o animal
gmd_ajustado_bio = gmd_base + 0.02 * (pesos_aferidos - centro_plato_bio)
gmd_final_bio = np.where(tipos_parto == 'Múltiplo', gmd_ajustado_bio - 0.025, gmd_ajustado_bio)

df = pd.DataFrame({
    'ID_Parto': range(1, n_partos + 1), 
    'Peso_Nascer_Kg': np.round(pesos_aferidos, 2),
    'Tipo_Parto': tipos_parto,
    'Idade_Dias': dias_vida,
    'Peso_atual': pesos_aferidos + (gmd_final_bio * dias_vida),
    'GMD_Real': gmd_final_bio # Salvo permanentemente
})

# ==========================================
# 2. LOOP DE PROCESSAMENTO (O Aprendizado da Máquina)
# ==========================================
fases, risco_sistema_lista, p_opt_min_memoria, p_opt_max_memoria = [], [], [], []

for index, linha in df.iterrows():
    peso_atual = linha['Peso_Nascer_Kg']
    
    if index == gatilho_retreino:
        amostra_historica = df.loc[0:(gatilho_retreino-1), 'Peso_Nascer_Kg']
        amostra_limpa = amostra_historica[(amostra_historica >= 2.0) & (amostra_historica <= 5.5)]
        if not amostra_limpa.empty:
            p_opt_min_sistema = np.round(amostra_limpa.quantile(0.25), 2)
            p_opt_max_sistema = np.round(amostra_limpa.quantile(0.75), 2)
            
    fases.append('Cold Start' if index < gatilho_retreino else 'Calibrado')
    p_opt_min_memoria.append(p_opt_min_sistema)
    p_opt_max_memoria.append(p_opt_max_sistema)
    
    # Lógica da Zona Morta
    if peso_atual < p_opt_min_sistema: penalidade_sistema = peso_atual - p_opt_min_sistema
    elif peso_atual > p_opt_max_sistema: penalidade_sistema = peso_atual - p_opt_max_sistema
    else: penalidade_sistema = 0 
        
    z_sistema = -2.5 + 1.2 * (penalidade_sistema ** 2)
    risco_sistema_lista.append((1 / (1 + np.exp(-z_sistema))) * 100)

df['Fase'], df['P_opt_Min_Sistema'], df['P_opt_Max_Sistema'], df['Risco_Sistema_%'] = fases, p_opt_min_memoria, p_opt_max_memoria, risco_sistema_lista

# --- CÁLCULO DA VERDADE BIOLÓGICA (Para análise de Erro) ---
condicoes_bio = [df['Peso_Nascer_Kg'] < p_opt_min_bio, df['Peso_Nascer_Kg'] > p_opt_max_bio]
escolhas_bio = [df['Peso_Nascer_Kg'] - p_opt_min_bio, df['Peso_Nascer_Kg'] - p_opt_max_bio]
penalidade_real = np.select(condicoes_bio, escolhas_bio, default=0)

z_real = -2.5 + 1.2 * (penalidade_real ** 2)
df['Risco_Real_%'] = (1 / (1 + np.exp(-z_real))) * 100
df['Erro_Predicao'] = df['Risco_Sistema_%'] - df['Risco_Real_%']

# ==========================================
# 3. CONE DE CRESCIMENTO (Sem cálculo duplo)
# ==========================================
vetor_tempo_grafico = np.arange(0, 91)
animal_real = df.iloc[-1]
peso_nascer_indiv = animal_real['Peso_Nascer_Kg']
gmd_real_indiv = animal_real['GMD_Real'] # Consome a biologia gravada no banco

# A INTELIGÊNCIA: O que o sistema ESPERA (Predição)
centro_plato_aprendido = (p_opt_min_sistema + p_opt_max_sistema) / 2
gmd_esperado_sistema = gmd_base + 0.02 * (peso_nascer_indiv - centro_plato_aprendido)
if animal_real['Tipo_Parto'] == 'Múltiplo':
    gmd_esperado_sistema -= 0.025

trajetoria_alvo = peso_nascer_indiv + (gmd_esperado_sistema * vetor_tempo_grafico)
sigma_t = 0.35 + (1.5 * (vetor_tempo_grafico / 90)**2) 
limite_atencao, limite_critico = trajetoria_alvo - (1.0 * sigma_t), trajetoria_alvo - (2.0 * sigma_t)

# PESAGENS NO CURRAL: Usa a biologia REAL recuperada + Ruído sanitário
dias_pesagem = np.array([15, 30, 45, 60, 75, 90])
pesos_biologicos_nos_dias = peso_nascer_indiv + (gmd_real_indiv * dias_pesagem)
ruido_biologico = np.array([np.random.normal(0, 0.2), np.random.normal(0, 0.3), -0.9, -2.1, -1.0, np.random.normal(0, 0.4)])
pesagens_reais = pesos_biologicos_nos_dias + ruido_biologico

# O Motor calculando o Z-Score das pesagens
cores_alerta = []
for t, peso_medido in zip(dias_pesagem, pesagens_reais):
    peso_alvo_dia = peso_nascer_indiv + (gmd_esperado_sistema * t)
    z_score_atual = (peso_medido - peso_alvo_dia) / (0.35 + (1.5 * (t / 90)**2))
    cores_alerta.append('#2ecc71' if z_score_atual >= -1.0 else '#f1c40f' if z_score_atual >= -2.0 else '#e74c3c')

# ==========================================
# 4. GERAÇÃO DOS GRÁFICOS
# ==========================================
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Análise Preditiva e Retreino Automático', fontsize=18, fontweight='bold', y=0.95)

# Gráfico 1: Queda do Erro
axs[0, 0].axvline(x=50, color='black', linestyle='--'); sns.lineplot(data=df, x='ID_Parto', y='Erro_Predicao', hue='Fase', palette=['#e74c3c', '#2ecc71'], ax=axs[0, 0]); axs[0, 0].set_title('Queda do Erro de Predição')

# Gráfico 2: Faixa Ótima
axs[0, 1].axvline(x=50, color='black', linestyle='--'); axs[0, 1].plot(df['ID_Parto'], df['P_opt_Min_Sistema'], color='#3498db', label='Limites do Sistema'); axs[0, 1].plot(df['ID_Parto'], df['P_opt_Max_Sistema'], color='#3498db'); axs[0, 1].axhspan(p_opt_min_bio, p_opt_max_bio, color='#e67e22', alpha=0.3, label='Realidade Biológica'); axs[0, 1].set_title('Ajuste do Platô à Realidade Local'); axs[0, 1].legend()

# Gráfico 3: Curva Logística
pesos_t = np.linspace(1.5, 5.0, 200); pen_t = np.select([pesos_t < p_opt_min_bio, pesos_t > p_opt_max_bio], [pesos_t - p_opt_min_bio, pesos_t - p_opt_max_bio], 0); risco_t = (1 / (1 + np.exp(-(-2.5 + 1.2 * (pen_t ** 2))))) * 100
sns.scatterplot(data=df, x='Peso_Nascer_Kg', y='Risco_Sistema_%', hue='Fase', palette=['#e74c3c', '#2ecc71'], ax=axs[1, 0]); axs[1, 0].plot(pesos_t, risco_t, 'k:'); axs[1, 0].set_title('Risco Epidemiológico (Faixa Ótima)')

# Gráfico 4: Cone
ax4 = axs[1, 1]
ax4.plot(vetor_tempo_grafico, trajetoria_alvo, '#2c3e50', lw=2, label=f'Alvo do Sistema (GMD: {gmd_esperado_sistema:.3f})')
ax4.fill_between(vetor_tempo_grafico, trajetoria_alvo, limite_atencao, color='#2ecc71', alpha=0.2, label='Normal (Z > -1.0)')
ax4.fill_between(vetor_tempo_grafico, limite_atencao, limite_critico, color='#f1c40f', alpha=0.3, label='Atenção (-2.0 < Z < -1.0)')
ax4.fill_between(vetor_tempo_grafico, limite_critico, limite_critico - 2, color='#e74c3c', alpha=0.2, label='Crítico (Z < -2.0)')
ax4.scatter(dias_pesagem, pesagens_reais, c=cores_alerta, s=80, zorder=5, edgecolors='black', label='Pesagens Reais')
ax4.set_title(f'Trajetória Animal P0={peso_nascer_indiv}kg (GMD Real={gmd_real_indiv:.3f})', fontsize=13, fontweight='bold')
ax4.legend(loc='upper left', fontsize=9)

plt.tight_layout()
plt.show()