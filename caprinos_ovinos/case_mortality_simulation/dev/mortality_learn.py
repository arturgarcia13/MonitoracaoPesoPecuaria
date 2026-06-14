import numpy as np
import pandas as pd

def simular_fase_aprendizado(nome_cenario, peso_medio_rebanho, p_opt_biologico, n_partos=150):
    print(f"\n--- Iniciando Simulação: Cenário {nome_cenario} ---")
    
    # Semente fixa para fins acadêmicos (reprodutibilidade dos mesmos pesos sempre que rodar)
    np.random.seed(42)
    
    # 1. GERAÇÃO DO DATASET SINTÉTICO (A Realidade Física)
    # Simulamos o peso com desvio padrão de 0.5kg (conforme a literatura base)
    pesos_aferidos = np.random.normal(loc=peso_medio_rebanho, scale=0.5, size=n_partos)
    
    df = pd.DataFrame({
        'ID_Parto': range(1, n_partos + 1), 
        'Peso_Nascer_Kg': np.round(pesos_aferidos, 2)
    })
    
    # 2. DEFINIÇÕES DA MÁQUINA DE ESTADO
    p_opt_sistema = 4.0  # Valor hardcoded inicial (Cold Start)
    gatilho_retreino = 50 # Quantidade de partos para aprender
    
    fases = []
    risco_sistema_lista = []
    
    # 3. MOTOR DE INFERÊNCIA SEQUENCIAL (Simulando o tempo real)
    for index, linha in df.iterrows():
        peso_atual = linha['Peso_Nascer_Kg']
        
        # --- O GATILHO DE RETREINO SILENCIOSO ---
        if index == gatilho_retreino:
            # Isola os 50 primeiros registros
            amostra_historica = df.loc[0:(gatilho_retreino-1), 'Peso_Nascer_Kg']
            
            # Truncamento de Sanidade: Só aprende com dados dentro da normalidade biológica
            amostra_limpa = amostra_historica[(amostra_historica >= 2.0) & (amostra_historica <= 5.5)]
            
            # Atualiza o P_opt do sistema (passa de 4.0 para a mediana local)
            if not amostra_limpa.empty:
                p_opt_sistema = np.round(amostra_limpa.median(), 2)
                print(f"[*] RETREINO ATIVADO no parto 50! Novo P_opt atualizado para: {p_opt_sistema} kg")
        
        # Identifica em qual fase o sistema está
        fase_atual = '1_Cold_Start' if index < gatilho_retreino else '2_Calibrado'
        fases.append(fase_atual)
        
        # --- CÁLCULO LOGÍSTICO DO SISTEMA ---
        z_sistema = -2.5 + 1.2 * ((peso_atual - p_opt_sistema) ** 2)
        risco_sistema = (1 / (1 + np.exp(-z_sistema))) * 100
        risco_sistema_lista.append(risco_sistema)
        
    df['Fase_do_Motor'] = fases
    df['Risco_Sistema_%'] = np.round(risco_sistema_lista, 1)
    
    # 4. AVALIAÇÃO DA VERDADE (Ground Truth Biológico)
    # Como a biologia daquele animal real reagiu?
    z_real = -2.5 + 1.2 * ((df['Peso_Nascer_Kg'] - p_opt_biologico) ** 2)
    df['Risco_Biologico_Real_%'] = np.round((1 / (1 + np.exp(-z_real))) * 100, 1)
    
    # 5. CÁLCULO DE ERRO (O Falso Alarme)
    # Se der positivo, o sistema está alarmando mais do que deveria (Superestimando).
    # Se der negativo, o sistema não alarmou o produtor quando deveria (Subestimando).
    df['Erro_Predicao'] = df['Risco_Sistema_%'] - df['Risco_Biologico_Real_%']
    
    return df

# ==========================================
# EXECUÇÃO DO PIPELINE DE PESQUISA
# ==========================================

# Cenário 1: Rebanho Pesado (Ex: Dorper puros)
df_bom = simular_fase_aprendizado("BOM", peso_medio_rebanho=4.5, p_opt_biologico=4.5)

# Cenário 2: Rebanho Comercial Padrão (Encaixa na literatura)
df_medio = simular_fase_aprendizado("MEDIO", peso_medio_rebanho=4.0, p_opt_biologico=4.0)

# Cenário 3: Rebanho Leve (Ex: Morada Nova pura - Teste de Estresse do Algoritmo)
df_ruim = simular_fase_aprendizado("RUIM", peso_medio_rebanho=3.1, p_opt_biologico=3.1)

# Exemplo de Análise: Imprimindo o erro médio do sistema nas duas fases para o Cenário Ruim
print("\n--- ANÁLISE DO CENÁRIO RUIM (Morada Nova) ---")
erro_cold = df_ruim[df_ruim['Fase_do_Motor'] == '1_Cold_Start']['Erro_Predicao'].mean()
erro_calibrado = df_ruim[df_ruim['Fase_do_Motor'] == '2_Calibrado']['Erro_Predicao'].mean()

print(f"Erro Médio do Sistema na Fase COLD START: +{erro_cold:.1f}% (Muito Falso Positivo)")
print(f"Erro Médio do Sistema APÓS CALIBRAÇÃO: {erro_calibrado:.1f}% (Precisão Recuperada)")