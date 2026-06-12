import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera, durbin_watson
import json
import os

def treinar_modelo(caminho_dados, caminho_saida):
    """
    Treina o modelo de regressão linear (MRLM) utilizando statsmodels para diagnósticos rigorosos.
    O objetivo (Setor Privado) é prever o Lucro com base em indicadores iniciais do animal.
    """
    df = pd.read_csv(caminho_dados)
    
    # Tratamento de Variáveis Categóricas (Dummy para Sexo)
    df = pd.get_dummies(df, columns=['sexo'], drop_first=True, dtype=int)
    
    # Variáveis Explicativas (X) e Resposta (Y)
    # Selecionando variaveis disponíveis precocemente para tomada de decisão (abate precoce ou retenção)
    features = ['peso_nascer', 'gpd_pre_desmama', 'peso_desmama', 'sexo_M']
    X = df[features]
    X = sm.add_constant(X)
    Y = df['lucro']
    
    # Ajuste do Modelo MQO (OLS)
    modelo = sm.OLS(Y, X).fit()
    
    # Diagnósticos Rigorosos Exigidos pelo Professor
    # 1. Teste de Normalidade dos Resíduos (Jarque-Bera)
    jb_test = jarque_bera(modelo.resid)
    # 2. Teste de Autocorrelação (Durbin-Watson)
    dw_test = durbin_watson(modelo.resid)
    
    # 3. Multicolinearidade (VIF)
    vif_data = {X.columns[i]: variance_inflation_factor(X.values, i) for i in range(X.shape[1])}
    
    # 4. Análise de Influência (Distância de Cook e Alavancagem)
    influence = modelo.get_influence()
    cooks_d = influence.cooks_distance[0]
    leverage = influence.hat_matrix_diag
    studentized_res = influence.resid_studentized_external
    
    # Identificar outliers/pontos influentes (Regra de ouro Cook's D > 4/n)
    n = len(Y)
    limite_cook = 4 / n
    outliers_influentes = np.where(cooks_d > limite_cook)[0]
    
    # Salvando resultados e sumário estatístico
    resultados = {
        'R2': float(modelo.rsquared),
        'R2_Adjusted': float(modelo.rsquared_adj),
        'F_statistic': float(modelo.fvalue),
        'F_pvalue': float(modelo.f_pvalue),
        'Jarque_Bera_pvalue': float(jb_test[1]),
        'Durbin_Watson': float(dw_test),
        'VIF': vif_data,
        'Num_Outliers_Influentes': int(len(outliers_influentes))
    }
    
    os.makedirs(caminho_saida, exist_ok=True)
    with open(f"{caminho_saida}/diagnostico_estatistico.json", 'w') as f:
        json.dump(resultados, f, indent=4)
        
    with open(f"{caminho_saida}/sumario_modelo.txt", 'w') as f:
        f.write(modelo.summary().as_text())
        
    # Adicionar predições e resíduos ao DF para gráficos
    df['predicao_lucro'] = modelo.fittedvalues
    df['residuos'] = modelo.resid
    df['residuos_estudentizados'] = studentized_res
    df['distancia_cook'] = cooks_d
    df['alavancagem'] = leverage
    
    df.to_csv(f"{caminho_saida}/dados_com_predicoes.csv", index=False)
    
    return modelo, resultados, df

if __name__ == "__main__":
    caminho_dados = 'E:/MYAREA/AREA_DEV/Faculdade/ModelosRegressaoI/MonitaracaoPesoBovinoWorkflow/v1/data/dados_bovinos.csv'
    caminho_saida = 'E:/MYAREA/AREA_DEV/Faculdade/ModelosRegressaoI/MonitaracaoPesoBovinoWorkflow/v1/outputs'
    treinar_modelo(caminho_dados, caminho_saida)
    print("Modelo treinado e diagnósticos gerados.")
