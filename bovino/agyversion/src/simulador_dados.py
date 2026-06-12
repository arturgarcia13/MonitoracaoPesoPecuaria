import numpy as np
import pandas as pd

def simular_dados_bovinos(n_amostras=1000, random_state=42):
    """
    Simula dados de crescimento de bovinos de corte baseados em modelos genético-quantitativos.
    Estrutura inspirada no Modelo Animal (Animal Model): Y = Xb + Za + Mm + Wp + e
    """
    np.random.seed(random_state)
    
    # Efeitos Fixos (Xb): Ano de nascimento (tendência genética) e Sexo
    ano_nascimento = np.random.randint(2015, 2024, n_amostras)
    sexo = np.random.choice(['M', 'F'], n_amostras)
    efeito_sexo = np.where(sexo == 'M', 1.15, 1.0) # Machos 15% mais pesados
    
    # Efeito de tendência genética: Ganho médio por ano (ex: 1.5 kg/ano)
    tendencia_genetica = (ano_nascimento - 2015) * 1.5
    
    # Efeitos Genéticos Aditivos Diretos (a) e Maternos (m)
    # Assumindo distribuição normal
    capacidade_genetica_a = np.random.normal(0, 15, n_amostras)
    habilidade_materna_m = np.random.normal(0, 10, n_amostras)
    
    # Efeito de Ambiente Permanente/Temporário (e)
    ambiente_pre_desmama = np.random.normal(0, 8, n_amostras)
    ambiente_pos_desmama = np.random.normal(0, 12, n_amostras)
    
    # Construção dos Pesos
    peso_nascer = 30 + (capacidade_genetica_a * 0.1) + np.random.normal(0, 2, n_amostras)
    
    # Ganho de Peso Diário (GPD) Pré-desmama (até ~210 dias)
    # Influenciado fortemente pela habilidade materna
    gpd_pre = 0.8 + (habilidade_materna_m * 0.01) + (capacidade_genetica_a * 0.005) + np.random.normal(0, 0.05, n_amostras)
    peso_desmama = peso_nascer + (gpd_pre * 210) + ambiente_pre_desmama + efeito_sexo * 10
    
    # GPD Pós-desmama (210 a 500 dias)
    # Influenciado mais pela capacidade genética direta
    gpd_pos = 0.6 + (capacidade_genetica_a * 0.015) + np.random.normal(0, 0.08, n_amostras)
    peso_adulto = peso_desmama + (gpd_pos * 290) + ambiente_pos_desmama + tendencia_genetica + efeito_sexo * 25
    
    # Algumas anomalias para testar a robustez do modelo (outliers / mascaramento)
    idx_outliers = np.random.choice(n_amostras, int(n_amostras * 0.02), replace=False)
    peso_adulto[idx_outliers] += np.random.choice([-100, 100], len(idx_outliers))
    
    # Custos e Receita (Setor Privado)
    custo_mantença = 1500 + (peso_adulto * 1.2) + np.random.normal(0, 100, n_amostras)
    preco_arroba = 250
    receita_estimada = (peso_adulto / 30) * preco_arroba # Assumindo 50% de rendimento de carcaça (peso/2 em kg = peso/30 em arrobas reais)
    lucro = receita_estimada - custo_mantença
    
    df = pd.DataFrame({
        'id_animal': range(1, n_amostras + 1),
        'ano_nascimento': ano_nascimento,
        'sexo': sexo,
        'peso_nascer': peso_nascer,
        'gpd_pre_desmama': gpd_pre,
        'peso_desmama': peso_desmama,
        'gpd_pos_desmama': gpd_pos,
        'peso_adulto': peso_adulto,
        'custo_mantenca': custo_mantença,
        'receita_estimada': receita_estimada,
        'lucro': lucro
    })
    
    return df

if __name__ == "__main__":
    df = simular_dados_bovinos()
    df.to_csv('E:/MYAREA/AREA_DEV/Faculdade/ModelosRegressaoI/MonitaracaoPesoBovinoWorkflow/v1/data/dados_bovinos.csv', index=False)
    print("Dados simulados e salvos com sucesso!")
