import numpy as np
import pandas as pd
from scipy.stats import truncnorm
import os

def simulate_data(n_samples=1000, seed=42):
    """
    Gera um dataset sintético de monitoramento de peso bovino baseado em 
    distribuições probabilísticas fundamentadas na literatura científica.
    """
    np.random.seed(seed)
    
    # X1 - Idade (dias): Uniforme(1, 180)
    idade = np.random.uniform(1, 180, n_samples)
    
    # X2 - Peso ao Nascimento (kg): N(29, 3.5^2) truncado [18, 45]
    # Em scipy, truncnorm leva limites em termos de z-score
    a_pn, b_pn = (18 - 29) / 3.5, (45 - 29) / 3.5
    pn = truncnorm.rvs(a_pn, b_pn, loc=29, scale=3.5, size=n_samples)
    
    # X3 - Brix (%): N(24, 4^2) truncado [12, 38]
    a_brix, b_brix = (12 - 24) / 4, (38 - 24) / 4
    brix = truncnorm.rvs(a_brix, b_brix, loc=24, scale=4, size=n_samples)
    
    # X4 - Isolamento (dias): Zero-Inflated Poisson(λ=2, p0=0.70)
    # 70% de chance de ser 0 (excesso de zeros). Os outros 30% vêm de uma Poisson(λ=2)
    is_zero = np.random.binomial(1, 0.70, n_samples)
    isolamento = (1 - is_zero) * np.random.poisson(2, n_samples)
    
    # X5 - Sexo (macho=1): Bernoulli(0.50)
    sexo = np.random.binomial(1, 0.50, n_samples)
    
    # X6 - Novilha (filho de novilha primípara=1): Bernoulli(0.25)
    novilha = np.random.binomial(1, 0.25, n_samples)
    
    # --- Parâmetros de Regressão Sugeridos ---
    # β0: Intercepto (o PN já atua como intercepto basal da vida do animal, 
    # mas ajustamos β0 para englobar efeitos fixos base)
    beta_0 = 0.0      
    
    # β1 (Idade): GMD médio em torno de 0.65 kg/dia
    beta_1 = 0.65     
    
    # β2 (PN): Peso ao nascer reflete no peso atual (assumimos repasse de 1:1 na base)
    beta_2 = 1.0      
    
    # β3 (Brix): Efeito indireto positivo via melhor imunidade (0.2 kg a mais por cada 1% de Brix)
    beta_3 = 0.2      
    
    # β4 (Isolamento): Efeito negativo de dias doentes (-1.5 kg por dia de apatia)
    beta_4 = -1.5     
    
    # β5 (Sexo): Machos pesam em média 3 a 5 kg a mais nessa janela pré-desmama
    beta_5 = 4.0      
    
    # β6 (Novilha): Filhos de novilha têm desenvolvimento menor (-3.0 kg)
    beta_6 = -3.0     
    
    # Erro aleatório do modelo normal (kg)
    # Assumimos um desvio padrão residual de 5 kg
    epsilon = np.random.normal(0, 5, n_samples) 
    
    # Modelo Linear de Regressão Múltipla
    peso = (beta_0 
            + beta_1 * idade 
            + beta_2 * pn 
            + beta_3 * brix 
            + beta_4 * isolamento 
            + beta_5 * sexo 
            + beta_6 * novilha 
            + epsilon)
            
    # Cria DataFrame com arredondamentos lógicos
    df = pd.DataFrame({
        'Idade_dias': np.round(idade, 0).astype(int),
        'PN_kg': np.round(pn, 1),
        'Brix_perc': np.round(brix, 1),
        'Isolamento_dias': isolamento,
        'Sexo_Macho': sexo,
        'Filho_Novilha': novilha,
        'Peso_kg': np.round(peso, 1)
    })
    
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Gerar simulação de monitoramento de peso bovino')
    parser.add_argument('--n', type=int, default=2000, help='Número de amostras a gerar')
    args = parser.parse_args()
    
    print(f"Iniciando simulação com {args.n} amostras...")
    df_simulado = simulate_data(n_samples=args.n)
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "dataset_simulado_peso_bovino.csv")
    
    df_simulado.to_csv(output_path, index=False)
    
    print("Simulação concluída com sucesso.")
    print(f"Arquivo salvo em: {output_path}")
    print("\n--- Estatísticas Descritivas ---")
    print(df_simulado.describe().round(2))
