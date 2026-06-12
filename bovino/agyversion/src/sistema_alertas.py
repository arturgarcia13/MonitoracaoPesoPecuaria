import pandas as pd

def rodar_alerta_prejuizo(caminho_dados):
    """
    Sistema prático de alerta focado em regras de negócio (Setor Privado).
    Se a predição de lucro do animal (baseada nos dados até a desmama) for negativa
    com os custos previstos, o animal recebe um alerta de "Descarte Precoce".
    """
    df = pd.read_csv(caminho_dados)
    
    # Animais com projeção de prejuízo financeiro
    risco = df[df['predicao_lucro'] < 0].copy()
    
    if len(risco) > 0:
        risco['recomendacao'] = 'Venda Imediata / Descarte Precoce'
        risco_alerta = risco[['id_animal', 'peso_desmama', 'predicao_lucro', 'recomendacao']]
        risco_alerta.to_csv('E:/MYAREA/AREA_DEV/Faculdade/ModelosRegressaoI/MonitaracaoPesoBovinoWorkflow/v1/outputs/animais_em_risco.csv', index=False)
        return len(risco)
    return 0

if __name__ == "__main__":
    caminho_dados = 'E:/MYAREA/AREA_DEV/Faculdade/ModelosRegressaoI/MonitaracaoPesoBovinoWorkflow/v1/outputs/dados_com_predicoes.csv'
    n_alertas = rodar_alerta_prejuizo(caminho_dados)
    print(f"Sistema de Alerta: {n_alertas} animais identificados com risco de prejuízo.")
