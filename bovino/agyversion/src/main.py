import os
import simulador_dados
import treinamento_modelo
import diagnosticos
import sistema_alertas

def executar_pipeline():
    print("Iniciando Pipeline de Regressão e Diagnóstico...")
    
    # Configuração de caminhos
    base_dir = 'E:/MYAREA/AREA_DEV/Faculdade/ModelosRegressaoI/MonitaracaoPesoBovinoWorkflow/v1'
    data_dir = os.path.join(base_dir, 'data')
    output_dir = os.path.join(base_dir, 'outputs')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    caminho_csv_base = os.path.join(data_dir, 'dados_bovinos.csv')
    caminho_csv_pred = os.path.join(output_dir, 'dados_com_predicoes.csv')
    
    # 1. Simulação
    print("\n1. Simulando Dados Genéticos e de Crescimento...")
    df_base = simulador_dados.simular_dados_bovinos()
    df_base.to_csv(caminho_csv_base, index=False)
    
    # 2. Treinamento e Diagnóstico Rigoroso
    print("2. Ajustando MRLM e executando Testes Formais (Statsmodels)...")
    _, _, _ = treinamento_modelo.treinar_modelo(caminho_csv_base, output_dir)
    
    # 3. Geração de Gráficos de Negócio / Pitch
    print("3. Gerando Artefatos Visuais para o Pitch Executivo...")
    diagnosticos.gerar_graficos_apresentacao(caminho_csv_pred, output_dir)
    
    # 4. Sistema de Alertas
    print("4. Executando Sistema de Alertas de Prejuízo...")
    n_alertas = sistema_alertas.rodar_alerta_prejuizo(caminho_csv_pred)
    print(f" -> Concluído. {n_alertas} animais sinalizados para descarte precoce.")
    
    print("\nPipeline finalizado com sucesso! Artefatos salvos em v1/outputs.")

if __name__ == "__main__":
    executar_pipeline()
