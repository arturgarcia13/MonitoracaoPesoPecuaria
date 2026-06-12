import os
from src.main.simulation.data_generator import GeradorDados
from src.main.ml.trainer import TreinadorModelos
from src.main.visualization.plotter import PlotadorEstatisticas


def orquestrar_pipeline():
    print("=" * 60)
    print("  SISTEMA DE MONITORAMENTO BIOMETRICO DE OVINOS")
    print("=" * 60)

    # 1. Simulacao dos Dados (Expert System)
    print("\n[1/4] Simulando banco de dados longitudinal...")
    gerador = GeradorDados(n_animais=500)
    df_animais, df_pesagens = gerador.gerar_dados()

    path_dados = "docs/requirements/dataset_simulado/"
    os.makedirs(path_dados, exist_ok=True)
    df_animais.to_csv(os.path.join(path_dados, "animais_simulados.csv"), index=False)
    df_pesagens.to_csv(os.path.join(path_dados, "pesagens_simuladas.csv"), index=False)

    n_vivos = (df_animais["Y_Morto"] == 0).sum()
    n_mortos = (df_animais["Y_Morto"] == 1).sum()
    print(f"    -> {len(df_animais)} animais simulados | {n_vivos} sobreviventes | {n_mortos} obitos neonatais")
    print(f"    -> {len(df_pesagens)} registros de pesagem longitudinal")

    # 2. Treinamento de Machine Learning
    print("\n[2/4] Treinando modelos de regressao e classificacao...")
    treinador = TreinadorModelos(caminho_backup="deploy/modelos/")

    treinador.treinar_modelo_crescimento(df_animais, df_pesagens)
    print("    -> Modelo de Regressao Linear (Curva de Crescimento) treinado e salvo!")

    treinador.treinar_modelo_mortalidade(df_animais)
    print("    -> Modelo de Regressao Logistica (Risco Neonatal) treinado e salvo!")

    # 3. Geracao de Graficos
    print("\n[3/4] Gerando visualizacoes de inteligencia...")
    caminho_graficos = "docs/architecture/graficos/"
    plotador = PlotadorEstatisticas(caminho_saida=caminho_graficos)

    plotador.gerar_graficos_zootecnia(df_animais, df_pesagens)
    plotador.gerar_graficos_setor_privado(df_animais, df_pesagens)
    plotador.gerar_grafico_zscore_rebanho(df_animais, df_pesagens)
    print(f"    -> 7 graficos exportados para: {caminho_graficos}")

    # 4. Demonstracao do Monitoramento Z-Score
    print("\n[4/4] Demonstracao do sistema de alertas (Z-Score individual)...")
    _demonstrar_alertas(df_animais, df_pesagens)

    print("\n" + "=" * 60)
    print("  Pipeline concluido com sucesso.")
    print("=" * 60)


def _demonstrar_alertas(df_animais, df_pesagens):
    """Demonstra o monitoramento individual para os 5 primeiros animais vivos."""
    from src.main.monitoring.monitorador_rebanho import MonitoradorRebanho

    monitor = MonitoradorRebanho()
    df_vivos = df_animais[df_animais["Y_Morto"] == 0].head(5)
    df_90 = df_pesagens[df_pesagens["Dias_Vida"] == 90]

    for _, animal in df_vivos.iterrows():
        pesagem = df_90[df_90["ID_Animal"] == animal["ID_Animal"]]
        if pesagem.empty:
            continue
        peso_real = pesagem.iloc[0]["Peso_Atual"]
        resultado = monitor.avaliar_pesagem(
            int(animal["ID_Animal"]),
            peso_real,
            90,
            animal["Sexo"],
            animal["Tipo_Parto"],
            animal["Ordem_Parto"],
            animal["Peso_Nascer"],
        )
        # Removendo emojis para compatibilidade com console Windows cp1252
        status_txt = resultado.status.value.replace("\U0001f7e2", "[OK]").replace("\U0001f7e1", "[!]").replace("\U0001f534", "[X]")
        print(f"    Animal #{resultado.id_animal:3d} | {status_txt} | Z={resultado.z_score:.2f}")


if __name__ == "__main__":
    orquestrar_pipeline()
