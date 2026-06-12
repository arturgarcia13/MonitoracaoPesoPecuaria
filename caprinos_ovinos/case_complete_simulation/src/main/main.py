import os
from simulation.data_generator import GeradorDados
from ml.trainer import TreinadorModelos
from ml.predictor import PreditivoModelos
from visualization.plotter import PlotadorEstatisticas


def orquestrar_pipeline():
    print("=" * 65)
    print("  SISTEMA DE MONITORAMENTO BIOMETRICO DE OVINOS")
    print("  Baseado em: regressao.md (Secs. 3.1, 3.2, 3.3)")
    print("=" * 65)

    # ── FASE 1: Simulacao dos dados ──────────────────────────────────────────
    print("\n[FASE 1] Simulando banco de dados longitudinal...")
    gerador = GeradorDados(n_animais=500)
    df_animais, df_pesagens = gerador.gerar_dados()

    path_dados = "docs/requirements/dataset_simulado/"
    os.makedirs(path_dados, exist_ok=True)
    df_animais.to_csv(os.path.join(path_dados, "animais_simulados.csv"), index=False)
    df_pesagens.to_csv(os.path.join(path_dados, "pesagens_simuladas.csv"), index=False)

    n_vivos = int((df_animais["Y_Morto"] == 0).sum())
    n_mortos = int((df_animais["Y_Morto"] == 1).sum())
    print(f"  -> {len(df_animais)} animais | {n_vivos} sobreviventes | {n_mortos} obitos")
    print(f"  -> {len(df_pesagens)} pesagens longitudinais (dias 0 a 90)")

    # ── FASE 2: Treinamento das equacoes do documento ────────────────────────
    print("\n[FASE 2] Treinando os tres modelos do documento regressao.md...")
    treinador = TreinadorModelos(caminho_backup="deploy/modelos/")

    print("\n  -- Equacao 1 (Secao 3.1): P0 = 4.10 + beta_parto + beta_sexo + beta_matriz + eta")
    treinador.treinar_equacao_p0(df_animais)

    print("\n  -- Equacao 2 (Secao 3.2): Pt = P0 + GMD * t + eta_t")
    treinador.treinar_equacao_pt(df_animais, df_pesagens)

    print("\n  -- Equacao 3 (Secao 3.3): P(Y=1) = sigmoid(alpha0 + alpha1*(P0-4)^2)")
    treinador.treinar_equacao_risco(df_animais)

    # ── FASE 3: Visualizacoes ────────────────────────────────────────────────
    print("\n[FASE 3] Gerando visualizacoes para Zootecnia e Setor Privado...")
    caminho_graficos = "docs/architecture/graficos/"
    plotador = PlotadorEstatisticas(caminho_saida=caminho_graficos)
    plotador.gerar_graficos_zootecnia(df_animais, df_pesagens)
    plotador.gerar_graficos_setor_privado(df_animais, df_pesagens)
    plotador.gerar_grafico_zscore_rebanho(df_animais, df_pesagens)
    print(f"  -> 7 graficos exportados para: {caminho_graficos}")

    # ── FASE 4: Demonstracao do fluxo sequencial de inferencia ───────────────
    print("\n[FASE 4] Demonstracao do fluxo de inferencia sequencial (Passo 1 -> 2 -> 3)...")
    preditivo = PreditivoModelos(caminho_backup="deploy/modelos/")

    cenarios = [
        {"sexo": "M", "tipo_parto": "Simples", "ordem": "Multipara",  "p0_real": 4.10, "desc": "Animal referencia"},
        {"sexo": "F", "tipo_parto": "Gemeo",   "ordem": "Primipara",  "p0_real": 2.80, "desc": "Animal alto risco"},
        {"sexo": "M", "tipo_parto": "Simples", "ordem": "Multipara",  "p0_real": 5.50, "desc": "Macrossomia"},
    ]

    print(f"\n  {'Cenario':<20} {'P0 Real':>8} {'Pt(90d)':>10} {'Risco Obito':>13}")
    print("  " + "-" * 55)
    for c in cenarios:
        resultado = preditivo.executar_pipeline_completo(
            sexo=c["sexo"], tipo_parto=c["tipo_parto"], ordem_parto=c["ordem"],
            peso_nascer_real=c["p0_real"], dias_projecao=90
        )
        risco_pct = resultado["prob_obito"] * 100
        print(
            f"  {c['desc']:<20} {c['p0_real']:>7.2f}kg"
            f" {resultado['pt_projetado_dia_90']:>9.2f}kg"
            f" {risco_pct:>12.1f}%"
        )

    # ── FASE 5: Sistema de alertas Z-Score ───────────────────────────────────
    print("\n[FASE 5] Demonstracao do sistema de alertas Z-Score individual...")
    _demonstrar_alertas_zscore(df_animais, df_pesagens)

    print("\n" + "=" * 65)
    print("  Pipeline concluido com sucesso.")
    print("=" * 65)


def _demonstrar_alertas_zscore(df_animais, df_pesagens):
    from monitoring.monitorador_rebanho import MonitoradorRebanho
    monitor = MonitoradorRebanho()
    df_vivos = df_animais[df_animais["Y_Morto"] == 0].head(6)
    df_90 = df_pesagens[df_pesagens["Dias_Vida"] == 90]

    print(f"  {'ID':>4} {'Status':<12} {'Z-Score':>8} {'Real':>8} {'Esperado':>10}")
    print("  " + "-" * 50)
    for _, animal in df_vivos.iterrows():
        pesagem = df_90[df_90["ID_Animal"] == animal["ID_Animal"]]
        if pesagem.empty:
            continue
        peso_real = pesagem.iloc[0]["Peso_Atual"]
        r = monitor.avaliar_pesagem(
            int(animal["ID_Animal"]), peso_real, 90,
            animal["Sexo"], animal["Tipo_Parto"], animal["Ordem_Parto"], animal["Peso_Nascer"],
        )
        status_curto = r.status.value.split()[0]
        print(f"  {r.id_animal:>4} {status_curto:<12} {r.z_score:>8.2f} {r.peso_real:>7.2f}kg {r.peso_esperado:>9.2f}kg")


if __name__ == "__main__":
    orquestrar_pipeline()
