import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,
)
from src.main.ml.trainer import TreinadorModelos
from src.main.simulation.data_generator import GeradorDados


class DiagnosticoModelos:
    """
    Responsabilidade: Gerar graficos de avaliacao estatistica para cada uma das
    tres equacoes do documento regressao.md, permitindo inspecao visual da
    qualidade e comportamento dos modelos treinados.

    Graficos gerados por equacao:

    Equacao 1 — P0 (Regressao Linear):
        - Previsto vs Real
        - Distribuicao dos Residuos
        - Residuos vs Previsto (homocedasticidade)
        - Coeficientes beta aprendidos (comparacao com documento)

    Equacao 2 — Pt (Regressao Linear):
        - Previsto vs Real ao longo do tempo
        - Curvas de crescimento: real vs previsto por animal
        - Residuos vs Dias_Vida (heterogeneidade temporal)
        - Distribuicao dos residuos

    Equacao 3 — Risco Neonatal (Regressao Logistica):
        - Curva ROC com AUC
        - Curva logistica ajustada sobreposta aos dados empiricos
        - Distribuicao de probabilidades por classe
        - Matriz de confusao
    """

    CAMINHO_SAIDA = "docs/architecture/graficos/diagnostico/"
    ESTILO_GRADE = dict(linestyle="--", alpha=0.4, color="gray")

    def __init__(self, caminho_saida: str = None):
        self.caminho_saida = caminho_saida or self.CAMINHO_SAIDA
        os.makedirs(self.caminho_saida, exist_ok=True)

    # ─── Equacao 1: P0 ───────────────────────────────────────────────────────

    def diagnosticar_equacao_p0(self, pipeline, df_animais: pd.DataFrame):
        """Gera painel de 4 graficos para o Modelo 1 (Peso ao Nascer)."""
        X = df_animais[["Sexo", "Tipo_Parto", "Ordem_Parto"]]
        y = df_animais["Peso_Nascer"]
        y_pred = pipeline.predict(X)
        residuos = y - y_pred

        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)

        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(
            f"Diagnostico — Equacao 1: P0 = beta0 + beta_parto + beta_sexo + beta_matriz\n"
            f"RMSE={rmse:.4f}kg  |  R2={r2:.4f}  |  MAE={mae:.4f}kg",
            fontsize=13, fontweight="bold", y=0.98
        )
        gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

        # Plot 1: Previsto vs Real
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(y, y_pred, alpha=0.4, s=20, color="#2c3e50")
        lim = [min(y.min(), y_pred.min()) - 0.2, max(y.max(), y_pred.max()) + 0.2]
        ax1.plot(lim, lim, "r--", linewidth=1.5, label="Linha ideal (y=x)")
        ax1.set_xlabel("P0 Real (kg)", fontsize=11)
        ax1.set_ylabel("P0 Previsto (kg)", fontsize=11)
        ax1.set_title("Previsto vs Real", fontsize=12)
        ax1.legend(fontsize=9)
        ax1.grid(**self.ESTILO_GRADE)

        # Plot 2: Distribuicao dos Residuos
        ax2 = fig.add_subplot(gs[0, 1])
        sns.histplot(residuos, kde=True, bins=25, color="#2980b9", edgecolor="white", ax=ax2)
        ax2.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax2.set_xlabel("Residuo (kg)", fontsize=11)
        ax2.set_title("Distribuicao dos Residuos", fontsize=12)
        ax2.grid(**self.ESTILO_GRADE)

        # Plot 3: Residuos vs Previsto
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.scatter(y_pred, residuos, alpha=0.4, s=20, color="#8e44ad")
        ax3.axhline(0, color="red", linestyle="--", linewidth=1.5)
        ax3.set_xlabel("P0 Previsto (kg)", fontsize=11)
        ax3.set_ylabel("Residuo (kg)", fontsize=11)
        ax3.set_title("Residuos vs Previsto (Homocedasticidade)", fontsize=12)
        ax3.grid(**self.ESTILO_GRADE)

        # Plot 4: Coeficientes beta aprendidos vs documentados
        ax4 = fig.add_subplot(gs[1, 1])
        coef_doc = {
            "Femea": -0.30,
            "Gemeo": -0.65,
            "Trigemeos": -1.40,
            "Primipara": -0.35,
        }
        # Extraindo coeficientes do modelo treinado
        ohe = pipeline.named_steps["preprocessador"].transformers_[0][1]
        reg = pipeline.named_steps["regressor"]
        feature_names = ohe.get_feature_names_out(["Sexo", "Tipo_Parto", "Ordem_Parto"])
        coef_aprendido = dict(zip(feature_names, reg.coef_))

        # Mapeando para comparacao
        mapa = {
            "Femea": coef_aprendido.get("Sexo_M", coef_aprendido.get("Sexo_F", 0)) * -1,
            "Gemeo": coef_aprendido.get("Tipo_Parto_Simples", 0) * -1,
            "Trigemeos": coef_aprendido.get("Tipo_Parto_Trigemeos", 0),
            "Primipara": coef_aprendido.get("Ordem_Parto_Primipara", 0),
        }

        nomes = list(coef_doc.keys())
        vals_doc = [coef_doc[n] for n in nomes]
        vals_ml = [coef_aprendido.get(k, 0) for k in feature_names[:4]]

        x = np.arange(len(nomes))
        w = 0.35
        ax4.bar(x - w/2, vals_doc, w, label="Documento (teorico)", color="#27ae60", alpha=0.8)

        # Coeficientes do modelo aprendido (posicoes reais do OHE)
        coefs_reais = reg.coef_
        labels_ohe = list(feature_names)
        vals_para_plot = []
        nomes_para_plot = ["Femea (F)", "Gemeo", "Trigemeos", "Primipara"]
        for feat in ["Sexo_F", "Tipo_Parto_Gemeo", "Tipo_Parto_Trigemeos", "Ordem_Parto_Primipara"]:
            if feat in labels_ohe:
                vals_para_plot.append(coefs_reais[labels_ohe.index(feat)])
            else:
                vals_para_plot.append(0.0)

        ax4.bar(x + w/2, vals_para_plot, w, label="Modelo ML (aprendido)", color="#e74c3c", alpha=0.8)
        ax4.set_xticks(x)
        ax4.set_xticklabels(nomes_para_plot, fontsize=9)
        ax4.set_ylabel("Coeficiente beta (kg)", fontsize=11)
        ax4.set_title("Coeficientes: Documento vs ML", fontsize=12)
        ax4.axhline(0, color="black", linewidth=0.8)
        ax4.legend(fontsize=9)
        ax4.grid(axis="y", **self.ESTILO_GRADE)

        plt.savefig(
            os.path.join(self.caminho_saida, "diagnostico_equacao1_P0.png"),
            dpi=300, bbox_inches="tight"
        )
        plt.close()
        print(f"  -> diagnostico_equacao1_P0.png salvo")

    # ─── Equacao 2: Pt ───────────────────────────────────────────────────────

    def diagnosticar_equacao_pt(
        self, pipeline, df_animais: pd.DataFrame, df_pesagens: pd.DataFrame
    ):
        """Gera painel de 4 graficos para o Modelo 2 (Trajetoria de Peso)."""
        df = pd.merge(df_pesagens, df_animais[["ID_Animal", "Peso_Nascer", "Y_Morto"]], on="ID_Animal")
        df_vivos = df[df["Y_Morto"] == 0].copy()

        X = df_vivos[["Peso_Nascer", "Dias_Vida"]]
        y = df_vivos["Peso_Atual"]
        y_pred = pipeline.predict(X)
        residuos = y - y_pred

        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)

        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(
            f"Diagnostico — Equacao 2: Pt = P0 + GMD * t + eta_t\n"
            f"RMSE={rmse:.4f}kg  |  R2={r2:.4f}  |  MAE={mae:.4f}kg",
            fontsize=13, fontweight="bold", y=0.98
        )
        gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

        # Plot 1: Previsto vs Real
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(y, y_pred, alpha=0.25, s=15, color="#16a085")
        lim = [0, max(y.max(), y_pred.max()) + 1]
        ax1.plot(lim, lim, "r--", linewidth=1.5, label="Linha ideal")
        ax1.set_xlabel("Pt Real (kg)", fontsize=11)
        ax1.set_ylabel("Pt Previsto (kg)", fontsize=11)
        ax1.set_title("Previsto vs Real", fontsize=12)
        ax1.legend(fontsize=9)
        ax1.grid(**self.ESTILO_GRADE)

        # Plot 2: Curvas de crescimento (amostra de 15 animais)
        ax2 = fig.add_subplot(gs[0, 1])
        amostra_ids = df_vivos["ID_Animal"].unique()[:15]
        cores = plt.cm.Blues(np.linspace(0.4, 0.9, len(amostra_ids)))
        for idx, (id_a, cor) in enumerate(zip(amostra_ids, cores)):
            sub = df_vivos[df_vivos["ID_Animal"] == id_a].sort_values("Dias_Vida")
            ax2.plot(sub["Dias_Vida"], sub["Peso_Atual"], color=cor, alpha=0.7, linewidth=1.2)
            pred_sub = pipeline.predict(sub[["Peso_Nascer", "Dias_Vida"]])
            ax2.plot(sub["Dias_Vida"], pred_sub, color="red", alpha=0.5, linewidth=1, linestyle="--")

        from matplotlib.lines import Line2D
        legenda = [
            Line2D([0], [0], color="steelblue", linewidth=2, label="Real"),
            Line2D([0], [0], color="red", linewidth=2, linestyle="--", label="Previsto ML"),
        ]
        ax2.legend(handles=legenda, fontsize=9)
        ax2.set_xlabel("Dias de Vida", fontsize=11)
        ax2.set_ylabel("Peso (kg)", fontsize=11)
        ax2.set_title("Trajetorias: Real vs Previsto (amostra)", fontsize=12)
        ax2.grid(**self.ESTILO_GRADE)

        # Plot 3: Residuos vs Dias_Vida (heterogeneidade temporal)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.scatter(df_vivos["Dias_Vida"], residuos, alpha=0.25, s=15, color="#8e44ad")
        ax3.axhline(0, color="red", linestyle="--", linewidth=1.5)
        # Banda de sigma_t teorico
        dias_ord = np.array(sorted(df_vivos["Dias_Vida"].unique()))
        sigma_t = 0.35 + 1.5 * (dias_ord / 90) ** 2
        ax3.fill_between(dias_ord, -sigma_t, sigma_t, alpha=0.15, color="orange", label="Banda sigma_t (doc.)")
        ax3.set_xlabel("Dias de Vida (t)", fontsize=11)
        ax3.set_ylabel("Residuo (kg)", fontsize=11)
        ax3.set_title("Residuos vs Tempo (Heterogeneidade Residual)", fontsize=12)
        ax3.legend(fontsize=9)
        ax3.grid(**self.ESTILO_GRADE)

        # Plot 4: Distribuicao dos residuos
        ax4 = fig.add_subplot(gs[1, 1])
        sns.histplot(residuos, kde=True, bins=30, color="#2980b9", edgecolor="white", ax=ax4)
        ax4.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax4.axvline(residuos.mean(), color="orange", linestyle="-.", linewidth=1.5,
                    label=f"Media={residuos.mean():.3f}")
        ax4.set_xlabel("Residuo (kg)", fontsize=11)
        ax4.set_title("Distribuicao dos Residuos", fontsize=12)
        ax4.legend(fontsize=9)
        ax4.grid(**self.ESTILO_GRADE)

        plt.savefig(
            os.path.join(self.caminho_saida, "diagnostico_equacao2_Pt.png"),
            dpi=300, bbox_inches="tight"
        )
        plt.close()
        print(f"  -> diagnostico_equacao2_Pt.png salvo")

    # ─── Equacao 3: Risco Neonatal ────────────────────────────────────────────

    def diagnosticar_equacao_risco(self, pipeline, df_animais: pd.DataFrame):
        """Gera painel de 4 graficos para o Modelo 3 (Risco Neonatal)."""
        df_model = df_animais.copy()
        df_model["Desvio_Quad_P0"] = (df_model["Peso_Nascer"] - 4.0) ** 2
        X = df_model[["Desvio_Quad_P0"]]
        y = df_model["Y_Morto"]

        y_prob = pipeline.predict_proba(X)[:, 1]
        y_pred = pipeline.predict(X)

        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(
            "Diagnostico — Equacao 3: P(Y=1) = sigmoid(alpha0 + alpha1*(P0-4)^2)\n"
            "Modelo de Risco Neonatal — Regressao Logistica",
            fontsize=13, fontweight="bold", y=0.98
        )
        gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

        # Plot 1: Curva ROC
        ax1 = fig.add_subplot(gs[0, 0])
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color="#c0392b", linewidth=2.5, label=f"AUC = {roc_auc:.4f}")
        ax1.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax1.fill_between(fpr, tpr, alpha=0.1, color="#c0392b")
        ax1.set_xlabel("Taxa de Falso Positivo", fontsize=11)
        ax1.set_ylabel("Taxa de Verdadeiro Positivo", fontsize=11)
        ax1.set_title("Curva ROC", fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(**self.ESTILO_GRADE)

        # Plot 2: Curva logistica ajustada sobre dados empiricos
        ax2 = fig.add_subplot(gs[0, 1])
        pesos_range = np.linspace(df_animais["Peso_Nascer"].min() - 0.5,
                                   df_animais["Peso_Nascer"].max() + 0.5, 300)
        desvio_range = (pesos_range - 4.0) ** 2
        df_range = pd.DataFrame({"Desvio_Quad_P0": desvio_range})
        prob_range = pipeline.predict_proba(df_range)[:, 1]

        # Pontos empiricos (taxa de obito por faixa)
        faixas = pd.cut(df_animais["Peso_Nascer"], bins=10)
        taxa_emp = df_animais.groupby(faixas, observed=True).agg(
            taxa=("Y_Morto", "mean"), centro=("Peso_Nascer", "mean")
        ).dropna()

        ax2.scatter(taxa_emp["centro"], taxa_emp["taxa"], color="#27ae60",
                    zorder=5, s=60, label="Taxa empirica por faixa")
        ax2.plot(pesos_range, prob_range, color="#c0392b", linewidth=2.5, label="Curva logistica ML")
        ax2.axvline(4.0, color="gray", linestyle="--", linewidth=1.2, label="Peso otimo (4.0 kg)")
        ax2.set_xlabel("Peso ao Nascer (kg)", fontsize=11)
        ax2.set_ylabel("P(Obito)", fontsize=11)
        ax2.set_title("Curva Logistica Ajustada vs Dados Empiricos", fontsize=12)
        ax2.legend(fontsize=9)
        ax2.grid(**self.ESTILO_GRADE)

        # Plot 3: Distribuicao de probabilidades por classe
        ax3 = fig.add_subplot(gs[1, 0])
        df_probs = pd.DataFrame({"prob": y_prob, "classe": y.map({0: "Sobreviveu", 1: "Obito"})})
        sns.histplot(data=df_probs, x="prob", hue="classe", kde=True,
                     palette={"Sobreviveu": "#27ae60", "Obito": "#e74c3c"},
                     bins=20, alpha=0.6, ax=ax3)
        ax3.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Limiar 0.5")
        ax3.set_xlabel("Probabilidade de Obito Prevista", fontsize=11)
        ax3.set_title("Distribuicao de Probabilidades por Classe Real", fontsize=12)
        ax3.legend(fontsize=9)
        ax3.grid(**self.ESTILO_GRADE)

        # Plot 4: Matriz de confusao
        ax4 = fig.add_subplot(gs[1, 1])
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Sobreviveu (0)", "Obito (1)"]
        )
        disp.plot(ax=ax4, colorbar=False, cmap="Blues")
        ax4.set_title("Matriz de Confusao", fontsize=12)

        plt.savefig(
            os.path.join(self.caminho_saida, "diagnostico_equacao3_risco.png"),
            dpi=300, bbox_inches="tight"
        )
        plt.close()
        print(f"  -> diagnostico_equacao3_risco.png salvo")


# ─── Execucao direta ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  GERANDO GRAFICOS DE DIAGNOSTICO DOS MODELOS")
    print("=" * 60)

    # Simular dados
    print("\n[1/3] Simulando dados...")
    gerador = GeradorDados(n_animais=500)
    df_animais, df_pesagens = gerador.gerar_dados()

    # Treinar os 3 modelos
    print("\n[2/3] Treinando modelos...")
    treinador = TreinadorModelos(caminho_backup="deploy/modelos/")
    pipeline_p0 = treinador.treinar_equacao_p0(df_animais)
    pipeline_pt = treinador.treinar_equacao_pt(df_animais, df_pesagens)
    pipeline_risco = treinador.treinar_equacao_risco(df_animais)

    # Gerar diagnosticos
    print("\n[3/3] Gerando paineis de diagnostico...")
    diag = DiagnosticoModelos()
    diag.diagnosticar_equacao_p0(pipeline_p0, df_animais)
    diag.diagnosticar_equacao_pt(pipeline_pt, df_animais, df_pesagens)
    diag.diagnosticar_equacao_risco(pipeline_risco, df_animais)

    print(f"\nPaineis salvos em: {diag.caminho_saida}")
    print("=" * 60)
