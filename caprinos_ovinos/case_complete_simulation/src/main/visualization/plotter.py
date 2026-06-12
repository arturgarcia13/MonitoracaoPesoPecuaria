import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
from domain.calculadora_biometrica import CalculadoraBiometrica
from domain.avaliador_risco import AvaliadorRiscoNeonatal
from monitoring.monitorador_rebanho import MonitoradorRebanho, StatusAlerta


class PlotadorEstatisticas:
    """
    Responsabilidade: Gerar visualizações focadas em dois perfis:
    - Zootecnistas (Foco biológico, sobrevivência, variação no tempo)
    - Setor Privado (Foco em performance financeira/peso, correlação de crescimento)
    """

    # Paleta de cores coerente com identidade visual técnica
    COR_PRIMARIA = "#2c3e50"
    COR_ALERTA_NORMAL = "#27ae60"
    COR_ALERTA_ATENCAO = "#f39c12"
    COR_ALERTA_CRITICO = "#e74c3c"

    def __init__(self, caminho_saida: str = "docs/graphs/"):
        self.caminho_saida = caminho_saida
        if not os.path.exists(self.caminho_saida):
            os.makedirs(self.caminho_saida)

    # ─── Interface pública ────────────────────────────────────────────────────

    def gerar_graficos_zootecnia(self, df_animais: pd.DataFrame, df_pesagens: pd.DataFrame):
        """Gera gráficos biológicos/técnicos."""
        self._plotar_curva_crescimento_percentis(df_animais, df_pesagens)
        self._plotar_mortalidade_vs_peso(df_animais)
        self._plotar_curva_logistica_continua()

    def gerar_graficos_setor_privado(self, df_animais: pd.DataFrame, df_pesagens: pd.DataFrame):
        """Gera gráficos de performance de negócio."""
        self._plotar_correlacao_peso_gmd(df_animais)
        self._plotar_distribuicao_peso_abate(df_pesagens, df_animais)
        self._plotar_distribuicao_por_tipo_parto(df_pesagens, df_animais)

    def gerar_grafico_zscore_rebanho(self, df_animais: pd.DataFrame, df_pesagens: pd.DataFrame):
        """Gera gráfico de Z-Score ao longo do tempo para amostra do rebanho."""
        self._plotar_zscore_temporal(df_animais, df_pesagens)

    # ─── Gráficos de Zootecnia ───────────────────────────────────────────────

    def _plotar_curva_crescimento_percentis(
        self, df_animais: pd.DataFrame, df_pesagens: pd.DataFrame
    ):
        """Plota o corredor de crescimento com P5, P50 e P95 por categoria."""
        df = pd.merge(df_pesagens, df_animais[["ID_Animal", "Y_Morto"]], on="ID_Animal")
        df_vivos = df[df["Y_Morto"] == 0]

        fig, ax = plt.subplots(figsize=(11, 6))

        # Calcular percentis por dia
        p5 = df_vivos.groupby("Dias_Vida")["Peso_Atual"].quantile(0.05)
        p50 = df_vivos.groupby("Dias_Vida")["Peso_Atual"].quantile(0.50)
        p95 = df_vivos.groupby("Dias_Vida")["Peso_Atual"].quantile(0.95)
        dias = p50.index

        ax.fill_between(dias, p5.values, p95.values, alpha=0.2, color=self.COR_PRIMARIA, label="Corredor P5-P95")
        ax.plot(dias, p50.values, color=self.COR_PRIMARIA, linewidth=2.5, label="Mediana (P50)")
        ax.plot(dias, p5.values, color=self.COR_ALERTA_ATENCAO, linewidth=1.5, linestyle="--", label="P5 (Limiar Inferior)")
        ax.plot(dias, p95.values, color=self.COR_ALERTA_NORMAL, linewidth=1.5, linestyle="--", label="P95 (Limiar Superior)")

        ax.set_title("Zootecnia: Corredor de Crescimento do Rebanho (P5 | Mediana | P95)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Dias de Vida", fontsize=12)
        ax.set_ylabel("Peso (kg)", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlim(0, 90)
        plt.tight_layout()
        plt.savefig(os.path.join(self.caminho_saida, "curva_crescimento_percentis.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def _plotar_mortalidade_vs_peso(self, df_animais: pd.DataFrame):
        """Demonstração empírica da Curva em U da mortalidade."""
        fig, ax = plt.subplots(figsize=(10, 6))

        faixas = pd.cut(
            df_animais["Peso_Nascer"],
            bins=[0, 2.5, 3.5, 4.5, 5.5, 10],
            labels=["< 2.5", "2.5-3.5", "3.5-4.5", "4.5-5.5", "> 5.5"],
        )
        taxa_morte = df_animais.groupby(faixas, observed=False)["Y_Morto"].mean() * 100

        cores = [self.COR_ALERTA_CRITICO if v > 30 else self.COR_ALERTA_ATENCAO if v > 7.5 else self.COR_ALERTA_NORMAL
                 for v in taxa_morte.values]

        ax.bar(taxa_morte.index, taxa_morte.values, color=cores, edgecolor="white", linewidth=0.5)
        ax.axhline(y=7.5, color=self.COR_ALERTA_NORMAL, linestyle="--", linewidth=1.5, label="Alvo ≤ 7.5%")

        ax.set_title("Zootecnia: Risco Neonatal — Curva em U de Mortalidade", fontsize=14, fontweight="bold")
        ax.set_xlabel("Faixa de Peso ao Nascer (kg)", fontsize=12)
        ax.set_ylabel("Taxa de Mortalidade (%)", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.caminho_saida, "risco_neonatal_curva_u.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def _plotar_curva_logistica_continua(self):
        """Plota a curva logística teórica contínua de mortalidade (documento, Cenário B)."""
        avaliador = AvaliadorRiscoNeonatal()
        pesos = np.linspace(1.0, 7.0, 300)
        probs = [avaliador.calcular_probabilidade_obito(p) * 100 for p in pesos]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(pesos, probs, color=self.COR_ALERTA_CRITICO, linewidth=2.5)
        ax.axvline(x=4.0, color=self.COR_ALERTA_NORMAL, linestyle="--", linewidth=1.5, label="Peso ótimo (4.0 kg)")

        # Anotações dos cenários do documento
        for p0, descricao in [(2.0, "2.0 kg → ~91% Risco"), (4.0, "4.0 kg → ~7.5% Risco"), (5.5, "5.5 kg → ~55% Risco")]:
            prob = avaliador.calcular_probabilidade_obito(p0) * 100
            ax.annotate(descricao, xy=(p0, prob), xytext=(p0 + 0.3, prob + 5),
                        arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)

        ax.fill_between(pesos, probs, alpha=0.1, color=self.COR_ALERTA_CRITICO)
        ax.set_title("Zootecnia: Curva Logística Teórica de Mortalidade Neonatal", fontsize=14, fontweight="bold")
        ax.set_xlabel("Peso ao Nascer (kg)", fontsize=12)
        ax.set_ylabel("Probabilidade de Óbito (%)", fontsize=12)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.caminho_saida, "curva_logistica_mortalidade.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # ─── Gráficos de Setor Privado ───────────────────────────────────────────

    def _plotar_correlacao_peso_gmd(self, df_animais: pd.DataFrame):
        """Correlação entre Peso ao Nascer e GMD para animais sobreviventes."""
        fig, ax = plt.subplots(figsize=(10, 6))
        df_vivos = df_animais[df_animais["Y_Morto"] == 0]

        sns.regplot(data=df_vivos, x="Peso_Nascer", y="GMD_Real",
                    scatter_kws={"alpha": 0.4, "color": self.COR_PRIMARIA, "s": 20},
                    line_kws={"color": self.COR_ALERTA_CRITICO, "linewidth": 2.5}, ax=ax)

        corr = df_vivos["Peso_Nascer"].corr(df_vivos["GMD_Real"])
        ax.text(0.05, 0.92, f"r = {corr:.3f}", transform=ax.transAxes, fontsize=12,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        ax.set_title("Negócios: Impacto do Peso de Nascimento no Ganho Diário (GMD)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Peso ao Nascer (kg)", fontsize=12)
        ax.set_ylabel("Ganho Médio Diário (kg/dia)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.caminho_saida, "correlacao_peso_gmd_negocios.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def _plotar_distribuicao_peso_abate(self, df_pesagens: pd.DataFrame, df_animais: pd.DataFrame):
        """Distribuição do peso final do rebanho aos 90 dias."""
        df_90 = df_pesagens[df_pesagens["Dias_Vida"] == 90]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df_90, x="Peso_Atual", kde=True, color=self.COR_ALERTA_NORMAL,
                     bins=25, edgecolor="white", ax=ax)

        media = df_90["Peso_Atual"].mean()
        p10 = df_90["Peso_Atual"].quantile(0.10)
        ax.axvline(x=media, color=self.COR_PRIMARIA, linestyle="--", linewidth=2, label=f"Média: {media:.1f}kg")
        ax.axvline(x=p10, color=self.COR_ALERTA_CRITICO, linestyle="--", linewidth=2, label=f"P10 (Refugo): {p10:.1f}kg")

        ax.set_title("Negócios: Distribuição do Peso do Rebanho aos 90 Dias (Desmama)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Peso Final (kg)", fontsize=12)
        ax.set_ylabel("Nº de Animais", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.caminho_saida, "distribuicao_peso_90d.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def _plotar_distribuicao_por_tipo_parto(self, df_pesagens: pd.DataFrame, df_animais: pd.DataFrame):
        """Comparação de peso final por tipo de parto — identifica o impacto econômico das gestações múltiplas."""
        df_90 = df_pesagens[df_pesagens["Dias_Vida"] == 90]
        df_merged = pd.merge(df_90, df_animais[["ID_Animal", "Tipo_Parto"]], on="ID_Animal")

        fig, ax = plt.subplots(figsize=(10, 6))
        paleta = {"Simples": self.COR_ALERTA_NORMAL, "Gemeo": self.COR_ALERTA_ATENCAO, "Trigemeos": self.COR_ALERTA_CRITICO}

        for tipo, grupo in df_merged.groupby("Tipo_Parto"):
            sns.kdeplot(data=grupo, x="Peso_Atual", label=f"{tipo} (n={len(grupo)})",
                        color=paleta.get(tipo, "gray"), linewidth=2.5, ax=ax)

        ax.set_title("Negócios: Distribuição de Peso Final por Tipo de Parto", fontsize=14, fontweight="bold")
        ax.set_xlabel("Peso aos 90 dias (kg)", fontsize=12)
        ax.set_ylabel("Densidade", fontsize=12)
        ax.legend(title="Tipo de Parto", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.caminho_saida, "distribuicao_peso_por_tipo_parto.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # ─── Gráfico de Z-Score ──────────────────────────────────────────────────

    def _plotar_zscore_temporal(self, df_animais: pd.DataFrame, df_pesagens: pd.DataFrame):
        """Plota a trajetória do Z-Score individual de uma amostra do rebanho."""
        monitor = MonitoradorRebanho()
        df_vivos = df_animais[df_animais["Y_Morto"] == 0].head(20)  # amostra de 20 animais
        df_merged = pd.merge(df_pesagens, df_vivos, on="ID_Animal")

        fig, ax = plt.subplots(figsize=(12, 6))

        for id_animal, grupo in df_merged.groupby("ID_Animal"):
            animal = df_vivos[df_vivos["ID_Animal"] == id_animal].iloc[0]
            zscores = []
            for _, row in grupo.iterrows():
                r = monitor.avaliar_pesagem(
                    int(id_animal), row["Peso_Atual"], int(row["Dias_Vida"]),
                    animal["Sexo"], animal["Tipo_Parto"], animal["Ordem_Parto"], animal["Peso_Nascer"]
                )
                zscores.append(r.z_score)
            ax.plot(grupo["Dias_Vida"].values, zscores, alpha=0.45, linewidth=1.2, color=self.COR_PRIMARIA)

        # Faixas de alerta
        ax.axhline(y=-1.0, color=self.COR_ALERTA_ATENCAO, linestyle="--", linewidth=2, label="Limiar Atenção (Z=-1)")
        ax.axhline(y=-2.0, color=self.COR_ALERTA_CRITICO, linestyle="--", linewidth=2, label="Limiar Crítico (Z=-2)")
        ax.axhline(y=0, color=self.COR_ALERTA_NORMAL, linestyle="-", linewidth=1.5, alpha=0.6, label="Média Esperada (Z=0)")

        ax.fill_between([0, 90], [-1, -1], [5, 5], alpha=0.06, color=self.COR_ALERTA_NORMAL)
        ax.fill_between([0, 90], [-2, -2], [-1, -1], alpha=0.1, color=self.COR_ALERTA_ATENCAO)
        ax.fill_between([0, 90], [-6, -6], [-2, -2], alpha=0.1, color=self.COR_ALERTA_CRITICO)

        ax.set_title("Zootecnia: Trajetória do Z-Score Individual (Amostra de 20 Animais)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Dias de Vida", fontsize=12)
        ax.set_ylabel("Z-Score (desvios do esperado individual)", fontsize=12)
        ax.set_xlim(0, 90)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(self.caminho_saida, "zscore_temporal_rebanho.png"), dpi=300, bbox_inches="tight")
        plt.close()
