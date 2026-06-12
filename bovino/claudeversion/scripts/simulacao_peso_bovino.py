"""
=============================================================================
SIMULAÇÃO SINTÉTICA DE PESO BOVINO — MODELO DE MONITORIZAÇÃO
=============================================================================
Fundamentação: Soberon et al. (2012), Laureano et al. (2011),
               Bielmann et al. (2010), de Souza et al. (2018),
               Mucari & Oliveira (2003), Boligon et al. (s.d.),
               Lalman & Holder (NASEM).

Estratégia de Geração (Estratégia 3 — GMD como variável latente):
    Peso_i = PN_i + GMD_i × Idade_i + ε_i
    GMD_i  = GMD_base_i + Σ δ_k · X_k,i + u_i

Cenários:
    M1 — Modelo Completo  : Idade + PN + Brix + Isolamento + Sexo + Novilha + Raça
    M2 — Sem Brix         : sem X3
    M3 — Sem Isolamento   : sem X4
    M4 — Sem Brix/Isol.   : sem X3 e X4

Comparação: R², AIC, BIC + gráficos diagnósticos completos
=============================================================================
"""

# ── Imports ────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURAÇÃO GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
SEED        = 42
N_PER_MODEL = 1_000          # ← altere aqui o n por cenário
IDADE_MIN   = 1
IDADE_MAX   = 180
OUTPUT_DIR  = "."             # pasta de saída para figuras/csv

np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PARÂMETROS DAS RAÇAS  (calibrados pela literatura)
# ─────────────────────────────────────────────────────────────────────────────
# Cada raça define:
#   pn_mu   : média PN (kg)         — Soberon 2012, Boligon s.d., Mucari 2003
#   pn_sd   : DP PN (kg)
#   gmd_mu  : GMD base médio (kg/d) — Laureano 2011, Soberon 2012
#   gmd_sd  : DP GMD base (kg/d)
#   prop    : proporção no plantel (deve somar 1.0)
#
# Referências por raça:
#   Nelore       → Laureano 2011 (GND/210d = 0.674), Boligon s.d. (PN≈29)
#   Guzerá       → Mucari 2003 (P8=150.3, PN≈26)
#   Holandês/Gir → Soberon 2012 (PN=42, GMD=0.82 leiteiro intensivo)
#   Cruzado ½    → média ponderada zebu×taurino (inferência biológica)
#   Angus/Sim.   → NASEM (Lalman), similar ao Holandês em GMD, PN maior

RACAS = {
    "Nelore": {
        "pn_mu": 29.0, "pn_sd": 3.5,
        "gmd_mu": 0.550, "gmd_sd": 0.100,
        "prop": 0.30
    },
    "Guzerá": {
        "pn_mu": 26.0, "pn_sd": 3.0,
        "gmd_mu": 0.500, "gmd_sd": 0.095,
        "prop": 0.15
    },
    "Holandês/Girolando": {
        # GMD ajustado para contexto semi-intensivo brasileiro (não Cornell intensivo).
        # Soberon 2012 (Cornell, substituto ad libitum): 0.82 kg/d.
        # Ajuste: -15% para regime semi-intensivo/pastagem + suplemento → 0.68 kg/d.
        "pn_mu": 40.0, "pn_sd": 5.0,
        "gmd_mu": 0.680, "gmd_sd": 0.115,
        "prop": 0.20
    },
    "Cruzado ½ sangue": {
        "pn_mu": 33.0, "pn_sd": 4.0,
        "gmd_mu": 0.640, "gmd_sd": 0.110,
        "prop": 0.20
    },
    "Angus/Simental": {
        "pn_mu": 38.0, "pn_sd": 4.5,
        "gmd_mu": 0.720, "gmd_sd": 0.115,
        "prop": 0.15
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# 2.  DELTAS DOS EFEITOS SOBRE O GMD  (todos com fonte na literatura)
# ─────────────────────────────────────────────────────────────────────────────
# δ_sex   : de Souza et al. 2018 — diferença 44.7 kg / 365d = +0.122 kg/d;
#            interpolado para período precoce (0-180d): +0.067 kg/d
# δ_brix  : Bielmann 2010 → FTP → Soberon 2012 (-40g/d doença × duração);
#            por unidade de Brix acima do limiar 22%: +0.002 kg/d  [INFERENCIAL]
# δ_iso   : Soberon 2012 — antibiotic + disease: -50 g/d (P<0.01)
#            adotado: -0.040 kg/d por dia de isolamento
# δ_nov   : Bielmann 2010 + Lalman NASEM — novilha prima: colostro inferior
#            + competição energética; estimado ≈ -2.5 kg/60d → -0.042 kg/d

DELTAS = {
    "sex":           +0.067,   # kg/d (macho vs fêmea)  — de Souza 2018
    "brix_per_unit": +0.002,   # kg/d por % Brix acima de 22  — INFERENCIAL
    "brix_ref":       22.0,    # % Brix de referência (limiar FTP) — Bielmann 2010
    "iso":           -0.040,   # kg/d por dia de isolamento  — Soberon 2012
    "novilha":       -0.042,   # kg/d (filho de novilha)     — Bielmann + Lalman
}

# ─────────────────────────────────────────────────────────────────────────────
# 3.  FUNÇÃO GERADORA DE DADOS SINTÉTICOS
# ─────────────────────────────────────────────────────────────────────────────

def gerar_dados(n: int, seed: int = SEED) -> pd.DataFrame:
    """
    Gera n observações sintéticas usando a Estratégia 3 (GMD latente).

    Retorna DataFrame com colunas:
        Raca, Idade, PN, Brix, Isolamento, Sexo, Novilha,
        GMD_base, GMD, Peso   (Y = PN + GMD × Idade + ε)

    Validação interna:
        Peso médio aos 60d (zebu) deve estar em [50, 80] kg  → CV ~12-20%
    """
    rng = np.random.default_rng(seed)

    racas_list = list(RACAS.keys())
    props      = [RACAS[r]["prop"] for r in racas_list]

    # 3.1  Raça (categórica, multinomial)
    raca_idx = rng.choice(len(racas_list), size=n, p=props)
    raca_nomes = np.array(racas_list)[raca_idx]

    # 3.2  Variáveis independentes
    # X1 — Idade (dias): Uniforme [1, 180]
    Idade = rng.uniform(IDADE_MIN, IDADE_MAX, n).astype(int)

    # X2 — Peso ao Nascimento: Normal truncada, por raça
    PN = np.array([
        np.clip(
            rng.normal(RACAS[r]["pn_mu"], RACAS[r]["pn_sd"]),
            RACAS[r]["pn_mu"] - 3*RACAS[r]["pn_sd"],
            RACAS[r]["pn_mu"] + 3*RACAS[r]["pn_sd"]
        )
        for r in raca_nomes
    ])

    # X3 — Brix (%): Normal truncada [12, 38]
    #   Média 24% (ligeiramente abaixo dos 26.1% de laboratório — Bielmann 2010 —
    #   para refletir condições de campo no Brasil)
    Brix = np.clip(rng.normal(24.0, 4.0, n), 12.0, 38.0)

    # X4 — Isolamento (dias): Poisson(λ=1.5), truncado em [0, 30]
    #   ~60% dos animais com 0 dias (rebanho bem manejado)
    Isolamento = np.clip(rng.poisson(1.5, n), 0, 30).astype(int)

    # X5 — Sexo: Bernoulli(0.50)  — 1=macho, 0=fêmea
    Sexo = rng.binomial(1, 0.50, n)

    # X6 — Novilha: Bernoulli(0.25)  — 1=filho de novilha primípara
    Novilha = rng.binomial(1, 0.25, n)

    # 3.3  GMD individual (variável latente)
    GMD_base = np.array([
        np.clip(
            rng.normal(RACAS[r]["gmd_mu"], RACAS[r]["gmd_sd"]),
            0.10, 1.60
        )
        for r in raca_nomes
    ])

    # Efeitos sobre GMD
    delta_sex    = DELTAS["sex"]           * Sexo
    delta_brix   = DELTAS["brix_per_unit"] * (Brix - DELTAS["brix_ref"])
    delta_iso    = DELTAS["iso"]           * Isolamento
    delta_novilha= DELTAS["novilha"]       * Novilha

    # Variação individual residual no GMD (genética não capturada pelas covariáveis)
    # SD derivada de: σ²_p P365 = 1354.58 (de Souza 2018) → σ(GMD) ≈ 0.101 kg/d
    u_i = rng.normal(0.0, 0.080, n)

    GMD = GMD_base + delta_sex + delta_brix + delta_iso + delta_novilha + u_i
    GMD = np.clip(GMD, 0.05, 1.60)

    # 3.4  Y = PN + GMD × Idade + ε
    # ε representa erro de mensuração + ruído ambiental não estrutural
    # Calibrado pela fração residual da variância fenotípica:
    #   σ²_e / σ²_p ≈ 68% (de Souza 2018, P365); para ~60d → σ_ε ≈ 2.5 kg
    epsilon = rng.normal(0.0, 2.5, n)

    Peso = PN + GMD * Idade + epsilon
    Peso = np.clip(Peso, PN * 0.85, None)   # truncamento inferior biológico

    return pd.DataFrame({
        "Raca":        raca_nomes,
        "Idade":       Idade,
        "PN":          PN,
        "Brix":        Brix,
        "Isolamento":  Isolamento,
        "Sexo":        Sexo,
        "Novilha":     Novilha,
        "GMD_base":    GMD_base,
        "GMD":         GMD,
        "Peso":        Peso,
    })


# ─────────────────────────────────────────────────────────────────────────────
# 4.  OLS MANUAL  (sem statsmodels)
#     R², AIC, BIC pela log-verossimilhança do modelo gaussiano (MLE)
# ─────────────────────────────────────────────────────────────────────────────

class OLSModel:
    """
    Regressão Múltipla por Mínimos Quadrados Ordinários.
    Métricas: R², R²_adj, AIC, BIC (via log-likelihood gaussiana MLE).
    """

    def __init__(self, name: str):
        self.name   = name
        self.beta   = None
        self.cols   = None
        self.n      = None
        self.k      = None          # nº parâmetros (incl. intercepto)
        self.r2     = None
        self.r2_adj = None
        self.aic    = None
        self.bic    = None
        self.y_hat  = None
        self.resid  = None
        self.y      = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list):
        self.cols = feature_names
        self.n, self.k = X.shape      # X já inclui coluna de 1s
        self.y = y.copy()

        # OLS normal equations
        self.beta  = np.linalg.lstsq(X, y, rcond=None)[0]
        self.y_hat = X @ self.beta
        self.resid = y - self.y_hat

        # R²
        sse = np.sum(self.resid**2)
        sst = np.sum((y - y.mean())**2)
        self.r2     = 1.0 - sse / sst
        self.r2_adj = 1.0 - (1 - self.r2) * (self.n - 1) / (self.n - self.k)

        # Log-likelihood (MLE σ² = SSE/n)
        sigma2_mle  = sse / self.n
        log_lik     = (-self.n / 2.0) * np.log(2 * np.pi * sigma2_mle) - sse / (2 * sigma2_mle)

        # AIC & BIC  (k+1 parâmetros: β's + σ²)
        p = self.k + 1
        self.aic = -2 * log_lik + 2 * p
        self.bic = -2 * log_lik + p * np.log(self.n)

        return self

    def summary_row(self) -> dict:
        return {
            "Modelo": self.name,
            "n":      self.n,
            "k":      self.k,
            "R²":     round(self.r2, 4),
            "R²_adj": round(self.r2_adj, 4),
            "AIC":    round(self.aic, 2),
            "BIC":    round(self.bic, 2),
        }

    def coef_df(self) -> pd.DataFrame:
        return pd.DataFrame({"Feature": self.cols, "Coeficiente": self.beta})


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PREPARAÇÃO DA MATRIX DE DESIGN
# ─────────────────────────────────────────────────────────────────────────────

def build_design_matrix(df: pd.DataFrame, include_brix: bool, include_iso: bool):
    """
    Constrói a matriz de design X e o vetor y.
    Raça é codificada como dummy (referência: Nelore).
    """
    racas_dummies = pd.get_dummies(df["Raca"], prefix="Raca", drop_first=True)

    # Colunas sempre presentes
    base_cols = ["Intercepto", "Idade", "PN", "Sexo", "Novilha"]
    X = pd.DataFrame({
        "Intercepto": 1.0,
        "Idade":      df["Idade"].values.astype(float),
        "PN":         df["PN"].values,
        "Sexo":       df["Sexo"].values.astype(float),
        "Novilha":    df["Novilha"].values.astype(float),
    })

    if include_brix:
        X["Brix"] = df["Brix"].values

    if include_iso:
        X["Isolamento"] = df["Isolamento"].values.astype(float)

    # Dummies de raça (bool → int64 para evitar dtype object)
    racas_dummies = racas_dummies.astype(int)
    X = pd.concat([X, racas_dummies], axis=1)

    feature_names = X.columns.tolist()
    return X.values.astype(np.float64), df["Peso"].values.astype(np.float64), feature_names


# ─────────────────────────────────────────────────────────────────────────────
# 6.  VALIDAÇÃO DO GERADOR
# ─────────────────────────────────────────────────────────────────────────────

def validar_gerador(df: pd.DataFrame) -> None:
    """
    Verifica se os dados gerados são biologicamente plausíveis,
    comparando com pontos de âncora da literatura.
    """
    print("\n" + "─"*60)
    print("VALIDAÇÃO DO GERADOR — Pontos de Âncora da Literatura")
    print("─"*60)

    anchors = {
        "Nelore P60 (Laureano GND=0.674)":          (55, 80,  "Nelore",              55, 70),
        "Holandês/Giro P49 (Soberon adj. s.-int.)": (40, 60,  "Holandês/Girolando",  60, 85),
    }

    for label, (age_lo, age_hi, raca, exp_lo, exp_hi) in anchors.items():
        mask = (df["Idade"].between(age_lo, age_hi)) & (df["Raca"] == raca)
        sub = df.loc[mask, "Peso"]
        if len(sub) == 0:
            print(f"  {label}: sem obs. suficientes")
            continue
        m, s, cv = sub.mean(), sub.std(), sub.std()/sub.mean()*100
        ok = "✓" if exp_lo <= m <= exp_hi else "⚠ FORA DO INTERVALO"
        print(f"  {label}")
        print(f"    Média={m:.1f} kg  DP={s:.1f} kg  CV={cv:.1f}%  esperado=[{exp_lo},{exp_hi}]  {ok}")

    # Geral
    mask60 = df["Idade"].between(50, 70)
    sub60  = df.loc[mask60, "Peso"]
    print(f"\n  Geral (dias 50–70, n={len(sub60)}): "
          f"média={sub60.mean():.1f} kg, DP={sub60.std():.1f} kg, "
          f"CV={sub60.std()/sub60.mean()*100:.1f}%")
    print("─"*60)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  GRÁFICOS DIAGNÓSTICOS
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "M1 — Completo":          "#2563EB",
    "M2 — Sem Brix":          "#16A34A",
    "M3 — Sem Isolamento":    "#D97706",
    "M4 — Sem Brix/Isol.":   "#DC2626",
}


def plot_diagnosticos(models: list[OLSModel], output_dir: str = ".") -> None:
    """
    4-panel diagnostic plot por modelo:
        1. Resíduos vs. Valores Ajustados
        2. QQ-Plot dos Resíduos
        3. Predito vs. Observado
        4. Histograma dos Resíduos
    """
    for mdl in models:
        color = COLORS.get(mdl.name, "#4B5563")
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(f"Diagnósticos — {mdl.name}", fontsize=14, fontweight="bold", y=1.01)
        gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.35)

        # ── 1. Resíduos vs. Ajustados ─────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(mdl.y_hat, mdl.resid, alpha=0.35, s=12, color=color)
        ax1.axhline(0, color="black", linewidth=1.0, linestyle="--")
        # Linha de tendência LOWESS manual (média móvel por bin)
        bins = np.percentile(mdl.y_hat, np.linspace(5, 95, 20))
        bin_means = [mdl.resid[np.abs(mdl.y_hat - b) < 5].mean() for b in bins]
        ax1.plot(bins, bin_means, color="red", linewidth=1.5, label="Tendência")
        ax1.set_xlabel("Valores Ajustados (kg)")
        ax1.set_ylabel("Resíduos (kg)")
        ax1.set_title("Resíduos vs. Ajustados")
        ax1.legend(fontsize=8)

        # ── 2. QQ-Plot ────────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        (osm, osr), (slope, intercept, r) = stats.probplot(mdl.resid, dist="norm")
        ax2.scatter(osm, osr, alpha=0.35, s=12, color=color)
        line_x = np.array([osm[0], osm[-1]])
        ax2.plot(line_x, slope * line_x + intercept, color="black",
                 linewidth=1.5, linestyle="--")
        ax2.set_xlabel("Quantis Teóricos (Normal)")
        ax2.set_ylabel("Quantis Amostrais")
        ax2.set_title(f"QQ-Plot  (r = {r:.4f})")

        # ── 3. Predito vs. Observado ──────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.scatter(mdl.y, mdl.y_hat, alpha=0.25, s=12, color=color)
        lims = [min(mdl.y.min(), mdl.y_hat.min()) - 2,
                max(mdl.y.max(), mdl.y_hat.max()) + 2]
        ax3.plot(lims, lims, "k--", linewidth=1.2, label="Linha Ideal (45°)")
        ax3.set_xlim(lims); ax3.set_ylim(lims)
        ax3.set_xlabel("Peso Observado (kg)")
        ax3.set_ylabel("Peso Predito (kg)")
        ax3.set_title(f"Predito vs. Observado  (R²={mdl.r2:.4f})")
        ax3.legend(fontsize=8)

        # ── 4. Histograma dos Resíduos ────────────────────────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.hist(mdl.resid, bins=40, color=color, alpha=0.65, edgecolor="white",
                 density=True)
        # Curva normal de referência
        xr = np.linspace(mdl.resid.min(), mdl.resid.max(), 200)
        ax4.plot(xr, stats.norm.pdf(xr, mdl.resid.mean(), mdl.resid.std()),
                 color="black", linewidth=1.5, label="N(0,σ²) ref.")
        ax4.set_xlabel("Resíduos (kg)")
        ax4.set_ylabel("Densidade")
        ax4.set_title("Distribuição dos Resíduos")
        ax4.legend(fontsize=8)

        fig.tight_layout()
        fname = f"{output_dir}/diag_{mdl.name.replace(' ', '_').replace('/', '')}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Salvo: {fname}")


def plot_comparativo(models: list[OLSModel], output_dir: str = ".") -> None:
    """
    Painel comparativo dos 4 modelos:
        - R² (barras)
        - AIC / BIC (barras agrupadas)
        - Distribuição dos resíduos sobreposta
    """
    names  = [m.name for m in models]
    r2s    = [m.r2    for m in models]
    aics   = [m.aic   for m in models]
    bics   = [m.bic   for m in models]
    colors = [COLORS.get(n, "#4B5563") for n in names]
    x      = np.arange(len(models))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Comparação dos 4 Cenários de Regressão", fontsize=14, fontweight="bold")

    # ── R² ──────────────────────────────────────────────────────────────────
    ax = axes[0]
    bars = ax.bar(x, r2s, color=colors, width=0.55, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels([n.split(" — ")[1] for n in names],
                                          rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("R²")
    ax.set_title("R² por Modelo")
    ax.set_ylim(max(0, min(r2s) - 0.05), min(1.0, max(r2s) + 0.03))
    for bar, v in zip(bars, r2s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    # ── AIC & BIC ────────────────────────────────────────────────────────────
    ax = axes[1]
    w = 0.25
    b1 = ax.bar(x - w/2, aics, width=w, color=colors, alpha=0.85,
                edgecolor="white", label="AIC")
    b2 = ax.bar(x + w/2, bics, width=w, color=colors, alpha=0.50,
                edgecolor="white", hatch="//", label="BIC")
    ax.set_xticks(x); ax.set_xticklabels([n.split(" — ")[1] for n in names],
                                          rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Critério de Informação")
    ax.set_title("AIC e BIC por Modelo\n(menor = melhor)")
    ax.legend(fontsize=8)
    ax.set_ylim(min(aics + bics) * 0.995, max(aics + bics) * 1.005)
    for b, v in zip(list(b1) + list(b2), aics + bics):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
                f"{v:.0f}", ha="center", va="bottom", fontsize=7, rotation=90)

    # ── Distribuição dos resíduos ────────────────────────────────────────────
    ax = axes[2]
    for mdl, col in zip(models, colors):
        ax.hist(mdl.resid, bins=50, color=col, alpha=0.45,
                density=True, label=mdl.name.split(" — ")[1])
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Resíduo (kg)")
    ax.set_ylabel("Densidade")
    ax.set_title("Distribuição dos Resíduos\n(sobrepostos)")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fname = f"{output_dir}/comparativo_modelos.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {fname}")


def plot_crescimento_raca(df: pd.DataFrame, output_dir: str = ".") -> None:
    """
    Curvas de crescimento médio por raça ao longo dos 180 dias.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = plt.cm.tab10(np.linspace(0, 0.8, len(RACAS)))

    for raca, cor in zip(RACAS.keys(), palette):
        sub = df[df["Raca"] == raca].copy()
        # Binagem por decil de idade
        bins = np.linspace(IDADE_MIN, IDADE_MAX, 19)
        sub["bin"] = pd.cut(sub["Idade"], bins=bins, labels=bins[:-1])
        grp = sub.groupby("bin", observed=True)["Peso"].agg(["mean", "std"]).dropna()
        idx = grp.index.astype(float)
        ax.plot(idx, grp["mean"], color=cor, linewidth=2.0, label=raca)
        ax.fill_between(idx,
                        grp["mean"] - grp["std"],
                        grp["mean"] + grp["std"],
                        color=cor, alpha=0.12)

    ax.set_xlabel("Idade (dias)")
    ax.set_ylabel("Peso Médio ± 1 DP (kg)")
    ax.set_title("Curvas de Crescimento Simuladas por Raça\n(0–180 dias)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = f"{output_dir}/crescimento_por_raca.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print(" SIMULAÇÃO SINTÉTICA — MODELO DE PESO BOVINO")
    print(f" n = {N_PER_MODEL:,} por cenário  |  4 cenários  |  seed={SEED}")
    print("=" * 65)

    # ── 8.1  Gerar dados ───────────────────────────────────────────────────
    print("\n[1/5] Gerando banco de dados sintético...")
    df = gerar_dados(N_PER_MODEL, seed=SEED)
    validar_gerador(df)

    # Exportar CSV
    csv_path = f"{OUTPUT_DIR}/dados_sinteticos.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Dataset salvo: {csv_path}")
    print(f"  Shape: {df.shape}   |   Colunas: {df.columns.tolist()}")

    # Estatísticas descritivas
    print("\n  Descritivas do dataset gerado:")
    desc = df[["Idade", "PN", "Brix", "Isolamento", "Peso"]].describe().round(2)
    print(desc.to_string())

    print("\n  Distribuição de raças:")
    print(df["Raca"].value_counts().to_string())

    # ── 8.2  Ajustar os 4 cenários ─────────────────────────────────────────
    print("\n[2/5] Ajustando os 4 modelos OLS...")

    cenarios = [
        ("M1 — Completo",        True,  True),
        ("M2 — Sem Brix",        False, True),
        ("M3 — Sem Isolamento",  True,  False),
        ("M4 — Sem Brix/Isol.", False,  False),
    ]

    models = []
    for nome, inc_brix, inc_iso in cenarios:
        X_mat, y_vec, feat = build_design_matrix(df, inc_brix, inc_iso)
        mdl = OLSModel(nome).fit(X_mat, y_vec, feat)
        models.append(mdl)
        print(f"  {nome:30s}  R²={mdl.r2:.4f}  AIC={mdl.aic:.1f}  BIC={mdl.bic:.1f}")

    # ── 8.3  Tabela comparativa ────────────────────────────────────────────
    print("\n[3/5] Tabela de comparação dos modelos:")
    summary = pd.DataFrame([m.summary_row() for m in models])
    print(summary.to_string(index=False))

    # Melhor modelo por AIC e BIC
    best_aic = summary.loc[summary["AIC"].idxmin(), "Modelo"]
    best_bic = summary.loc[summary["BIC"].idxmin(), "Modelo"]
    print(f"\n  Melhor AIC → {best_aic}")
    print(f"  Melhor BIC → {best_bic}")

    # ── 8.4  Coeficientes do modelo completo ──────────────────────────────
    print("\n[4/5] Coeficientes recuperados — Modelo Completo:")
    coef_m1 = models[0].coef_df()
    print(coef_m1.to_string(index=False))
    print("\n  [Interpretação]")
    print("  β(Idade)     — ganho diário médio por raça de referência (kg/dia)")
    print("  β(PN)        — persistência do peso ao nascimento (deve ser ≈ 1)")
    print("  β(Brix)      — efeito do colostro no crescimento acumulado")
    print("  β(Isolamento)— penalidade por dia de apatia (negativo esperado)")
    print("  β(Sexo)      — vantagem absoluta dos machos (positivo esperado)")
    print("  β(Novilha)   — penalidade por ser filho de primípara (negativo)")

    # ── 8.5  Gráficos ─────────────────────────────────────────────────────
    print("\n[5/5] Gerando gráficos...")
    plot_crescimento_raca(df, OUTPUT_DIR)
    plot_comparativo(models, OUTPUT_DIR)
    plot_diagnosticos(models, OUTPUT_DIR)

    print("\n" + "=" * 65)
    print(" CONCLUÍDO — Arquivos gerados:")
    print(f"   dados_sinteticos.csv")
    print(f"   crescimento_por_raca.png")
    print(f"   comparativo_modelos.png")
    print(f"   diag_M1_*.png  |  diag_M2_*.png  |  diag_M3_*.png  |  diag_M4_*.png")
    print("=" * 65)

    return df, models, summary


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df, models, summary = main()
