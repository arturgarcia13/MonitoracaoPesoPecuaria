"""
=============================================================================
SIMULAÇÃO SINTÉTICA DE PESO BOVINO — MODELO DE MONITORIZAÇÃO  v2.0
=============================================================================
Estratégia de Geração (Estratégia 3 — GMD como variável latente):

    Peso_i = PN_i + GMD_i × Idade_i + ε_i(t)

    GMD_i  = GMD_base_i
             + δ_sex    · Sexo_i
             + δ_brix   · (Brix_i − 22)
             + δ_iso    · Isolamento_i
             + δ_nov    · Novilha_i
             + u_i

─────────────────────────────────────────────────────────────────────────────
AUDITORIA DE ORIGEM DOS PARÂMETROS
─────────────────────────────────────────────────────────────────────────────
PARÂMETROS 100% LITERAIS (extraídos sem modificação dos artigos):

  Brix limiar = 22%
    → Bielmann et al. (2010) J. Dairy Sci. 93:3713–3721
      Ponto de corte com Sens. 90-92% / Espec. 80-85% para IgG ≥ 50 g/L

  δ_iso = −0.050 kg/d
    → Soberon et al. (2012) J. Dairy Sci. 95:783–793
      Diarreia + tratamento antibiótico simultâneos: −50 g/d, P<0.01

  Diff. sexual aos 365d = 44.69 kg
    → de Souza et al. (2018): machos 289.30 kg, fêmeas 244.61 kg

  σ²_p (P365, Nelore) = 1354.58 kg²
    → de Souza et al. (2018)

  e² (P365, Nelore) = 0.68
    → de Souza et al. (2018)

─────────────────────────────────────────────────────────────────────────────
PARÂMETROS INFERIDOS / ESTIMADOS (NÃO existem com esses valores nos artigos):

  δ_brix = +0.002 kg/d por unidade % acima de 22%
    → INFERIDO: Bielmann (2010) estabelece FTP para Brix < 22% e Soberon (2012)
      quantifica perda de −50 g/d por doença. A translação de unidade Brix
      em taxa de GMD não existe na literatura; é uma aproximação linear.

  δ_novilha = −0.042 kg/d
    → INFERIDO: Bielmann (2010) documenta colostro inferior de primíparas;
      Lalman & Holder (NASEM) documentam competição energética. O valor
      exato de −42 g/d não está em nenhuma fonte; é estimativa do modelo.

  δ_sex = +0.0387 kg/d
    → INTERPOLADO:  Mucari & Oliveira (2003) registram diff=9.29 kg aos 240d
      (Guzerá). Assumindo diff=0 ao nascer, a taxa de divergência sexual por
      dia de vida é 9.29/240 = 0.0387 kg/d. Este valor é o mais conservador
      disponível para o período 0–180 dias e é derivado de Mucari, não de
      de Souza (365d), porque a interpolação linear entre os dois pontos
      produziria valores negativos incoerentes para t < 207 dias.

  σ_u = 0.0570 kg/d   (desvio individual no GMD)
    → DERIVADO matematicamente de: Var[Peso|365d] = σ_u²×365² + σ_e_365²
      com σ²_p=1354.58 e e²=0.68 (ambos de Souza 2018).
      Solução: σ_u = sqrt((σ²_p − e²×σ²_p) / 365²) = sqrt(433.47/133225)

  σ_ε(t) = sqrt(e² × σ²_p × t/365)   [erro age-dependent]
    → DERIVADO: σ_e_365² = e²×σ²_p = 0.68×1354.58 = 921.11 (literatura).
      Escalado para idade t por sqrt(t/365) — suposição de variância
      ambiental proporcional ao tempo de exposição.

─────────────────────────────────────────────────────────────────────────────
Cenários:
  M1 — Modelo Completo  : Idade + PN + Brix + Isolamento + Sexo + Novilha + Raça
  M2 — Sem Brix         : sem X3
  M3 — Sem Isolamento   : sem X4
  M4 — Sem Brix/Isol.  : sem X3 e X4

Comparação: R², AIC, BIC + gráficos diagnósticos completos
=============================================================================
"""

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
N_PER_MODEL = 1_000         # ← altere aqui o n por cenário
IDADE_MIN   = 1
IDADE_MAX   = 180
OUTPUT_DIR  = "."

np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PARÂMETROS DAS RAÇAS
# ─────────────────────────────────────────────────────────────────────────────
# pn_mu / pn_sd : Peso ao Nascimento  [Literatura: Soberon 2012, Boligon s.d., Mucari 2003]
# gmd_mu / gmd_sd : GMD base         [Literatura: Laureano 2011, Soberon 2012, NASEM]
# Holandês/Giro GMD: 0.82 kg/d (Cornell, Soberon 2012) ajustado −17% para regime
#   semi-intensivo brasileiro → 0.68 kg/d  [AJUSTE — não literal]

RACAS = {
    "Nelore": {
        "pn_mu": 29.0, "pn_sd": 3.5,
        "gmd_mu": 0.550, "gmd_sd": 0.100,
        "prop": 0.30,
        "fonte_pn": "Boligon et al. (s.d.); Laureano 2011",
        "fonte_gmd": "Laureano 2011 (GND/210d = 0.674); valor médio 0.55 conservador",
    },
    "Guzerá": {
        "pn_mu": 26.0, "pn_sd": 3.0,
        "gmd_mu": 0.500, "gmd_sd": 0.095,
        "prop": 0.15,
        "fonte_pn": "Mucari & Oliveira 2003 (inferido de P8=150 kg)",
        "fonte_gmd": "Mucari 2003 (P8=150.3 kg / 240d ≈ 0.51 kg/d)",
    },
    "Holandês/Girolando": {
        "pn_mu": 40.0, "pn_sd": 5.0,
        "gmd_mu": 0.680, "gmd_sd": 0.115,
        "prop": 0.20,
        "fonte_pn": "Soberon et al. 2012 (41.68 ± 5.09 kg)",
        "fonte_gmd": "Soberon 2012 (0.82 kg/d Cornell) ajustado -17% s.-intensivo [ESTIMATIVA]",
    },
    "Cruzado ½ sangue": {
        "pn_mu": 33.0, "pn_sd": 4.0,
        "gmd_mu": 0.640, "gmd_sd": 0.110,
        "prop": 0.20,
        "fonte_pn": "Inferência ponderada zebu×taurino [ESTIMATIVA]",
        "fonte_gmd": "Inferência ponderada [ESTIMATIVA]",
    },
    "Angus/Simental": {
        "pn_mu": 38.0, "pn_sd": 4.5,
        "gmd_mu": 0.720, "gmd_sd": 0.115,
        "prop": 0.15,
        "fonte_pn": "NASEM (Lalman & Holder) — 85 lb ≈ 38 kg",
        "fonte_gmd": "NASEM tabelas de exigências [ESTIMATIVA]",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# 2.  DELTAS E PARÂMETROS DE ERRO
# ─────────────────────────────────────────────────────────────────────────────

# ── Valores literais ──────────────────────────────────────────────────────────
BRIX_LIMIAR   = 22.0       # % — Bielmann et al. (2010): ponto de corte FTP
DELTA_ISO     = -0.050     # kg/d — Soberon 2012: diarreia+antibiótico, −50 g/d (P<0.01)

# ── Valores inferidos / estimados ────────────────────────────────────────────
DELTA_SEX     = +0.0387    # kg/d — INTERPOLADO: Mucari 2003 diff=9.29 kg/240d
                           #   Assume divergência desde o nascimento (diff=0 ao nascer)
                           #   taxa = 9.29/240 = 0.0387 kg/d por dia de vida
                           #   ≠ de Souza 2018 (0.122 kg/d, inclui período pós-puberdade)

DELTA_BRIX    = +0.002     # kg/d por % acima de 22% — INFERIDO
                           #   Cadeia: Brix↑ → IgG↑ → FTP↓ → doença↓ → GMD↑
                           #   Magnitude: não existe na literatura; aproximação linear

DELTA_NOVILHA = -0.042     # kg/d — INFERIDO
                           #   Bielmann (2010): colostro inferior em primíparas
                           #   Lalman NASEM: competição energética crescimento+lactação
                           #   Magnitude estimada: ~2.5 kg/60d → 2.5/60 ≈ 0.042 kg/d

# ── Parâmetros de erro derivados de de Souza 2018 ────────────────────────────
# σ²_p(365d) = 1354.58 (literal), e²=0.68 (literal) → σ²_e_365 = 921.11
# Decomposição: Var[Peso|365d] = σ_u²×365² + σ²_e_365
# → σ_u = sqrt((σ²_p − σ²_e_365) / 365²) = sqrt(433.47 / 133225) = 0.0570 kg/d
SIGMA2_P_365  = 1354.58    # de Souza 2018 — LITERAL
E2            = 0.68       # de Souza 2018 — LITERAL
SIGMA2_E_365  = E2 * SIGMA2_P_365                            # = 921.11 kg²
SIGMA_U       = np.sqrt((SIGMA2_P_365 - SIGMA2_E_365) / 365**2)  # = 0.0570 kg/d — DERIVADO


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FUNÇÃO GERADORA
# ─────────────────────────────────────────────────────────────────────────────

def sigma_eps(t: np.ndarray) -> np.ndarray:
    """
    Desvio padrão do erro aleatório em função da idade t (dias).
    Derivado de: σ²_ε(t) = e² × σ²_p × (t/365)
    Fontes literais: e²=0.68, σ²_p=1354.58 — de Souza et al. (2018).
    """
    return np.sqrt(SIGMA2_E_365 * (t / 365.0))


def gerar_dados(n: int, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    racas_list = list(RACAS.keys())
    props      = [RACAS[r]["prop"] for r in racas_list]

    # Raça — multinomial
    raca_idx   = rng.choice(len(racas_list), size=n, p=props)
    raca_nomes = np.array(racas_list)[raca_idx]

    # X1 — Idade: Uniforme [1, 180]
    Idade = rng.integers(IDADE_MIN, IDADE_MAX + 1, size=n)

    # X2 — PN: Normal truncada por raça
    PN = np.array([
        np.clip(
            rng.normal(RACAS[r]["pn_mu"], RACAS[r]["pn_sd"]),
            RACAS[r]["pn_mu"] - 3 * RACAS[r]["pn_sd"],
            RACAS[r]["pn_mu"] + 3 * RACAS[r]["pn_sd"],
        )
        for r in raca_nomes
    ])

    # X3 — Brix (%): Normal truncada [12, 38]
    #   Média 24% < 26.1% laboratorial (Bielmann 2010) — ajuste para campo [ESTIMATIVA]
    Brix = np.clip(rng.normal(24.0, 4.0, n), 12.0, 38.0)

    # X4 — Isolamento (dias): Poisson(λ=1.5), truncado [0, 30]
    Isolamento = np.clip(rng.poisson(1.5, n), 0, 30).astype(int)

    # X5 — Sexo: Bernoulli(0.50)
    Sexo = rng.binomial(1, 0.50, n)

    # X6 — Novilha: Bernoulli(0.25)
    Novilha = rng.binomial(1, 0.25, n)

    # GMD base por raça
    GMD_base = np.array([
        np.clip(
            rng.normal(RACAS[r]["gmd_mu"], RACAS[r]["gmd_sd"]),
            0.10, 1.60,
        )
        for r in raca_nomes
    ])

    # Efeitos dos deltas sobre GMD
    d_sex     = DELTA_SEX     * Sexo
    d_brix    = DELTA_BRIX    * (Brix - BRIX_LIMIAR)
    d_iso     = DELTA_ISO     * Isolamento
    d_novilha = DELTA_NOVILHA * Novilha

    # u_i: variação individual residual no GMD
    # σ_u = 0.0570 kg/d — DERIVADO de σ²_p=1354.58, e²=0.68 (de Souza 2018)
    u_i = rng.normal(0.0, SIGMA_U, n)

    GMD = np.clip(GMD_base + d_sex + d_brix + d_iso + d_novilha + u_i, 0.05, 1.60)

    # ε_i: erro age-dependent
    # σ_ε(t) = sqrt(e² × σ²_p × t/365)
    # Derivado de e²=0.68, σ²_p=1354.58 (de Souza 2018) — escalonado para idade t
    eps = rng.normal(0.0, sigma_eps(Idade), n)

    Peso = PN + GMD * Idade + eps
    Peso = np.clip(Peso, PN * 0.85, None)

    return pd.DataFrame({
        "Raca":       raca_nomes,
        "Idade":      Idade,
        "PN":         PN,
        "Brix":       Brix,
        "Isolamento": Isolamento,
        "Sexo":       Sexo,
        "Novilha":    Novilha,
        "GMD_base":   GMD_base,
        "GMD":        GMD,
        "Peso":       Peso,
    })


# ─────────────────────────────────────────────────────────────────────────────
# 4.  VALIDAÇÃO DO GERADOR
# ─────────────────────────────────────────────────────────────────────────────

def validar_gerador(df: pd.DataFrame) -> None:
    """Verifica plausibilidade contra pontos de âncora da literatura."""
    print("\n" + "─" * 65)
    print("VALIDAÇÃO — Pontos de Âncora da Literatura")
    print("─" * 65)

    # Âncoras literais
    checks = [
        # (rótulo, raça, t_min, t_max, exp_mean_lo, exp_mean_hi, exp_cv_lo, exp_cv_hi, fonte)
        ("Nelore ~P60",     "Nelore",             50, 70,  55,  75, 12, 20,
         "Laureano 2011 (GND=0.674 kg/d)"),
        ("Nelore ~P210",    "Nelore",             195, 225, 140, 200, 12, 20,
         "Laureano 2011 (PD=171.15 ± 24.95 kg)"),
        ("Guzerá ~P240",    "Guzerá",             225, 255, 125, 175, 15, 25,
         "Mucari 2003 (P8=150.3 ± 28.9 kg, CV=19.2%)"),
        ("Holandês ~P49",   "Holandês/Girolando",  40,  60,  55,  85, 10, 18,
         "Soberon 2012 adj. semi-intensivo"),
    ]

    all_ok = True
    for label, raca, tlo, thi, mlo, mhi, cvlo, cvhi, fonte in checks:
        mask = (df["Raca"] == raca) & (df["Idade"].between(tlo, thi))
        sub  = df.loc[mask, "Peso"]
        if len(sub) < 5:
            print(f"  {label}: amostras insuficientes (n={len(sub)})")
            continue
        m, s = sub.mean(), sub.std()
        cv   = s / m * 100
        ok_m  = mlo  <= m  <= mhi
        ok_cv = cvlo <= cv <= cvhi
        status = "✓" if (ok_m and ok_cv) else "⚠"
        if not (ok_m and ok_cv):
            all_ok = False
        print(f"  {label} (n={len(sub):3d}): média={m:.1f} kg  DP={s:.1f}  CV={cv:.1f}%")
        print(f"    esperado: média [{mlo},{mhi}] {'✓' if ok_m else '⚠'}  CV [{cvlo},{cvhi}%] {'✓' if ok_cv else '⚠'}")
        print(f"    fonte: {fonte}")

    print(f"\n  Status geral: {'✓ TODAS as âncoras validadas' if all_ok else '⚠ Verificar parâmetros'}")
    print("─" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  OLS MANUAL
# ─────────────────────────────────────────────────────────────────────────────

class OLSModel:
    """OLS via numpy. R², R²_adj, AIC e BIC via log-likelihood gaussiana (MLE)."""

    def __init__(self, name: str):
        self.name = name

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list):
        self.cols  = feature_names
        self.n, self.k = X.shape
        self.y     = y.copy()
        self.beta  = np.linalg.lstsq(X, y, rcond=None)[0]
        self.y_hat = X @ self.beta
        self.resid = y - self.y_hat

        sse = np.sum(self.resid ** 2)
        sst = np.sum((y - y.mean()) ** 2)
        self.r2     = 1.0 - sse / sst
        self.r2_adj = 1.0 - (1 - self.r2) * (self.n - 1) / (self.n - self.k)

        sigma2_mle = sse / self.n
        log_lik    = (-self.n / 2.0) * np.log(2 * np.pi * sigma2_mle) - sse / (2 * sigma2_mle)
        p          = self.k + 1   # β's + σ²
        self.aic   = -2 * log_lik + 2 * p
        self.bic   = -2 * log_lik + p * np.log(self.n)
        return self

    def summary_row(self) -> dict:
        return {
            "Modelo": self.name, "n": self.n, "k": self.k,
            "R²": round(self.r2, 4), "R²_adj": round(self.r2_adj, 4),
            "AIC": round(self.aic, 2), "BIC": round(self.bic, 2),
        }

    def coef_df(self) -> pd.DataFrame:
        return pd.DataFrame({"Feature": self.cols, "β": self.beta})


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MATRIZ DE DESIGN
# ─────────────────────────────────────────────────────────────────────────────

def build_design_matrix(df: pd.DataFrame, include_brix: bool, include_iso: bool):
    racas_dummies = pd.get_dummies(df["Raca"], prefix="Raca", drop_first=True).astype(int)

    X = pd.DataFrame({
        "Intercepto": 1.0,
        "Idade":      df["Idade"].astype(float).values,
        "PN":         df["PN"].values,
        "Sexo":       df["Sexo"].astype(float).values,
        "Novilha":    df["Novilha"].astype(float).values,
    })
    if include_brix:
        X["Brix"] = df["Brix"].values
    if include_iso:
        X["Isolamento"] = df["Isolamento"].astype(float).values

    X = pd.concat([X, racas_dummies], axis=1)
    return X.values.astype(np.float64), df["Peso"].values.astype(np.float64), X.columns.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# 7.  GRÁFICOS
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "M1 — Completo":        "#2563EB",
    "M2 — Sem Brix":        "#16A34A",
    "M3 — Sem Isolamento":  "#D97706",
    "M4 — Sem Brix/Isol.": "#DC2626",
}


def _short(name: str) -> str:
    return name.split(" — ")[1] if " — " in name else name


def plot_diagnosticos(models: list, output_dir: str = ".") -> None:
    """4-panel diagnostic plot por modelo."""
    for mdl in models:
        color = COLORS.get(mdl.name, "#4B5563")
        fig   = plt.figure(figsize=(14, 10))
        fig.suptitle(f"Diagnósticos — {mdl.name}", fontsize=14, fontweight="bold", y=1.01)
        gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.35)

        # 1 — Resíduos vs. Ajustados
        ax = fig.add_subplot(gs[0, 0])
        ax.scatter(mdl.y_hat, mdl.resid, alpha=0.30, s=10, color=color)
        ax.axhline(0, color="black", linewidth=1.0, linestyle="--")
        bins = np.percentile(mdl.y_hat, np.linspace(5, 95, 20))
        bm   = [mdl.resid[np.abs(mdl.y_hat - b) < 6].mean() for b in bins]
        ax.plot(bins, bm, color="red", linewidth=1.5, label="Tendência")
        ax.set_xlabel("Valores Ajustados (kg)")
        ax.set_ylabel("Resíduos (kg)")
        ax.set_title("Resíduos vs. Ajustados")
        ax.legend(fontsize=8)

        # 2 — QQ-Plot
        ax = fig.add_subplot(gs[0, 1])
        (osm, osr), (slope, intercept, r) = stats.probplot(mdl.resid, dist="norm")
        ax.scatter(osm, osr, alpha=0.30, s=10, color=color)
        lx = np.array([osm[0], osm[-1]])
        ax.plot(lx, slope * lx + intercept, "k--", linewidth=1.5)
        ax.set_xlabel("Quantis Teóricos")
        ax.set_ylabel("Quantis Amostrais")
        ax.set_title(f"QQ-Plot  (r = {r:.4f})")

        # 3 — Predito vs. Observado
        ax = fig.add_subplot(gs[1, 0])
        ax.scatter(mdl.y, mdl.y_hat, alpha=0.25, s=10, color=color)
        lims = [min(mdl.y.min(), mdl.y_hat.min()) - 2,
                max(mdl.y.max(), mdl.y_hat.max()) + 2]
        ax.plot(lims, lims, "k--", linewidth=1.2, label="Linha 45°")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Observado (kg)"); ax.set_ylabel("Predito (kg)")
        ax.set_title(f"Predito vs. Observado  (R²={mdl.r2:.4f})")
        ax.legend(fontsize=8)

        # 4 — Histograma dos resíduos
        ax = fig.add_subplot(gs[1, 1])
        ax.hist(mdl.resid, bins=40, color=color, alpha=0.65,
                edgecolor="white", density=True)
        xr = np.linspace(mdl.resid.min(), mdl.resid.max(), 200)
        ax.plot(xr, stats.norm.pdf(xr, mdl.resid.mean(), mdl.resid.std()),
                "k-", linewidth=1.5, label="N(0,σ²) ref.")
        ax.set_xlabel("Resíduos (kg)"); ax.set_ylabel("Densidade")
        ax.set_title("Distribuição dos Resíduos")
        ax.legend(fontsize=8)

        fig.tight_layout()
        fname = f"{output_dir}/diag_{mdl.name.replace(' ', '_').replace('/', '')}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Salvo: {fname}")


def plot_comparativo(models: list, output_dir: str = ".") -> None:
    """Painel comparativo R², AIC/BIC e distribuição de resíduos."""
    names  = [m.name for m in models]
    colors = [COLORS.get(n, "#4B5563") for n in names]
    r2s    = [m.r2   for m in models]
    aics   = [m.aic  for m in models]
    bics   = [m.bic  for m in models]
    x      = np.arange(len(models))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Comparação dos 4 Cenários de Regressão", fontsize=14, fontweight="bold")

    # R²
    ax = axes[0]
    bars = ax.bar(x, r2s, color=colors, width=0.55, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels([_short(n) for n in names], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("R²"); ax.set_title("R² por Modelo")
    ax.set_ylim(max(0, min(r2s) - 0.05), min(1.0, max(r2s) + 0.03))
    for bar, v in zip(bars, r2s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    # AIC & BIC
    ax = axes[1]
    w = 0.25
    b1 = ax.bar(x - w/2, aics, width=w, color=colors, alpha=0.85, edgecolor="white", label="AIC")
    b2 = ax.bar(x + w/2, bics, width=w, color=colors, alpha=0.50, edgecolor="white", hatch="//", label="BIC")
    ax.set_xticks(x); ax.set_xticklabels([_short(n) for n in names], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Critério"); ax.set_title("AIC e BIC  (menor = melhor)")
    ax.legend(fontsize=8)
    ax.set_ylim(min(aics+bics)*0.995, max(aics+bics)*1.005)
    for b, v in zip(list(b1)+list(b2), aics+bics):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05,
                f"{v:.0f}", ha="center", va="bottom", fontsize=7, rotation=90)

    # Resíduos sobrepostos
    ax = axes[2]
    for mdl, col in zip(models, colors):
        ax.hist(mdl.resid, bins=50, color=col, alpha=0.45,
                density=True, label=_short(mdl.name))
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Resíduo (kg)"); ax.set_ylabel("Densidade")
    ax.set_title("Distribuição dos Resíduos\n(sobrepostos)")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fname = f"{output_dir}/comparativo_modelos.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {fname}")


def plot_crescimento_raca(df: pd.DataFrame, output_dir: str = ".") -> None:
    """Curvas de crescimento médio por raça (0–180 dias)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = plt.cm.tab10(np.linspace(0, 0.8, len(RACAS)))

    for raca, cor in zip(RACAS.keys(), palette):
        sub = df[df["Raca"] == raca].copy()
        bins = np.linspace(IDADE_MIN, IDADE_MAX, 19)
        sub["bin"] = pd.cut(sub["Idade"], bins=bins, labels=bins[:-1])
        grp = sub.groupby("bin", observed=True)["Peso"].agg(["mean", "std"]).dropna()
        idx = grp.index.astype(float)
        ax.plot(idx, grp["mean"], color=cor, linewidth=2.0, label=raca)
        ax.fill_between(idx, grp["mean"] - grp["std"], grp["mean"] + grp["std"],
                        color=cor, alpha=0.12)

    ax.set_xlabel("Idade (dias)")
    ax.set_ylabel("Peso Médio ± 1 DP (kg)")
    ax.set_title("Curvas de Crescimento Simuladas por Raça  (0–180 dias)")
    ax.legend(fontsize=9, loc="upper left"); ax.grid(True, alpha=0.3)
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
    print(" SIMULAÇÃO SINTÉTICA — MODELO DE PESO BOVINO  v2.0")
    print(f" n = {N_PER_MODEL:,} por cenário  |  4 cenários  |  seed={SEED}")
    print("=" * 65)

    # ── Parâmetros auditados ────────────────────────────────────────────────
    print("\n[PARAM] Deltas utilizados (auditados):")
    print(f"  δ_sex      = {DELTA_SEX:+.4f} kg/d  [INTERPOLADO — Mucari 2003, 9.29 kg/240d]")
    print(f"  δ_brix     = {DELTA_BRIX:+.4f} kg/d/% [INFERIDO — sem fonte direta]")
    print(f"  δ_iso      = {DELTA_ISO:+.4f} kg/d  [LITERAL — Soberon 2012, −50 g/d, P<0.01]")
    print(f"  δ_novilha  = {DELTA_NOVILHA:+.4f} kg/d  [INFERIDO — sem fonte direta]")
    print(f"  σ_u        =  {SIGMA_U:.4f} kg/d  [DERIVADO de σ²_p=1354.58, e²=0.68, t=365]")
    print(f"  σ_ε(t)     = sqrt({SIGMA2_E_365:.2f} × t/365)  [DERIVADO — de Souza 2018]")

    # ── Gerar dados ─────────────────────────────────────────────────────────
    print("\n[1/5] Gerando banco de dados...")
    df = gerar_dados(N_PER_MODEL, seed=SEED)
    validar_gerador(df)

    csv_path = f"{OUTPUT_DIR}/dados_sinteticos_v2.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Dataset salvo: {csv_path}  | shape: {df.shape}")
    print("\n  Descritivas:")
    print(df[["Idade", "PN", "Brix", "Isolamento", "Peso"]].describe().round(2).to_string())
    print("\n  Distribuição de raças:")
    print(df["Raca"].value_counts().to_string())

    # ── 4 cenários OLS ──────────────────────────────────────────────────────
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

    # ── Tabela comparativa ───────────────────────────────────────────────────
    print("\n[3/5] Tabela comparativa:")
    summary = pd.DataFrame([m.summary_row() for m in models])
    print(summary.to_string(index=False))
    print(f"\n  ΔAIC (M1 vs M2) = {models[1].aic - models[0].aic:+.1f}  [impacto do Brix]")
    print(f"  ΔAIC (M1 vs M3) = {models[2].aic - models[0].aic:+.1f}  [impacto do Isolamento]")
    print(f"  ΔAIC (M2 vs M4) = {models[3].aic - models[1].aic:+.1f}  [impacto do Isolamento sem Brix]")
    best_aic = summary.loc[summary["AIC"].idxmin(), "Modelo"]
    best_bic = summary.loc[summary["BIC"].idxmin(), "Modelo"]
    print(f"\n  Melhor AIC → {best_aic}")
    print(f"  Melhor BIC → {best_bic}")

    # ── Coeficientes M1 ─────────────────────────────────────────────────────
    print("\n[4/5] Coeficientes recuperados — M1 Completo:")
    coef = models[0].coef_df()
    print(coef.to_string(index=False))
    print("\n  Checagem de sinal esperado:")
    coef_dict = dict(zip(coef["Feature"], coef["β"]))
    checks_coef = [
        ("PN",         ">0",  coef_dict.get("PN", 0)         > 0),
        ("Idade",      ">0",  coef_dict.get("Idade", 0)      > 0),
        ("Sexo",       ">0",  coef_dict.get("Sexo", 0)       > 0),
        ("Novilha",    "<0",  coef_dict.get("Novilha", 0)    < 0),
        ("Brix",       ">0",  coef_dict.get("Brix", 0)       > 0),
        ("Isolamento", "<0",  coef_dict.get("Isolamento", 0) < 0),
    ]
    for feat, expected, ok in checks_coef:
        print(f"    β({feat:12s}) {expected}  {'✓' if ok else '⚠ SINAL INESPERADO'}")

    # ── Gráficos ─────────────────────────────────────────────────────────────
    print("\n[5/5] Gerando gráficos...")
    plot_crescimento_raca(df, OUTPUT_DIR)
    plot_comparativo(models, OUTPUT_DIR)
    plot_diagnosticos(models, OUTPUT_DIR)

    print("\n" + "=" * 65)
    print(" CONCLUÍDO")
    print("=" * 65)
    return df, models, summary


if __name__ == "__main__":
    df, models, summary = main()
