"""
=============================================================================
SIMULAÇÃO SINTÉTICA DE PESO BOVINO  v5.0
Modelo final: ln(Peso) ~ Idade + Idade² + PN + Sexo + Raça
=============================================================================

MOTIVAÇÃO DA TRANSFORMAÇÃO (justificativa estatística, não literária)
──────────────────────────────────────────────────────────────────────
O modelo linear v4 apresentou heterocedasticidade severa confirmada por:
  • Breusch-Pagan: LM=144,84, p≈0  (variância cresce com ŷ)
  • Correlação |resíduo| vs. ŷ: r=0,42

Abordagens testadas e resultados (Breusch-Pagan LM / AIC):
  Linear base                   LM=144,84  AIC=10156,8
  ln(Peso) ~ Idade              LM= 44,04  AIC=10051,4
  WLS (w=1/Idade)               LM=159,12  AIC=10216,4   ← piorou
  WLS (w=1/ŷ²)                  LM=159,54  AIC=10236,1   ← piorou
  ln(Peso) ~ ln(Idade)          LM=111,83  AIC=10144,3
  ln(Peso) ~ Idade + Idade²     LM= 26,72  AIC= 9764,4   ← MELHOR

Redução da heterocedasticidade: 82% (LM 144→26).
Melhora de AIC: −392 pontos vs. linear base.

MODELO MQO FINAL
──────────────────────────────────────────────────────────────────────
  ln(Peso_i) = β₀ + β₁·Idade_i + β₂·Idade_i² + β₃·PN_i
             + β₄·Sexo_i + Σ γ_r·Raça_r_i + ε_i

  R²=0,8717  R²_adj=0,8707  AIC=9764,4  BIC=9813,5

FUNÇÃO DE ALERTA
──────────────────────────────────────────────────────────────────────
  Critério: ln(Peso_real) < ln(Ŷ(t)) − σ_log(t)
  Equivalente: Peso_real < Ŷ(t) × exp(−σ_log(t))

  σ_log(t) = σ_ε(t) / Ŷ(t)   [método delta]
  σ_ε(t)   = sqrt(e² × σ²_p × t/365) = sqrt(921,11 × t/365)
    → e²=0,68 e σ²_p=1354,58 de de Souza et al. (2018) — LITERAIS

MAPA DE ORIGEM — PARÂMETROS DO GERADOR
──────────────────────────────────────────────────────────────────────
  🟢 LITERAL
    diff_sex P365 = 44,69 kg       de Souza 2018
    diff_sex P550 = 83,87 kg       de Souza 2018
    σ²_p (P365)  = 1354,58 kg²    de Souza 2018
    e² (P365)    = 0,68            de Souza 2018
    GND Nelore   = 141,47 kg       Laureano 2011
    dias desmama = 210 d           Laureano 2011
    P8 Guzerá    = 150,34 kg       Mucari 2003
    σ²_F P8 Guz. = 661,70 kg²     Mucari 2003
    ADG Cornell  = 0,820 kg/d      Soberon 2012
    SD ADG C.    = 0,180 kg/d      Soberon 2012
    SD ADG com.  = 0,110 kg/d      Soberon 2012

  🟡 ARITMÉTICA / ÁLGEBRA
    GMD_Nelore   = 141,47/210      Laureano 2011
    GMD_Guzerá   = 150,34/240      Mucari 2003
    GMD_Cruzado  = (Nelore+Hol)/2  Laureano + Soberon
    b_sex, a_sex = reta P365/P550  de Souza 2018
    σ_u          = sqrt(Δσ²/365²)  de Souza 2018
    σ_ε(t)       = sqrt(921,11×t/365) de Souza 2018

  ❌ REMOVIDO
    Brix, Novilha, Isolamento — sem delta quantificado no corpus
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
# 0.  CONFIGURAÇÃO
# ─────────────────────────────────────────────────────────────────────────────
SEED       = 42
N_ANIMAIS  = 1_000      # ← altere aqui
IDADE_MIN  = 1
IDADE_MAX  = 365
OUTPUT_DIR = "."

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONSTANTES LITERAIS — de Souza 2018
# ─────────────────────────────────────────────────────────────────────────────
SIGMA2_P_365  = 1354.58
E2            = 0.68
DIFF_SEX_P365 = 44.69
DIFF_SEX_P550 = 83.87
T_P365, T_P550 = 365.0, 550.0

# ─────────────────────────────────────────────────────────────────────────────
# 2.  DERIVAÇÕES ALGÉBRICAS
# ─────────────────────────────────────────────────────────────────────────────
_B_SEX      = (DIFF_SEX_P550 - DIFF_SEX_P365) / (T_P550 - T_P365)  # 0.211784
_A_SEX      = DIFF_SEX_P365 - _B_SEX * T_P365                       # -32.6111
T0_SEX      = -_A_SEX / _B_SEX                                       # 154.0 d
_SIGMA2_E   = E2 * SIGMA2_P_365                                      # 921.11
_SIGMA2_U   = (SIGMA2_P_365 - _SIGMA2_E) / T_P365**2                 # 0.003253
SIGMA_U     = np.sqrt(_SIGMA2_U)                                     # 0.05704


def diff_sex(t):
    """Diferença de peso M−F em kg. Reta por (P365,44.69) e (P550,83.87)."""
    return np.maximum(0.0, _A_SEX + _B_SEX * np.asarray(t, float))


def sigma_eps(t):
    """σ_ε(t) = sqrt(e²·σ²_p·t/365) — de Souza 2018."""
    return np.sqrt(_SIGMA2_E * np.asarray(t, float) / T_P365)


def sigma_log(t, yhat):
    """
    σ na escala log via método delta: σ_log(t) ≈ σ_ε(t) / Ŷ(t).
    Converte o desvio absoluto de de Souza 2018 para a escala logarítmica.
    Usado no limiar de alerta age-specific.
    """
    return sigma_eps(t) / np.asarray(yhat, float)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  RAÇAS
# ─────────────────────────────────────────────────────────────────────────────
RACAS = {
    "Nelore":            (29.0, 3.5,  141.47/210,                  np.sqrt(SIGMA2_P_365)/T_P365, 0.30),
    "Guzerá":            (26.0, 3.0,  150.34/240,                  np.sqrt(661.70)/240,           0.15),
    "Holandês/Girolando":(40.0, 5.0,  0.820,                       0.180,                         0.20),
    "Cruzado ½ sangue":  (33.0, 4.0,  (141.47/210 + 0.820)/2,      0.110,                         0.20),
    "Angus/Simental":    (38.0, 4.5,  0.820,                       0.110,                         0.15),
}
# Média ponderada para "Geral"
_r = [r for r in RACAS]
RACAS["Geral"] = (
    sum(RACAS[r][0]*RACAS[r][4] for r in _r),
    sum(RACAS[r][1]*RACAS[r][4] for r in _r),
    sum(RACAS[r][2]*RACAS[r][4] for r in _r),
    sum(RACAS[r][3]*RACAS[r][4] for r in _r),
    None,
)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  GERADOR
# ─────────────────────────────────────────────────────────────────────────────

def gerar_dados(n: int, seed: int = SEED) -> pd.DataFrame:
    rng        = np.random.default_rng(seed)
    base_racas = [r for r in RACAS if r != "Geral"]
    props      = [RACAS[r][4] for r in base_racas]
    idx        = rng.choice(len(base_racas), size=n, p=props)
    raca_nomes = np.array(base_racas)[idx]

    Idade = rng.integers(IDADE_MIN, IDADE_MAX + 1, n).astype(float)
    PN    = np.array([np.clip(rng.normal(RACAS[r][0], RACAS[r][1]),
                              RACAS[r][0]-3*RACAS[r][1],
                              RACAS[r][0]+3*RACAS[r][1]) for r in raca_nomes])
    Sexo  = rng.binomial(1, 0.50, n).astype(float)

    GMD   = np.array([np.clip(rng.normal(RACAS[r][2], RACAS[r][3]), 0.10, 1.60)
                      for r in raca_nomes])
    GMD  += rng.normal(0.0, SIGMA_U, n)
    GMD   = np.clip(GMD, 0.05, 1.60)

    eps   = rng.normal(0.0, sigma_eps(Idade), n)
    dsex  = diff_sex(Idade)
    Peso  = PN + GMD * Idade + (Sexo - 0.5) * dsex + eps
    Peso  = np.clip(Peso, PN * 0.85, None)

    return pd.DataFrame({"Raca": raca_nomes, "Idade": Idade.astype(int),
                         "PN": PN, "Sexo": Sexo.astype(int),
                         "GMD": GMD, "Peso": Peso})

# ─────────────────────────────────────────────────────────────────────────────
# 5.  VALIDAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def validar_gerador(df: pd.DataFrame) -> None:
    print("\n" + "─"*65)
    print("VALIDAÇÃO — Âncoras Literais")
    print("─"*65)
    checks = [
        ("Nelore ~P210",           "Nelore",            200,220, 140,205, "Laureano 2011: 171,15 kg"),
        ("Guzerá ~P240",           "Guzerá",            230,250, 105,195, "Mucari 2003: 150,34 kg"),
        ("Holandês ~P49",          "Holandês/Girolando", 40, 60,  60,110, "Soberon 2012: ~82 kg"),
        ("Nelore P365 machos",     "Nelore",            355,375, 255,325, "de Souza 2018: 289,30 kg"),
        ("Nelore P365 fêmeas",     "Nelore",            355,375, 210,280, "de Souza 2018: 244,61 kg"),
    ]
    all_ok = True
    for label, raca, tlo, thi, elo, ehi, fonte in checks:
        mask = (df["Raca"]==raca) & df["Idade"].between(tlo,thi)
        if "macho" in label:  mask &= df["Sexo"]==1
        elif "fêmea" in label: mask &= df["Sexo"]==0
        sub = df.loc[mask,"Peso"]
        if len(sub) < 3:
            print(f"  — {label}: n={len(sub)} insuficiente"); continue
        m, s = sub.mean(), sub.std()
        ok = elo <= m <= ehi
        if not ok: all_ok = False
        print(f"  {'✓' if ok else '⚠'} {label}: {m:.1f}±{s:.1f} kg (n={len(sub)}, esp.[{elo},{ehi}])")
        print(f"      {fonte}")
    print("─"*65)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  OLS — ln(Peso) ~ Idade + Idade² + PN + Sexo + Raça
# ─────────────────────────────────────────────────────────────────────────────

class OLSModel:
    def fit(self, X: np.ndarray, y_log: np.ndarray, y_orig: np.ndarray,
            feature_names: list):
        self.cols   = feature_names
        self.n, self.k = X.shape
        self.y_log  = y_log.copy()
        self.y_orig = y_orig.copy()
        self.beta   = np.linalg.lstsq(X, y_log, rcond=None)[0]
        self.yh_log = X @ self.beta
        self.resid  = y_log - self.yh_log

        sse = np.sum(self.resid**2)
        sst = np.sum((y_log - y_log.mean())**2)
        self.r2     = 1 - sse/sst
        self.r2_adj = 1 - (1-self.r2)*(self.n-1)/(self.n-self.k)
        self.sigma_res = np.std(self.resid, ddof=self.k)

        # AIC/BIC com jacobiano (transformação log)
        s2  = sse/self.n
        ll  = (-self.n/2*np.log(2*np.pi*s2) - sse/(2*s2)
               - np.sum(np.log(y_orig)))
        p   = self.k + 1
        self.aic = -2*ll + 2*p
        self.bic = -2*ll + p*np.log(self.n)
        return self

    def predict_log(self, X: np.ndarray) -> np.ndarray:
        return X @ self.beta

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.predict_log(X))

    def coef_df(self):
        return pd.DataFrame({"Feature": self.cols, "β": self.beta})


def build_matrix(df: pd.DataFrame):
    """Constrói X com Idade, Idade², PN, Sexo, Dummies de Raça."""
    dummies = pd.get_dummies(df["Raca"], prefix="Raca", drop_first=True).astype(int)
    Idade   = df["Idade"].astype(float).values
    X = pd.DataFrame({
        "Intercepto": 1.0,
        "Idade":      Idade,
        "Idade2":     Idade**2,
        "PN":         df["PN"].values,
        "Sexo":       df["Sexo"].astype(float).values,
    })
    X = pd.concat([X, dummies], axis=1)
    return (X.values.astype(np.float64),
            np.log(df["Peso"].values.astype(np.float64)),
            df["Peso"].values.astype(np.float64),
            X.columns.tolist())

# ─────────────────────────────────────────────────────────────────────────────
# 7.  FUNÇÃO DE ALERTA
# ─────────────────────────────────────────────────────────────────────────────

def verificar_alerta(modelo: OLSModel, feature_names: list,
                     idade: float, pn: float, sexo: int,
                     raca: str, peso_real: float) -> dict:
    """
    Avalia se o peso real de um animal está abaixo do limiar de alerta.

    Critério (escala log, método delta):
        ALERTA  se  ln(peso_real) < ln(Ŷ) − σ_log(t)
        ↔           peso_real    < Ŷ × exp(−σ_log(t))

    σ_log(t) = σ_ε(t) / Ŷ(t)   com σ_ε(t) = sqrt(921,11 × t/365)
    Fonte: e²=0,68, σ²_p=1354,58 — de Souza 2018.
    """
    row = {"Intercepto":1.0, "Idade":float(idade),
           "Idade2":float(idade)**2, "PN":float(pn), "Sexo":float(sexo)}
    for f in feature_names:
        if f.startswith("Raca_"):
            row[f] = 0
    rc = f"Raca_{raca}"
    if rc in feature_names:
        row[rc] = 1

    x_vec   = np.array([row.get(f, 0.0) for f in feature_names])
    ln_yhat = float(modelo.predict_log(x_vec.reshape(1,-1))[0])
    yhat    = float(np.exp(ln_yhat))

    # σ_log age-specific via método delta
    s_log   = float(sigma_log(np.array([idade]), np.array([yhat]))[0])
    limiar  = yhat * np.exp(-s_log)
    ln_real = np.log(peso_real)
    desvio_log = ln_real - ln_yhat          # em unidades log
    desvio_dp  = desvio_log / s_log         # em nº de DP log
    alerta  = peso_real < limiar

    pct_desvio = (peso_real/yhat - 1) * 100

    if alerta:
        msg = (f"⚠ ALERTA: {peso_real:.1f} kg  |  esperado {yhat:.1f} kg  "
               f"({pct_desvio:+.1f}%)  |  limiar {limiar:.1f} kg  "
               f"|  {desvio_dp:.1f} DP  →  avaliação recomendada")
    else:
        msg = (f"✓ OK: {peso_real:.1f} kg  |  esperado {yhat:.1f} kg  "
               f"({pct_desvio:+.1f}%)  |  limiar {limiar:.1f} kg")

    return {"y_hat": yhat, "limiar": limiar, "ln_yhat": ln_yhat,
            "sigma_log": s_log, "desvio_pct": pct_desvio,
            "desvio_dp": desvio_dp, "alerta": alerta, "mensagem": msg}

# ─────────────────────────────────────────────────────────────────────────────
# 8.  GRÁFICOS
# ─────────────────────────────────────────────────────────────────────────────

def plot_diagnosticos(modelo: OLSModel, output_dir: str = ".") -> None:
    """4-panel diagnóstico: resíduos na escala log."""
    color = "#16A34A"
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Diagnósticos — ln(Peso) ~ Idade + Idade² + PN + Sexo + Raça\n"
        f"R²={modelo.r2:.4f}  R²_adj={modelo.r2_adj:.4f}  "
        f"AIC={modelo.aic:.1f}  BIC={modelo.bic:.1f}",
        fontsize=12, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)

    def smooth(xv, yv, bw=0.08):
        bw_abs = (xv.max()-xv.min())*bw
        bs = np.percentile(xv, np.linspace(5,95,16))
        return bs, [yv[np.abs(xv-b)<bw_abs].mean()
                    if np.any(np.abs(xv-b)<bw_abs) else np.nan for b in bs]

    ax = fig.add_subplot(gs[0,0])
    ax.scatter(modelo.yh_log, modelo.resid, alpha=0.22, s=9, color=color)
    ax.axhline(0, color="black", lw=1.0, ls="--")
    bx,by = smooth(modelo.yh_log, modelo.resid)
    ax.plot(bx, by, "r-", lw=1.8, label="Tendência")
    ax.set_xlabel("ln(Ŷ)"); ax.set_ylabel("Resíduo (log)")
    ax.set_title("Resíduos vs. Ajustados"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0,1])
    sa = np.sqrt(np.abs(modelo.resid / modelo.resid.std()))
    ax.scatter(modelo.yh_log, sa, alpha=0.22, s=9, color=color)
    bx2,by2 = smooth(modelo.yh_log, sa)
    ax.plot(bx2, by2, "r-", lw=1.8)
    ax.set_xlabel("ln(Ŷ)"); ax.set_ylabel("√|Resíduo std.|")
    ax.set_title("Scale-Location  (homocedasticidade)")

    ax = fig.add_subplot(gs[1,0])
    (osm,osr),(slope,intercept,r_qq) = stats.probplot(modelo.resid, dist="norm")
    ax.scatter(osm, osr, alpha=0.22, s=9, color=color)
    lx = np.array([osm[0], osm[-1]])
    ax.plot(lx, slope*lx+intercept, "k--", lw=1.5)
    ax.set_xlabel("Quantis Teóricos"); ax.set_ylabel("Quantis Amostrais")
    ax.set_title(f"QQ-Plot  (r={r_qq:.4f})")

    ax = fig.add_subplot(gs[1,1])
    ax.hist(modelo.resid, bins=42, color=color, alpha=0.65,
            edgecolor="white", density=True)
    xr = np.linspace(modelo.resid.min(), modelo.resid.max(), 200)
    ax.plot(xr, stats.norm.pdf(xr, modelo.resid.mean(), modelo.resid.std()),
            "k-", lw=1.5, label="N(0,σ²) ref.")
    ax.set_xlabel("Resíduo (log)"); ax.set_ylabel("Densidade")
    ax.set_title("Distribuição dos Resíduos"); ax.legend(fontsize=8)

    fig.tight_layout()
    fname = f"{output_dir}/diagnosticos_v5.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {fname}")


def plot_curvas_alerta(modelo: OLSModel, feature_names: list,
                       output_dir: str = ".") -> None:
    """Curvas de crescimento com faixa de alerta age-specific."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Curvas de Crescimento e Limiar de Alerta  (v5.0)",
                 fontsize=13, fontweight="bold")

    t_range = np.arange(1, 366)
    palette = plt.cm.tab10(np.linspace(0, 0.8, 5))
    racas_plot = ["Nelore","Guzerá","Holandês/Girolando",
                  "Cruzado ½ sangue","Angus/Simental"]

    # ── Painel esquerdo: curvas preditas M e F por raça ──────────────────
    ax = axes[0]
    for raca, cor in zip(racas_plot, palette):
        for sexo, ls, lbl_suf in [(1,"-","M"),(0,"--","F")]:
            preds = []
            for t in t_range:
                row = {"Intercepto":1.0,"Idade":float(t),"Idade2":float(t)**2,
                       "PN":RACAS[raca][0],"Sexo":float(sexo)}
                for f in feature_names:
                    if f.startswith("Raca_"): row[f]=0
                rc = f"Raca_{raca}"
                if rc in feature_names: row[rc]=1
                x = np.array([row.get(f,0.0) for f in feature_names])
                preds.append(modelo.predict(x.reshape(1,-1))[0])
            lbl = f"{raca} {lbl_suf}" if sexo==1 else None
            ax.plot(t_range, preds, color=cor, lw=1.8, ls=ls, label=lbl)

    # Âncoras literais
    for t,w,m in [(210,171.15,"★"),(240,150.34,"★"),
                  (49,82.1,"★"),(365,289.30,"★"),(365,244.61,"★")]:
        ax.plot(t, w, "k*", ms=10, zorder=6)

    ax.set_xlabel("Idade (dias)"); ax.set_ylabel("Peso (kg)")
    ax.set_title("Predições por Raça e Sexo\n(★ = âncoras literais)")
    ax.legend(fontsize=6.5, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.22)

    # ── Painel direito: faixa de alerta para Nelore M e F ────────────────
    ax = axes[1]
    for sexo, cor, lbl in [(1,"#2563EB","Macho"),(0,"#DC2626","Fêmea")]:
        preds, limiares = [], []
        for t in t_range:
            row = {"Intercepto":1.0,"Idade":float(t),"Idade2":float(t)**2,
                   "PN":29.0,"Sexo":float(sexo)}
            for f in feature_names:
                if f.startswith("Raca_"): row[f]=0
            row["Raca_Nelore"] = 1
            x    = np.array([row.get(f,0.0) for f in feature_names])
            yhat = float(modelo.predict(x.reshape(1,-1))[0])
            lim  = yhat * np.exp(-float(sigma_log(np.array([t]),np.array([yhat]))[0]))
            preds.append(yhat); limiares.append(lim)
        preds    = np.array(preds)
        limiares = np.array(limiares)
        ax.plot(t_range, preds, color=cor, lw=2.2, label=f"Ŷ {lbl}")
        ax.fill_between(t_range, limiares, preds, color=cor, alpha=0.14)
        ax.plot(t_range, limiares, color=cor, lw=1.4, ls="--",
                label=f"Limiar {lbl}")

    # Âncoras literais Nelore
    ax.plot(365, 289.30, "b*", ms=12, zorder=7, label="P365 M lit. (289,30 kg)")
    ax.plot(365, 244.61, "r*", ms=12, zorder=7, label="P365 F lit. (244,61 kg)")

    ax.set_xlabel("Idade (dias)"); ax.set_ylabel("Peso (kg)")
    ax.set_title("Limiar de Alerta — Nelore PN=29 kg\n"
                 "Faixa: Ŷ(t) a Ŷ(t)×exp(−σ_log(t))")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.22)

    fig.tight_layout()
    fname = f"{output_dir}/curvas_alerta_v5.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {fname}")


def plot_comparativo_modelos(output_dir: str = ".") -> None:
    """Barra comparativa das 6 abordagens testadas."""
    nomes = ["1.Linear","2.ln(Y)~Age","3.WLS(1/t)",
             "4.WLS(1/ŷ²)","5.ln(Y)~ln(t)","6.ln(Y)~t+t²"]
    aics  = [10156.8, 10051.4, 10216.4, 10236.1, 10144.3, 9764.4]
    bps   = [144.84,   44.04,  159.12,  159.54,  111.83,  26.72]
    cores = ["#DC2626","#F97316","#DC2626","#DC2626","#F97316","#16A34A"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Comparação das 6 Abordagens Testadas para Heterocedasticidade",
                 fontsize=12, fontweight="bold")

    x = np.arange(len(nomes))
    bars = ax1.bar(x, aics, color=cores, width=0.6, edgecolor="white")
    ax1.set_xticks(x); ax1.set_xticklabels(nomes, rotation=25, ha="right", fontsize=8)
    ax1.set_ylabel("AIC"); ax1.set_title("AIC (menor = melhor)")
    ax1.set_ylim(min(aics)*0.994, max(aics)*1.003)
    for b, v in zip(bars, aics):
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()+5,
                 f"{v:.0f}", ha="center", va="bottom", fontsize=7.5)

    bars2 = ax2.bar(x, bps, color=cores, width=0.6, edgecolor="white")
    ax2.axhline(stats.chi2.ppf(0.95, df=8), color="black", ls="--",
                lw=1.2, label="Limiar BP 5% (χ²)")
    ax2.set_xticks(x); ax2.set_xticklabels(nomes, rotation=25, ha="right", fontsize=8)
    ax2.set_ylabel("Breusch-Pagan LM"); ax2.set_title("BP LM (menor = menos heteroced.)")
    ax2.legend(fontsize=8)
    for b, v in zip(bars2, bps):
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+1,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)

    fig.tight_layout()
    fname = f"{output_dir}/comparativo_abordagens_v5.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {fname}")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("="*65)
    print(" SIMULAÇÃO DE PESO BOVINO  v5.0")
    print(" ln(Peso) ~ Idade + Idade² + PN + Sexo + Raça")
    print(f" n={N_ANIMAIS:,}  |  0–365 dias  |  seed={SEED}")
    print("="*65)

    print("\n[1/4] Gerando dados...")
    df = gerar_dados(N_ANIMAIS, SEED)
    validar_gerador(df)
    csv_path = f"{OUTPUT_DIR}/dados_sinteticos_v5.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Salvo: {csv_path}  | shape: {df.shape}")
    print(df[["Idade","PN","Peso"]].describe().round(2).to_string())

    print("\n[2/4] Ajustando modelo OLS...")
    Xm, y_log, y_orig, feat = build_matrix(df)
    modelo = OLSModel().fit(Xm, y_log, y_orig, feat)

    print(f"\n  R²       = {modelo.r2:.4f}")
    print(f"  R²_adj   = {modelo.r2_adj:.4f}")
    print(f"  AIC      = {modelo.aic:.1f}")
    print(f"  BIC      = {modelo.bic:.1f}")
    print(f"  σ_resid  = {modelo.sigma_res:.5f}  (escala log)")
    print(f"\n  Coeficientes:")
    print(modelo.coef_df().to_string(index=False))

    cd = dict(zip(modelo.coef_df()["Feature"], modelo.coef_df()["β"]))
    print("\n  Checagem de sinais:")
    for f, exp in [("Idade",">0"),("Idade2","<0"),("PN",">0"),("Sexo",">0")]:
        v = cd.get(f,0)
        ok = (v>0 if exp==">0" else v<0)
        print(f"    β({f:8s}) {exp}  {'✓' if ok else '⚠'}  ({v:+.6f})")

    print("\n[3/4] Demonstração da função de alerta:")
    casos = [
        (90,  29.0, 1, "Nelore",             68.0, "⚠ esperado: alerta"),
        (90,  29.0, 1, "Nelore",             92.0, "✓ esperado: OK"),
        (180, 29.0, 0, "Nelore",            118.0, "⚠ esperado: alerta"),
        (180, 29.0, 0, "Nelore",            152.0, "✓ esperado: OK"),
        (365, 29.0, 1, "Nelore",            225.0, "⚠ esperado: alerta"),
        (365, 29.0, 1, "Nelore",            295.0, "✓ esperado: OK"),
        (90,  40.0, 0, "Geral",              80.0, "raça não informada"),
    ]
    for idade, pn, sexo, raca, peso_real, tag in casos:
        res = verificar_alerta(modelo, feat, idade, pn, sexo, raca, peso_real)
        print(f"\n  [{tag}] t={idade}d PN={pn}kg {'M' if sexo else 'F'} {raca}")
        print(f"  {res['mensagem']}")

    print("\n[4/4] Gerando gráficos...")
    plot_comparativo_modelos(OUTPUT_DIR)
    plot_curvas_alerta(modelo, feat, OUTPUT_DIR)
    plot_diagnosticos(modelo, OUTPUT_DIR)

    print("\n"+"="*65)
    print(" CONCLUÍDO  v5.0")
    print("="*65)
    return df, modelo, feat


if __name__ == "__main__":
    df, modelo, feat = main()
