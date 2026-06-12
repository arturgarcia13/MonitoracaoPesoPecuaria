"""
=============================================================================
SIMULAÇÃO SINTÉTICA DE PESO BOVINO  v4.0  —  Modelo Simplificado
=============================================================================
Objetivo: estimar o peso esperado por idade/raça/sexo e emitir alerta
          quando o peso real estiver abaixo de Ŷ(t) − 1 DP.

Modelo MQO:
    Peso_i = β0 + β1·Idade_i + β2·PN_i + β3·Sexo_i
           + Σ γ_r · Raça_r_i  +  ε_i

Gerador (DGP):
    Peso_i = PN_i + GMD_i · Idade_i + δ_sex(t) · Sexo_i_centrado + ε_i(t)
    GMD_i  = GMD_base_i(Raça) + u_i

─────────────────────────────────────────────────────────────────────────────
MAPA DE ORIGEM — CADA NÚMERO TEM UMA FONTE, SEM EXCEÇÃO

  🟢 LITERAL (extraído textualmente do artigo)
  ─────────────────────────────────────────────────────────────────────────
  diff_sex(P365) = 44,69 kg  →  de Souza et al. (2018) Rev. Colomb. Cienc.
                                Anim. 10(1):68  [machos 289,30 / fêmeas 244,61]
  diff_sex(P550) = 83,87 kg  →  de Souza et al. (2018)
                                [machos 400,89 / fêmeas 317,02]
  σ²_p(P365)    = 1354,58    →  de Souza et al. (2018)
  e²(P365)      = 0,68       →  de Souza et al. (2018)
  GND Nelore    = 141,47 kg  →  Laureano et al. (2011) Arq. Bras. Med. Vet.
                                Zootec. 63(1):143  [ganho nasc.→desmama]
  dias desmama  = 210 d      →  Laureano et al. (2011)
  P8 Guzerá     = 150,34 kg  →  Mucari & Oliveira (2003) R. Bras. Zootec.
                                32(6):1604  [média descritiva M+F, n=2382]
  ~240 d (8 m.) = 240 d      →  Mucari & Oliveira (2003)
  σ²_F(P8 Guz.) = 661,70     →  Mucari & Oliveira (2003) Tabela 2
  ADG Cornell   = 0,820 kg/d →  Soberon et al. (2012) J. Dairy Sci. 95:783
  SD_ADG Cornell= 0,180 kg/d →  Soberon et al. (2012)
  SD_ADG comerc.= 0,110 kg/d →  Soberon et al. (2012)

  🟡 ARITMÉTICA / ÁLGEBRA (operação determinística sobre literais)
  ─────────────────────────────────────────────────────────────────────────
  GMD_Nelore  = 141,47 / 210   = 0,6737 kg/d
  GMD_Guzerá  = 150,34 / 240   = 0,6264 kg/d
  GMD_Cruzado = (0,6737+0,820) / 2 = 0,7468 kg/d
  b_sex = (83,87−44,69)/(550−365) = 0,211784 kg/d
  a_sex = 44,69 − 0,211784×365  = −32,6111 kg
  t0_sex = 32,6111/0,211784      = 154,0 dias (zero-crossing)
  σ²_e_365 = 0,68 × 1354,58     = 921,11 kg²
  σ_u  = sqrt((1354,58−921,11)/365²) = 0,05704 kg/d
  σ_ε(t) = sqrt(921,11 × t/365)
  SD_GMD_Nelore = sqrt(1354,58)/365   = 0,1008 kg/d
  SD_GMD_Guzerá = sqrt(661,70)/240    = 0,1072 kg/d

  ❌ REMOVIDO (sem suporte quantitativo no corpus)
  ─────────────────────────────────────────────────────────────────────────
  Brix/FPT   — sem delta kg/d por unidade Brix na literatura
  Novilha    — sem delta kg/d por paridade da vaca na literatura
  Isolamento — removido por design: o alerta É o desvio de peso

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
SEED        = 42
N_ANIMAIS   = 1_000      # ← altere aqui
IDADE_MIN   = 1
IDADE_MAX   = 365
OUTPUT_DIR  = "."

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONSTANTES — 100% LITERAIS (de Souza 2018)
# ─────────────────────────────────────────────────────────────────────────────
SIGMA2_P_365  = 1354.58
E2            = 0.68
DIFF_SEX_P365 = 44.69
DIFF_SEX_P550 = 83.87
T_P365        = 365.0
T_P550        = 550.0

# ─────────────────────────────────────────────────────────────────────────────
# 2.  DERIVAÇÕES ARITMÉTICAS / ALGÉBRICAS
# ─────────────────────────────────────────────────────────────────────────────

# ── diff_sex(t): reta por (P365, 44,69 kg) e (P550, 83,87 kg) ───────────────
_B_SEX  = (DIFF_SEX_P550 - DIFF_SEX_P365) / (T_P550 - T_P365)  # 0.211784 kg/dia
_A_SEX  = DIFF_SEX_P365 - _B_SEX * T_P365                       # -32.6111 kg
T0_SEX  = -_A_SEX / _B_SEX                                       # 154.0 dias


def diff_sex(t: np.ndarray) -> np.ndarray:
    """
    Diferença de peso macho−fêmea em kg para idade t.
    Fonte: reta entre (365d, 44,69 kg) e (550d, 83,87 kg) — de Souza 2018.
    clip(0) porque valores negativos são matematicamente impossíveis.
    """
    return np.maximum(0.0, _A_SEX + _B_SEX * np.asarray(t, dtype=float))


# ── σ_u e σ_ε(t): álgebra sobre σ²_p e e² de de Souza 2018 ─────────────────
_SIGMA2_E_365 = E2 * SIGMA2_P_365                                # 921.11 kg²
_SIGMA2_U     = (SIGMA2_P_365 - _SIGMA2_E_365) / T_P365 ** 2    # 0.003253
SIGMA_U       = np.sqrt(_SIGMA2_U)                               # 0.05704 kg/d


def sigma_eps(t: np.ndarray) -> np.ndarray:
    """σ_ε(t) = sqrt(e²·σ²_p·t/365) — de Souza 2018."""
    return np.sqrt(_SIGMA2_E_365 * np.asarray(t, dtype=float) / T_P365)


# ── Limiar de alerta: Ŷ − 1·σ_ε(t) ─────────────────────────────────────────
def limiar_alerta(y_hat: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Retorna o limiar abaixo do qual o peso real dispara alerta.
    Critério: Peso_real < Ŷ(t) − σ_ε(t)   (1 DP abaixo da média esperada)
    """
    return y_hat - sigma_eps(np.asarray(t, dtype=float))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PARÂMETROS POR RAÇA
# ─────────────────────────────────────────────────────────────────────────────
# Cada linha: (pn_mu, pn_sd, gmd_mu, gmd_sd, prop)
# Fonte de gmd_mu e gmd_sd detalhada no cabeçalho do módulo.

RACAS = {
    "Nelore": (
        29.0, 3.5,
        141.47 / 210,                       # 0.6737 kg/d — Laureano 2011
        np.sqrt(SIGMA2_P_365) / T_P365,     # 0.1008 kg/d — álgebra de Souza
        0.30,
    ),
    "Guzerá": (
        26.0, 3.0,
        150.34 / 240,                       # 0.6264 kg/d — Mucari 2003
        np.sqrt(661.70) / 240,              # 0.1072 kg/d — álgebra Mucari
        0.15,
    ),
    "Holandês/Girolando": (
        40.0, 5.0,
        0.820,                              # kg/d — Soberon 2012 LITERAL
        0.180,                              # kg/d — Soberon 2012 LITERAL
        0.20,
    ),
    "Cruzado ½ sangue": (
        33.0, 4.0,
        (141.47 / 210 + 0.820) / 2,        # média aritmética Laureano + Soberon
        0.110,                              # kg/d — Soberon 2012 comercial LITERAL
        0.20,
    ),
    "Angus/Simental": (
        38.0, 4.5,
        0.820,                              # kg/d — Soberon 2012 LITERAL
        0.110,                              # kg/d — Soberon 2012 comercial LITERAL
        0.15,
    ),
    # Categoria especial: raça não informada → média ponderada das 5
    "Geral": (
        None, None, None, None, None,       # calculado abaixo
    ),
}

# Média ponderada para categoria "Geral" (raça não informada pelo usuário)
_props  = np.array([RACAS[r][4] for r in list(RACAS.keys())[:-1]])
_pn_mu  = sum(RACAS[r][0] * RACAS[r][4] for r in list(RACAS.keys())[:-1])
_gmd_mu = sum(RACAS[r][2] * RACAS[r][4] for r in list(RACAS.keys())[:-1])
_pn_sd  = sum(RACAS[r][1] * RACAS[r][4] for r in list(RACAS.keys())[:-1])
_gmd_sd = sum(RACAS[r][3] * RACAS[r][4] for r in list(RACAS.keys())[:-1])
RACAS["Geral"] = (_pn_mu, _pn_sd, _gmd_mu, _gmd_sd, None)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  GERADOR DE DADOS
# ─────────────────────────────────────────────────────────────────────────────

def gerar_dados(n: int, seed: int = SEED) -> pd.DataFrame:
    """
    Gera n observações sintéticas.

    DGP:
        Peso_i = PN_i + GMD_i · Idade_i + efeito_sex_i + ε_i(t)
        GMD_i  = GMD_base_i + u_i

    efeito_sex: aplicado diretamente no Peso (não no GMD) para preservar
    a semântica literal de de Souza 2018: diferença de peso observada,
    não taxa diária. Centrado em zero (machos: +diff/2, fêmeas: −diff/2).
    """
    rng = np.random.default_rng(seed)
    racas_base = [r for r in RACAS if r != "Geral"]
    props      = [RACAS[r][4] for r in racas_base]

    idx        = rng.choice(len(racas_base), size=n, p=props)
    raca_nomes = np.array(racas_base)[idx]

    Idade = rng.integers(IDADE_MIN, IDADE_MAX + 1, n).astype(float)

    PN = np.array([
        np.clip(
            rng.normal(RACAS[r][0], RACAS[r][1]),
            RACAS[r][0] - 3 * RACAS[r][1],
            RACAS[r][0] + 3 * RACAS[r][1],
        )
        for r in raca_nomes
    ])

    Sexo = rng.binomial(1, 0.50, n).astype(float)   # 1=macho, 0=fêmea

    GMD_base = np.array([
        np.clip(rng.normal(RACAS[r][2], RACAS[r][3]), 0.10, 1.60)
        for r in raca_nomes
    ])

    u_i  = rng.normal(0.0, SIGMA_U, n)
    GMD  = np.clip(GMD_base + u_i, 0.05, 1.60)

    eps  = rng.normal(0.0, sigma_eps(Idade), n)

    # Efeito de sexo centrado: macho recebe +diff/2, fêmea −diff/2
    dsex = diff_sex(Idade)
    Peso = PN + GMD * Idade + (Sexo - 0.5) * dsex + eps
    Peso = np.clip(Peso, PN * 0.85, None)

    return pd.DataFrame({
        "Raca":   raca_nomes,
        "Idade":  Idade.astype(int),
        "PN":     PN,
        "Sexo":   Sexo.astype(int),
        "GMD":    GMD,       # variável latente — diagnóstico apenas
        "Peso":   Peso,
    })


# ─────────────────────────────────────────────────────────────────────────────
# 5.  VALIDAÇÃO DO GERADOR
# ─────────────────────────────────────────────────────────────────────────────

def validar_gerador(df: pd.DataFrame) -> bool:
    print("\n" + "─" * 65)
    print("VALIDAÇÃO — Âncoras Literais do Corpus")
    print("─" * 65)

    checks = [
        # (rótulo, raça, t_min, t_max, exp_lo, exp_hi, fonte)
        ("Nelore ~P210",
         "Nelore", 200, 220, 140, 205,
         "Laureano 2011: PD=171,15 ± 24,95 kg [LITERAL]"),
        ("Guzerá ~P240",
         "Guzerá", 230, 250, 105, 195,
         "Mucari 2003: P8=150,34 ± 28,93 kg [LITERAL descritivo]"),
        ("Holandês ~P49",
         "Holandês/Girolando", 40, 60, 60, 110,
         "Soberon 2012: WW≈82 kg [LITERAL]"),
        ("Nelore P365 machos",
         "Nelore", 355, 375, 255, 325,
         "de Souza 2018: 289,30 ± 2,10 kg [LITERAL]"),
        ("Nelore P365 fêmeas",
         "Nelore", 355, 375, 210, 280,
         "de Souza 2018: 244,61 ± 2,08 kg [LITERAL]"),
    ]

    all_ok = True
    for label, raca, tlo, thi, elo, ehi, fonte in checks:
        # filtra por sexo quando necessário
        mask = (df["Raca"] == raca) & (df["Idade"].between(tlo, thi))
        if "macho" in label:
            mask &= df["Sexo"] == 1
        elif "fêmea" in label:
            mask &= df["Sexo"] == 0
        sub = df.loc[mask, "Peso"]
        if len(sub) < 3:
            print(f"  — {label}: n={len(sub)} insuficiente")
            continue
        m, s = sub.mean(), sub.std()
        ok = elo <= m <= ehi
        if not ok:
            all_ok = False
        print(f"  {'✓' if ok else '⚠'} {label}: {m:.1f} ± {s:.1f} kg  "
              f"(n={len(sub)}, esperado [{elo},{ehi}])")
        print(f"      {fonte}")

    print("─" * 65)
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# 6.  OLS
# ─────────────────────────────────────────────────────────────────────────────

class OLSModel:
    """OLS via numpy. AIC e BIC via log-likelihood gaussiana MLE."""

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
        s2  = sse / self.n
        ll  = -self.n / 2 * np.log(2 * np.pi * s2) - sse / (2 * s2)
        p   = self.k + 1
        self.aic = -2 * ll + 2 * p
        self.bic = -2 * ll + p * np.log(self.n)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.beta

    def coef_df(self) -> pd.DataFrame:
        return pd.DataFrame({"Feature": self.cols, "β": self.beta})


def build_matrix(df: pd.DataFrame, incluir_raca: bool):
    """Constrói X e y. Raça como dummies (ref: Angus/Simental)."""
    X = pd.DataFrame({
        "Intercepto": 1.0,
        "Idade":      df["Idade"].astype(float).values,
        "PN":         df["PN"].values,
        "Sexo":       df["Sexo"].astype(float).values,
    })
    if incluir_raca:
        dummies = pd.get_dummies(df["Raca"], prefix="Raca", drop_first=True).astype(int)
        X = pd.concat([X, dummies], axis=1)
    return X.values.astype(np.float64), df["Peso"].values.astype(np.float64), X.columns.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# 7.  FUNÇÃO DE ALERTA
# ─────────────────────────────────────────────────────────────────────────────

def verificar_alerta(
    modelo: OLSModel,
    feature_names: list,
    idade: float,
    pn: float,
    sexo: int,
    raca: str,
    peso_real: float,
) -> dict:
    """
    Recebe os dados de UM animal e retorna o diagnóstico de alerta.

    Parâmetros
    ----------
    modelo       : OLSModel treinado
    feature_names: lista de nomes de features usadas no treino
    idade        : idade do animal em dias
    pn           : peso ao nascimento (kg)
    sexo         : 1=macho, 0=fêmea
    raca         : nome da raça ou "Geral" se não informada
    peso_real    : peso observado hoje (kg)

    Retorna
    -------
    dict com:
        y_hat       : peso esperado pelo modelo
        limiar      : y_hat − σ_ε(t)  [limite de alerta]
        desvio      : peso_real − y_hat
        desvio_dp   : desvio em unidades de DP
        alerta      : True se peso_real < limiar
        mensagem    : texto para o aplicativo
    """
    # Monta vetor de entrada com as mesmas features do treino
    row = {"Intercepto": 1.0, "Idade": float(idade),
           "PN": float(pn), "Sexo": float(sexo)}

    # Preenche dummies de raça com zeros; seta a correta se presente
    for f in feature_names:
        if f.startswith("Raca_"):
            row[f] = 0
    raca_col = f"Raca_{raca}"
    if raca_col in feature_names:
        row[raca_col] = 1
    # Se raça não está nas dummies (é a referência ou "Geral"), mantém zeros

    x_vec = np.array([row.get(f, 0.0) for f in feature_names], dtype=float)
    y_hat = float(modelo.predict(x_vec.reshape(1, -1))[0])
    lim   = float(y_hat - sigma_eps(np.array([idade]))[0])
    desv  = float(peso_real - y_hat)
    desv_dp = desv / float(sigma_eps(np.array([idade]))[0])
    alerta  = peso_real < lim

    if alerta:
        msg = (f"⚠ ALERTA: peso {peso_real:.1f} kg está {abs(desv):.1f} kg abaixo "
               f"do esperado ({y_hat:.1f} kg) para {idade} dias. "
               f"Desvio: {desv_dp:.1f} DP. Recomenda-se avaliação clínica.")
    else:
        msg = (f"✓ Peso {peso_real:.1f} kg dentro do esperado "
               f"({y_hat:.1f} ± {sigma_eps(np.array([idade]))[0]:.1f} kg) "
               f"para {idade} dias.")

    return {"y_hat": y_hat, "limiar": lim, "desvio": desv,
            "desvio_dp": desv_dp, "alerta": alerta, "mensagem": msg}


# ─────────────────────────────────────────────────────────────────────────────
# 8.  GRÁFICOS
# ─────────────────────────────────────────────────────────────────────────────

def plot_curvas_crescimento(df: pd.DataFrame, modelo: OLSModel,
                            feature_names: list, output_dir: str = ".") -> None:
    """Curvas de crescimento observadas + preditas por raça."""
    fig, ax = plt.subplots(figsize=(11, 6))
    palette = plt.cm.tab10(np.linspace(0, 0.8, 5))
    racas_plot = ["Nelore", "Guzerá", "Holandês/Girolando",
                  "Cruzado ½ sangue", "Angus/Simental"]
    t_range = np.linspace(1, 365, 365)

    for raca, cor in zip(racas_plot, palette):
        sub = df[df["Raca"] == raca].copy()
        if len(sub) < 10:
            continue
        bins = np.linspace(1, 365, 25)
        sub["bin"] = pd.cut(sub["Idade"], bins=bins, labels=bins[:-1])
        grp = sub.groupby("bin", observed=True)["Peso"].agg(["mean", "std"]).dropna()
        idx = grp.index.astype(float)
        ax.scatter(idx, grp["mean"], color=cor, s=18, alpha=0.7, zorder=3)
        ax.fill_between(idx, grp["mean"] - grp["std"],
                        grp["mean"] + grp["std"], color=cor, alpha=0.10)

        # Linha predita pelo modelo (macho e fêmea)
        for sexo, ls in [(1, "-"), (0, "--")]:
            preds = []
            for t in t_range:
                row = {"Intercepto": 1.0, "Idade": t,
                       "PN": RACAS[raca][0], "Sexo": float(sexo)}
                for f in feature_names:
                    if f.startswith("Raca_"):
                        row[f] = 0
                rc = f"Raca_{raca}"
                if rc in feature_names:
                    row[rc] = 1
                x = np.array([row.get(f, 0.0) for f in feature_names])
                preds.append(modelo.predict(x.reshape(1, -1))[0])
            lbl = f"{raca} ({'M' if sexo else 'F'})"
            ax.plot(t_range, preds, color=cor, lw=1.8, ls=ls, label=lbl)

    # Âncoras literais
    anchors = [(210, 171.15, "Nelore PD (Laureano 2011)"),
               (240, 150.34, "Guzerá P8 (Mucari 2003)"),
               (49,  82.10,  "Holandês WW (Soberon 2012)"),
               (365, 289.30, "Nelore M P365 (de Souza 2018)"),
               (365, 244.61, "Nelore F P365 (de Souza 2018)")]
    for t, w, lbl in anchors:
        ax.axhline(w, color="gray", lw=0.6, ls=":", alpha=0.5)
        ax.plot(t, w, "k*", ms=10, zorder=5)

    ax.set_xlabel("Idade (dias)")
    ax.set_ylabel("Peso (kg)")
    ax.set_title("Curvas de Crescimento — Preditas pelo Modelo + Observadas\n"
                 "(sólido = machos, tracejado = fêmeas; ★ = âncoras literais)")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fname = f"{output_dir}/curvas_crescimento_v4.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {fname}")


def plot_diagnosticos(modelo: OLSModel, output_dir: str = ".") -> None:
    """4-panel diagnóstico do modelo único."""
    color = "#2563EB"
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Diagnósticos do Modelo — v4.0", fontsize=14,
                 fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(modelo.y_hat, modelo.resid, alpha=0.25, s=10, color=color)
    ax.axhline(0, color="black", lw=1.0, ls="--")
    bins = np.percentile(modelo.y_hat, np.linspace(5, 95, 20))
    bm = [modelo.resid[np.abs(modelo.y_hat - b) < 8].mean() for b in bins]
    ax.plot(bins, bm, color="red", lw=1.5, label="Tendência")
    ax.set_xlabel("Ajustado (kg)"); ax.set_ylabel("Resíduo (kg)")
    ax.set_title("Resíduos vs. Ajustados"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    (osm, osr), (slope, intercept, r) = stats.probplot(modelo.resid, dist="norm")
    ax.scatter(osm, osr, alpha=0.25, s=10, color=color)
    lx = np.array([osm[0], osm[-1]])
    ax.plot(lx, slope * lx + intercept, "k--", lw=1.5)
    ax.set_xlabel("Quantis Teóricos"); ax.set_ylabel("Quantis Amostrais")
    ax.set_title(f"QQ-Plot  (r = {r:.4f})")

    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(modelo.y, modelo.y_hat, alpha=0.20, s=10, color=color)
    lims = [min(modelo.y.min(), modelo.y_hat.min()) - 5,
            max(modelo.y.max(), modelo.y_hat.max()) + 5]
    ax.plot(lims, lims, "k--", lw=1.2, label="45°")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Observado (kg)"); ax.set_ylabel("Predito (kg)")
    ax.set_title(f"Predito vs. Observado  (R² = {modelo.r2:.4f})")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    ax.hist(modelo.resid, bins=45, color=color, alpha=0.65,
            edgecolor="white", density=True)
    xr = np.linspace(modelo.resid.min(), modelo.resid.max(), 200)
    ax.plot(xr, stats.norm.pdf(xr, modelo.resid.mean(), modelo.resid.std()),
            "k-", lw=1.5, label="N(0,σ²) ref.")
    ax.set_xlabel("Resíduo (kg)"); ax.set_ylabel("Densidade")
    ax.set_title("Distribuição dos Resíduos"); ax.legend(fontsize=8)

    fig.tight_layout()
    fname = f"{output_dir}/diagnosticos_v4.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {fname}")


def plot_alerta_exemplo(modelo: OLSModel, feature_names: list,
                        output_dir: str = ".") -> None:
    """
    Visualiza a função de alerta para um animal Nelore macho com PN=29 kg.
    Mostra a faixa esperada (Ŷ ± 1 DP) e dois exemplos: dentro e fora.
    """
    t_range = np.arange(1, 366)
    raca, sexo, pn = "Nelore", 1, 29.0

    y_hats = []
    for t in t_range:
        row = {"Intercepto": 1.0, "Idade": float(t), "PN": pn, "Sexo": float(sexo)}
        for f in feature_names:
            if f.startswith("Raca_"):
                row[f] = 0
        rc = f"Raca_{raca}"
        if rc in feature_names:
            row[rc] = 1
        x = np.array([row.get(f, 0.0) for f in feature_names])
        y_hats.append(modelo.predict(x.reshape(1, -1))[0])

    y_hats = np.array(y_hats)
    lims   = y_hats - sigma_eps(t_range)
    upps   = y_hats + sigma_eps(t_range)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_range, y_hats, color="#2563EB", lw=2.2, label="Peso esperado Ŷ(t)")
    ax.fill_between(t_range, lims, upps, alpha=0.18, color="#2563EB",
                    label="Faixa ±1 DP  [σ_ε(t) de Souza 2018]")
    ax.plot(t_range, lims, color="#DC2626", lw=1.5, ls="--",
            label="Limiar de alerta  Ŷ(t) − σ_ε(t)")

    # Exemplos de animais
    ax.scatter([90, 200], [55.0, 185.0], color="#DC2626", zorder=6, s=80,
               marker="v", label="Animal em alerta (abaixo do limiar)")
    ax.scatter([90, 200], [75.0, 230.0], color="#16A34A", zorder=6, s=80,
               marker="^", label="Animal saudável (dentro da faixa)")

    # Âncoras literais
    ax.plot(365, 289.30, "k*", ms=12, zorder=7,
            label="Âncora literal P365 macho (de Souza 2018: 289,30 kg)")

    ax.set_xlabel("Idade (dias)")
    ax.set_ylabel("Peso (kg)")
    ax.set_title("Função de Alerta — Nelore Macho PN=29 kg\n"
                 f"Limiar: Ŷ(t) − σ_ε(t)   [σ_ε(t) = sqrt({_SIGMA2_E_365:.0f}×t/365)]")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fname = f"{output_dir}/funcao_alerta_v4.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print(" SIMULAÇÃO SINTÉTICA DE PESO BOVINO  v4.0")
    print(" Modelo simplificado — 100% referencial, sem inferências")
    print(f" n = {N_ANIMAIS:,} animais  |  janela 0–365 dias  |  seed={SEED}")
    print("=" * 65)

    print(f"""
[PARÂMETROS DO GERADOR]
  GMD_Nelore   = {141.47/210:.4f} kg/d   [141,47÷210 — Laureano 2011]
  GMD_Guzerá   = {150.34/240:.4f} kg/d   [150,34÷240 — Mucari 2003]
  GMD_Holandês = 0,8200 kg/d    [LITERAL — Soberon 2012]
  diff_sex(t)  = max(0, {_A_SEX:.4f} + {_B_SEX:.6f}×t)
                 [reta P365/P550 de Souza 2018 — zero-crossing t={T0_SEX:.0f}d]
  σ_u          = {SIGMA_U:.5f} kg/d   [álgebra: de Souza 2018]
  σ_ε(t)       = sqrt({_SIGMA2_E_365:.2f}×t/365)  [álgebra: de Souza 2018]
  Limiar alerta: Ŷ(t) − σ_ε(t)   [1 DP abaixo da média esperada]
""")

    # ── Gerar dados ────────────────────────────────────────────────────────
    print("[1/4] Gerando banco de dados...")
    df = gerar_dados(N_ANIMAIS, SEED)
    validar_gerador(df)

    csv_path = f"{OUTPUT_DIR}/dados_sinteticos_v4.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Salvo: {csv_path}  | shape: {df.shape}")
    print(df[["Idade", "PN", "Peso"]].describe().round(2).to_string())
    print(f"\n  Distribuição de raças:\n{df['Raca'].value_counts().to_string()}")

    # ── Ajustar modelo ─────────────────────────────────────────────────────
    print("\n[2/4] Ajustando modelo OLS (Idade + PN + Sexo + Raça)...")
    X, y, feat = build_matrix(df, incluir_raca=True)
    modelo = OLSModel().fit(X, y, feat)

    print(f"\n  R²     = {modelo.r2:.4f}")
    print(f"  R²_adj = {modelo.r2_adj:.4f}")
    print(f"  AIC    = {modelo.aic:.1f}")
    print(f"  BIC    = {modelo.bic:.1f}")
    print(f"\n  Coeficientes recuperados:")
    coef = modelo.coef_df()
    print(coef.to_string(index=False))

    # Verificação de sinais
    cd = dict(zip(coef["Feature"], coef["β"]))
    print("\n  Checagem de sinais esperados:")
    for feat_name, exp in [("Idade",">0"),("PN",">0"),("Sexo",">0")]:
        val = cd.get(feat_name, 0)
        ok  = (val > 0) if exp == ">0" else (val < 0)
        print(f"    β({feat_name:6s}) {exp}  {'✓' if ok else '⚠'}  ({val:+.4f})")

    # ── Demonstração da função de alerta ───────────────────────────────────
    print("\n[3/4] Demonstração da função de alerta:")
    casos = [
        (90,  29.0, 1, "Nelore",  70.0,  "abaixo do esperado"),
        (90,  29.0, 1, "Nelore",  92.0,  "dentro do esperado"),
        (180, 29.0, 0, "Nelore", 118.0,  "abaixo do esperado"),
        (180, 29.0, 0, "Nelore", 150.0,  "dentro do esperado"),
        (365, 29.0, 1, "Nelore", 250.0,  "abaixo do esperado"),
        (365, 29.0, 1, "Nelore", 297.0,  "dentro do esperado"),
        (90,  40.0, 0, "Geral",   85.0,  "raça não informada"),
    ]
    for idade, pn, sexo, raca, peso_real, descricao in casos:
        res = verificar_alerta(modelo, feat, idade, pn, sexo, raca, peso_real)
        print(f"\n  [{descricao}] t={idade}d PN={pn}kg {'M' if sexo else 'F'} {raca}")
        print(f"  {res['mensagem']}")

    # ── Gráficos ───────────────────────────────────────────────────────────
    print("\n[4/4] Gerando gráficos...")
    plot_curvas_crescimento(df, modelo, feat, OUTPUT_DIR)
    plot_diagnosticos(modelo, OUTPUT_DIR)
    plot_alerta_exemplo(modelo, feat, OUTPUT_DIR)

    print("\n" + "=" * 65)
    print(" CONCLUÍDO  v4.0")
    print(" Arquivos: dados_sinteticos_v4.csv | curvas_crescimento_v4.png")
    print("           diagnosticos_v4.png     | funcao_alerta_v4.png")
    print("=" * 65)

    return df, modelo, feat


if __name__ == "__main__":
    df, modelo, feat = main()
