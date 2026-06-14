"""
Simulação Monte Carlo — Validação do Sistema de Alertas em Zootecnia de Precisão
Espécie: Ovinos (Morada Nova / Santa Inês / Dorper)

Objetivo:
    Estimar a sensibilidade, especificidade, tempo de detecção e AUC do sistema
    de alertas R(t) ao longo de 1 000 simulações independentes de 10 000 animais,
    sob distribuições realistas de morbidade e peso ao nascer.

Fundamentação teórica:
    Freitas et al. (1980)  — peso médio Morada Nova: 3.1 kg
    McMillan et al. (1983) — faixa ótima de sobrevivência: 3.3–4.1 kg
    Hatcher (2009)         — P_opt por tipo de parto
    Gardner et al.         — efeitos de parto, sexo e paridade
    Sarmento et al. (2010) — variância heterogênea crescente σ(t)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ENUMERAÇÕES
# ═══════════════════════════════════════════════════════════════════════════════

class TipoParto(IntEnum):
    SIMPLES  = 0
    GEMEO    = 1
    TRIGEMEO = 2


class GrupoMorbidade(IntEnum):
    SAUDAVEL   = 0   # sem comprometimento sanitário
    SUBNUTRIDO = 1   # déficit nutricional crônico (leite insuficiente, competição)
    DOENTE     = 2   # evento infeccioso agudo a partir de DIA_INICIO_DOENCA
    CRITICO    = 3   # falência múltipla — trajetória de crescimento severamente comprometida


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PARÂMETROS DA SIMULAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SimParams:
    # ── Controle da simulação ─────────────────────────────────────────────────
    n_simulacoes: int   = 1_000
    n_animais:    int   = 10_000
    seed:         int   = 42

    # ── Sistema de alerta ─────────────────────────────────────────────────────
    limiar_alerta_pct: float = 30.0   # R(t) em % acima do qual o alerta é emitido
    k_decaimento:      float = 0.2    # sensibilidade do R(t) ao Z negativo

    # ── Equação 1: Peso ao Nascer ─────────────────────────────────────────────
    # β0 = 4.10 kg: Macho, Parto Simples, Mãe Multípara (Hatcher 2009)
    beta_0:           float = 4.10
    beta_gemeo:       float = -0.65   # Hatcher (2009): 4.00 vs 3.35 kg
    beta_trigemeo:    float = -1.40   # Gardner et al.: decréscimo progressivo
    beta_femea:       float = -0.30   # média consensual Everts / Gardner / Medeiros
    beta_primipara:   float = -0.35   # Gardner et al.: incremento 1ª→2ª gestação
    sigma_nascimento: float = 0.66    # kg — desvio padrão do erro aleatório 

    # ── Equação 2: GMD e Trajetória ───────────────────────────────────────────
    gmd_base:                  float = 0.252   # kg/dia — ovinos corte
    gamma:                     float = 0.02    # sensibilidade GMD ao P0
    penalidade_gmd_gemeo:      float = -0.025  # kg/dia
    penalidade_gmd_trigemeo:   float = -0.030  # kg/dia
    sigma_gmd:                 float = 0.015   # kg/dia — variabilidade genética individual
                                                # (empírico; calibrar com dados locais)
    sigma_balanca:             float = 0.10    # kg — erro de medição da balança

    # ── Equação 3: Risco Logístico Basal ─────────────────────────────────────
    # z = alpha_0 + alpha_1 * (P0 - P_opt)²  /  P(óbito) = sigmoid(z)
    alpha_0_macho:   float = -2.5    # intercepto machos (regressao.md Cenário B)
    alpha_0_femea:   float = -3.0    # fêmeas têm vantagem de sobrevivência
    alpha_1:         float =  1.2    # coeficiente quadrático (regressao.md Cenário B)
    p_opt_simples:   float =  3.96   # kg — Hatcher (2009)
    p_opt_gemeo:     float =  3.63   # kg — Hatcher (2009)
    p_opt_trigemeo:  float =  3.44   # kg — extrapolação progressiva Gardner et al.

    # R(t) = p_morte_nasc * 100 * exp(k * max(0, -Z))
    # p_morte_nasc é calculada uma vez pela curva U (peso/sexo/parto).
    # A morbidade afeta R(t) indiretamente: GMD reduzido → peso abaixo do alvo
    # → Z negativo → fator exponencial amplifica o risco basal do animal.

    # ── Multiplicadores de GMD por grupo ─────────────────────────────────────
    mult_gmd_saudavel:   float = 1.00
    mult_gmd_subnutrido: float = 0.70
    mult_gmd_doente:     float = 0.50
    mult_gmd_critico:    float = 0.20

    # O grupo DOENTE tem evento com início definido; antes disso, cresce normalmente
    dia_inicio_doenca: int = 20   # dia em que o evento infeccioso se manifesta

    # ── Heterogeneidade Residual — Sarmento et al. (2010) ────────────────────
    # σ(t) = sigma_nascimento + lambda_sigma * (t / T_MAX)²
    lambda_sigma: float = 1.5    # fator de escala biométrica
    t_max:        int   = 90     # dias até desmama — horizonte biológico de referência

    # ── Proporções populacionais ──────────────────────────────────────────────
    prop_simples:   float = 0.65
    prop_gemeo:     float = 0.30
    prop_trigemeo:  float = 0.05
    prop_femeas:    float = 0.50
    prop_primipara: float = 0.20

    # ── Proporções de morbidade ───────────────────────────────────────────────
    prop_saudavel:   float = 0.70
    prop_subnutrido: float = 0.15
    prop_doente:     float = 0.10
    prop_critico:    float = 0.05

    # ── Dias de pesagem ───────────────────────────────────────────────────────
    dias_pesagem: tuple[int, ...] = (15, 30, 45, 60)

    # ── Código interno para fêmea ─────────────────────────────────────────────
    codigo_femea: int = 1


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ETAPA 1–3: GERAÇÃO DO REBANHO
# ═══════════════════════════════════════════════════════════════════════════════

def gerar_rebanho(
    n: int,
    rng: np.random.Generator,
    p: SimParams,
) -> dict[str, np.ndarray]:
    """Gera os atributos biológicos fixos de cada animal (verdade oculta).

    Retorna:
        tipo_parto, sexo, primipara, p0, p_opt_bio, alpha0,
        z_nasc, p_morte_nasc, grupos
    """
    # Atributos categóricos
    tipo_parto = rng.choice(
        [TipoParto.SIMPLES, TipoParto.GEMEO, TipoParto.TRIGEMEO],
        size=n,
        p=[p.prop_simples, p.prop_gemeo, p.prop_trigemeo],
    )
    sexo      = rng.binomial(1, p.prop_femeas, size=n)   # 1 = fêmea (ver p.codigo_femea)
    primipara = rng.binomial(1, p.prop_primipara, size=n)

    # Equação 1: Peso ao Nascer — P0 = β0 + β_parto + β_sexo + β_matriz + η
    beta_parto = np.select(
        [tipo_parto == TipoParto.SIMPLES,
         tipo_parto == TipoParto.GEMEO,
         tipo_parto == TipoParto.TRIGEMEO],
        [0.0, p.beta_gemeo, p.beta_trigemeo],
    )
    beta_sexo   = np.where(sexo == p.codigo_femea, p.beta_femea, 0.0)
    beta_matriz = np.where(primipara == 1, p.beta_primipara, 0.0)
    eta = rng.normal(0.0, p.sigma_nascimento, size=n)

    p0 = np.clip(p.beta_0 + beta_parto + beta_sexo + beta_matriz + eta, 0.5, 8.0)

    # P_opt biológico por tipo de parto (Hatcher 2009 / Gardner et al.)
    p_opt_bio = np.select(
        [tipo_parto == TipoParto.SIMPLES,
         tipo_parto == TipoParto.GEMEO,
         tipo_parto == TipoParto.TRIGEMEO],
        [p.p_opt_simples, p.p_opt_gemeo, p.p_opt_trigemeo],
    )

    # Equação 3: z e P(óbito) basais no nascimento
    alpha0  = np.where(sexo == p.codigo_femea, p.alpha_0_femea, p.alpha_0_macho)
    z_nasc  = alpha0 + p.alpha_1 * (p0 - p_opt_bio) ** 2
    p_morte_nasc = expit(z_nasc)

    # Grupos de morbidade — independentes do peso ao nascer
    grupos = rng.choice(
        [GrupoMorbidade.SAUDAVEL, GrupoMorbidade.SUBNUTRIDO,
         GrupoMorbidade.DOENTE,   GrupoMorbidade.CRITICO],
        size=n,
        p=[p.prop_saudavel, p.prop_subnutrido, p.prop_doente, p.prop_critico],
    )

    return {
        "tipo_parto"   : tipo_parto,
        "sexo"         : sexo,
        "primipara"    : primipara,
        "p0"           : p0,
        "p_opt_bio"    : p_opt_bio,
        "alpha0"       : alpha0,
        "z_nasc"       : z_nasc,
        "p_morte_nasc" : p_morte_nasc,
        "grupos"       : grupos,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ETAPAS 4–6: TRAJETÓRIA DE PESO
# ═══════════════════════════════════════════════════════════════════════════════

def _mult_gmd_por_grupo(grupos: np.ndarray, p: SimParams) -> np.ndarray:
    """Retorna o multiplicador de GMD de cada animal segundo seu grupo base.

    Nota: o grupo DOENTE tem multiplicador normal até DIA_INICIO_DOENCA;
    a redução é aplicada no cálculo de cada pesagem em gerar_trajetoria().
    """
    return np.select(
        [grupos == GrupoMorbidade.SAUDAVEL,
         grupos == GrupoMorbidade.SUBNUTRIDO,
         grupos == GrupoMorbidade.DOENTE,
         grupos == GrupoMorbidade.CRITICO],
        [p.mult_gmd_saudavel,
         p.mult_gmd_subnutrido,
         p.mult_gmd_doente,     # placeholder; ajustado por dia no loop
         p.mult_gmd_critico],
    )


def gerar_trajetoria(
    rebanho: dict[str, np.ndarray],
    rng: np.random.Generator,
    p: SimParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula P_t_real e P_t_alvo para cada animal em cada dia de pesagem.

    O GMD base (trajetória alvo do sistema) usa p_opt_bio como referência,
    o que é biologicamente correto: cada tipo de parto tem seu próprio ponto
    de máxima eficiência de crescimento.

    O GMD efetivo (trajetória real) é penalizado por grupo de morbidade.
    O grupo DOENTE cresce normalmente até DIA_INICIO_DOENCA.

    Returns:
        P_t_real: (n_animais, n_pesagens) — pesos reais observados (c/ ruído de balança)
        P_t_alvo: (n_animais, n_pesagens) — pesos esperados pelo sistema
    """
    p0         = rebanho["p0"]
    p_opt_bio  = rebanho["p_opt_bio"]
    tipo_parto = rebanho["tipo_parto"]
    grupos     = rebanho["grupos"]
    n          = len(p0)
    n_pesagens = len(p.dias_pesagem)

    penalidade_gmd = np.select(
        [tipo_parto == TipoParto.SIMPLES,
         tipo_parto == TipoParto.GEMEO,
         tipo_parto == TipoParto.TRIGEMEO],
        [0.0,
         p.penalidade_gmd_gemeo,
         p.penalidade_gmd_trigemeo],
    )
    eps      = rng.normal(0.0, p.sigma_gmd, size=n)
    gmd_base = p.gmd_base + penalidade_gmd + p.gamma * (p0 - p_opt_bio) + eps

    P_t_real = np.empty((n, n_pesagens))
    P_t_alvo = np.empty((n, n_pesagens))

    mult_base = _mult_gmd_por_grupo(grupos, p)

    for j, t in enumerate(p.dias_pesagem):
        # Trajetória alvo: o que o sistema espera sem interferência sanitária
        P_t_alvo[:, j] = p0 + gmd_base * t

        # GMD efetivo: reseta a cada dia para aplicar multiplicadores independentes
        gmd_efetivo = gmd_base.copy()

        # SUBNUTRIDO e CRÍTICO: multiplicador constante desde o início
        mask_sub  = grupos == GrupoMorbidade.SUBNUTRIDO
        mask_crit = grupos == GrupoMorbidade.CRITICO
        gmd_efetivo[mask_sub]  *= p.mult_gmd_subnutrido
        gmd_efetivo[mask_crit] *= p.mult_gmd_critico

        # DOENTE: penalidade só se aplica após DIA_INICIO_DOENCA
        mask_doente = grupos == GrupoMorbidade.DOENTE
        if t <= p.dia_inicio_doenca:
            # Antes da doença: cresce normalmente
            peso_t = p0 + gmd_efetivo * t
        else:
            # Após o início da doença: fase saudável até dia_inicio + fase doente depois
            gmd_efetivo_doente         = gmd_base.copy()
            gmd_efetivo_doente[mask_sub]  *= p.mult_gmd_subnutrido
            gmd_efetivo_doente[mask_crit] *= p.mult_gmd_critico
            gmd_efetivo_doente[mask_doente] *= p.mult_gmd_doente

            peso_normal = p0 + gmd_efetivo * t
            peso_doente = (
                p0
                + gmd_base * p.dia_inicio_doenca
                + gmd_efetivo_doente * (t - p.dia_inicio_doenca)
            )
            peso_t = np.where(mask_doente, peso_doente, peso_normal)

        ruido_balanca   = rng.normal(0.0, p.sigma_balanca, size=n)
        P_t_real[:, j] = peso_t + ruido_balanca

    return P_t_real, P_t_alvo


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ETAPAS 7–9: RISCO DINÂMICO E ALERTAS
# ═══════════════════════════════════════════════════════════════════════════════

def calcular_risco_dinamico(
    rebanho: dict[str, np.ndarray],
    P_t_real: np.ndarray,
    P_t_alvo: np.ndarray,
    p: SimParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula Z, R(t) e alertas para cada animal e pesagem.

         A morbidade não altera p_morte diretamente — ela age sobre a trajetória
         de peso, produzindo Z negativos que amplificam o risco basal via:

             R(t) = p_morte_nasc * 100 * exp(k * max(0, -Z))

         Z negativo (peso abaixo do alvo) → fator exponencial eleva R(t).
         Z positivo (peso acima do alvo)  → exp(0) = 1, R(t) = baseline.
         Clamp bilateral: R(t) ∈ [0, 100].

    Returns:
        Z_atual:  (n_animais, n_pesagens) — Z-scores por pesagem
        R_t:      (n_animais, n_pesagens) — risco dinâmico em %
        alertas:  (n_animais, n_pesagens) — bool: R(t) > limiar_alerta_pct
    """
    dias    = np.array(p.dias_pesagem, dtype=float)
    sigma_t = p.sigma_nascimento + p.lambda_sigma * (dias / p.t_max) ** 2

    # Z-score: (peso_real - peso_alvo) / sigma_t — broadcast (n, d) / (d,)
    Z_atual = (P_t_real - P_t_alvo) / sigma_t

    # p_morte estática do nascimento — vetor (n,), broadcast para (n, d)
    p_morte_nasc = rebanho["p_morte_nasc"]

    R_t = (p_morte_nasc[:, None] * 100.0) * np.exp(
        p.k_decaimento * np.maximum(0.0, -Z_atual)
    )
    R_t = np.clip(R_t, 0.0, 100.0)

    alertas = R_t > p.limiar_alerta_pct

    return Z_atual, R_t, alertas


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MÉTRICAS DE AVALIAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

def avaliar_simulacao(
    rebanho: dict[str, np.ndarray],
    R_t: np.ndarray,
    alertas: np.ndarray,
    p: SimParams,
) -> dict[str, float]:
    """Calcula sensibilidade, especificidade, tempo de detecção e AUC.

    is_problematico = grupos > SAUDAVEL.
    O sinal de detecção chega via Z negativo (trajetória de peso abaixo do alvo),
    que amplifica p_morte_nasc no cálculo de R(t).

    Returns:
        dict com sensibilidade, especificidade, tempo_deteccao_dias, auc
    """
    grupos         = rebanho["grupos"]
    is_problematico = grupos > GrupoMorbidade.SAUDAVEL
    alertou         = alertas.any(axis=1)

    vp = np.sum( is_problematico &  alertou)
    fn = np.sum( is_problematico & ~alertou)
    vn = np.sum(~is_problematico & ~alertou)
    fp = np.sum(~is_problematico &  alertou)

    sensibilidade  = vp / (vp + fn) if (vp + fn) > 0 else 0.0
    especificidade = vn / (vn + fp) if (vn + fp) > 0 else 0.0

    # [C3] Antecedência: T_MAX - dia_do_primeiro_alerta
    dias_arr      = np.array(p.dias_pesagem)
    mask_vp       = is_problematico & alertou
    if mask_vp.any():
        idx_primeiro  = np.argmax(alertas[mask_vp], axis=1)
        dia_primeiro  = dias_arr[idx_primeiro]
        antecedencia  = p.t_max - dia_primeiro          # dias antes da desmama
        tempo_medio   = float(np.mean(antecedencia))
    else:
        tempo_medio = float("nan")

    # AUC: R(t) máximo como score de risco do animal
    max_risk = R_t.max(axis=1)
    auc = roc_auc_score(is_problematico, max_risk)

    return {
        "sensibilidade"       : sensibilidade,
        "especificidade"      : especificidade,
        "tempo_deteccao_dias" : tempo_medio,
        "auc"                 : auc,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 7. LOOP MONTE CARLO
# ═══════════════════════════════════════════════════════════════════════════════

def rodar_monte_carlo(p: SimParams | None = None) -> dict[str, np.ndarray]:
    """Executa N_SIMULACOES rodadas independentes e agrega as métricas.

    Args:
        p: SimParams com todos os parâmetros. Se None, usa os padrões.

    Returns:
        dict com arrays de métricas por simulação.
    """
    if p is None:
        p = SimParams()

    print(
        f"Iniciando Monte Carlo: {p.n_simulacoes} simulações "
        f"× {p.n_animais:,} animais..."
    )
    t0  = time.perf_counter()
    rng = np.random.default_rng(p.seed)

    metricas: dict[str, list[float]] = {
        "sensibilidade"       : [],
        "especificidade"      : [],
        "tempo_deteccao_dias" : [],
        "auc"                 : [],
    }

    for sim in range(p.n_simulacoes):
        rebanho             = gerar_rebanho(p.n_animais, rng, p)
        P_t_real, P_t_alvo  = gerar_trajetoria(rebanho, rng, p)
        _, R_t, alertas     = calcular_risco_dinamico(rebanho, P_t_real, P_t_alvo, p)
        resultado           = avaliar_simulacao(rebanho, R_t, alertas, p)

        for chave, valor in resultado.items():
            metricas[chave].append(valor)

    t1 = time.perf_counter()

    arrays = {k: np.array(v) for k, v in metricas.items()}
    _imprimir_relatorio(arrays, p, duracao=t1 - t0)
    return arrays


# ═══════════════════════════════════════════════════════════════════════════════
# 8. RELATÓRIO FINAL
# ═══════════════════════════════════════════════════════════════════════════════

def _imprimir_relatorio(
    metricas: dict[str, np.ndarray],
    p: SimParams,
    duracao: float,
) -> None:
    """Imprime o resumo estatístico das N simulações."""
    sep = "─" * 62

    def resumo(arr: np.ndarray) -> str:
        validos = arr[~np.isnan(arr)]
        return (
            f"Média: {np.mean(validos):.2f}  "
            f"± {np.std(validos):.2f}  "
            f"[IC95: {np.percentile(validos, 2.5):.2f} – "
            f"{np.percentile(validos, 97.5):.2f}]"
        )

    print(f"\n{sep}")
    print(f" SIMULAÇÃO MONTE CARLO — {p.n_simulacoes} rodadas × {p.n_animais:,} animais")
    print(f" Tempo de execução: {duracao:.1f} s")
    print(sep)

    sens_pct = metricas["sensibilidade"] * 100
    spec_pct = metricas["especificidade"] * 100

    print("\n 1. Sensibilidade — detecção de animais problemáticos")
    print(f"    {resumo(sens_pct)} %")

    print("\n 2. Especificidade — silêncio em animais saudáveis")
    print(f"    {resumo(spec_pct)} %")

    print(f"\n 3. Tempo médio de detecção (dias antes da desmama, t={p.t_max}d)")
    print(f"    {resumo(metricas['tempo_deteccao_dias'])} dias")

    print("\n 4. AUC-ROC (discriminação morbidade vs. saudável)")
    print(f"    {resumo(metricas['auc'])}  [ideal > 0.80]")

    print("\n Parâmetros-chave:")
    print(f"    Limiar de alerta  : R(t) > {p.limiar_alerta_pct:.0f}%")
    print(f"    k_decaimento      : {p.k_decaimento}")
    print(f"    Dias de pesagem   : {list(p.dias_pesagem)}")
    print(f"{sep}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUÇÃO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rodar_monte_carlo(SimParams())