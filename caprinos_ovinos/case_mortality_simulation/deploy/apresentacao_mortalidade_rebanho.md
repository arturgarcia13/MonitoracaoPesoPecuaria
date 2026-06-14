# Simulação Monte Carlo para Validação de um Sistema de Alertas em Zootecnia de Precisão
## Documento de Apresentação — Modelos de Regressão I

---

> [!IMPORTANT]
> Este documento é a referência completa para a apresentação do projeto. Leia na íntegra antes de apresentar. Cada seção é construída de forma que você possa aprofundar ou resumir conforme o tempo disponível.

---

## Sumário

1. [Contexto e Motivação](#1-contexto-e-motivação)
2. [Objetivo do Estudo](#2-objetivo-do-estudo)
3. [Fundamentação Teórica](#3-fundamentação-teórica)
4. [Arquitetura do Modelo](#4-arquitetura-do-modelo)
5. [Equação 1 — Peso ao Nascer](#5-equação-1--peso-ao-nascer-p0)
6. [Equação 2 — Ganho Médio Diário e Trajetória de Peso](#6-equação-2--ganho-médio-diário-gmd-e-trajetória-de-peso)
7. [Equação 3 — Risco Logístico Basal ao Nascimento](#7-equação-3--risco-logístico-basal-ao-nascimento)
8. [Equação 4 — Risco Dinâmico R(t)](#8-equação-4--risco-dinâmico-rt)
9. [Equação 5 — Heteroscedasticidade Residual σ(t)](#9-equação-5--heteroscedasticidade-residual-σt)
10. [Z-score Longitudinal e Disparo de Alerta](#10-z-score-longitudinal-e-disparo-de-alerta)
11. [Grupos de Morbidade e Multiplicadores de GMD](#11-grupos-de-morbidade-e-multiplicadores-de-gmd)
12. [O Loop Monte Carlo](#12-o-loop-monte-carlo)
13. [Métricas de Avaliação](#13-métricas-de-avaliação)
14. [Os Cinco Gráficos e o que Revelam](#14-os-cinco-gráficos-e-o-que-revelam)
15. [Limitações e Possíveis Extensões](#15-limitações-e-possíveis-extensões)
16. [Conexão Explícita com Modelos de Regressão I](#16-conexão-explícita-com-modelos-de-regressão-i)
17. [Glossário Técnico](#17-glossário-técnico)
18. [Referências](#18-referências)

---

## 1. Contexto e Motivação

A mortalidade neonatal de ovinos (cordeiros) é um dos principais fatores de perda econômica em sistemas de produção de pequenos ruminantes no Brasil e no mundo. Estudos indicam que a **mortalidade perinatal** (primeiros 30 dias de vida) pode superar **20% dos nascimentos** em rebanhos mal manejados, sendo o peso ao nascer o preditor mais robusto de sobrevivência.

**Por que usar simulação?**

Dados reais de rebanho são escassos, heterogêneos e muitas vezes inacessíveis em quantidade suficiente para validar modelos estatísticos de alerta com rigor científico. A **simulação Monte Carlo** permite:

- Gerar populações artificiais com distribuições probabilísticas calibradas na literatura científica.
- Testar o desempenho do sistema de alertas sob condições controladas e repetíveis.
- Obter distribuições empíricas de métricas de performance (sensibilidade, especificidade, AUC) ao invés de apenas estimativas pontuais.

O sistema desenvolvido simula **1.000 rebanhos independentes** de **10.000 animais** cada, totalizando **10 milhões de animais virtuais** avaliados.

---

## 2. Objetivo do Estudo

> **Estimar empiricamente, por simulação Monte Carlo, a sensibilidade, a especificidade, o tempo médio de detecção e a AUC-ROC do sistema dinâmico de alertas R(t) para identificação precoce de animais em risco em rebanhos ovinos.**

Secundariamente:
- Verificar se o risco previsto pelo modelo é **calibrado** (probabilidades declaradas correspondem às frequências observadas).
- Visualizar a **trajetória temporal** do peso e o momento de disparo dos alertas.
- Quantificar a **velocidade de detecção acumulada** ao longo dos dias de pesagem.

---

## 3. Fundamentação Teórica

### 3.1 Peso ao Nascer como Preditor de Mortalidade

A relação entre peso ao nascer e mortalidade neonatal segue uma **curva em U**: animais muito leves (hipotrofia) e muito pesados (distocia) têm risco elevado em relação ao peso ótimo.

| Referência | Contribuição |
|-----------|-------------|
| **Freitas et al. (1980)** | Peso médio ao nascer: 3,1 kg para raça Morada Nova |
| **McMillan et al. (1983)** | Faixa ótima de sobrevivência: 3,3–4,1 kg |
| **Hatcher (2009)** | Peso ótimo por tipo de parto (simples, gemelar, trigemelar) |
| **Gardner et al.** | Efeitos de parto, sexo e paridade sobre o peso ao nascer |
| **Sarmento et al. (2010)** | Variância heterogênea crescente σ(t) ao longo da vida |

### 3.2 Raças Contempladas

O modelo é parametrizado para três raças ovinas comuns no Nordeste do Brasil:
- **Morada Nova** — raça nativa; adaptação ao semiárido.
- **Santa Inês** — raça de corte; boa produção de leite.
- **Dorper** — raça exótica; alto rendimento de carcaça.

Os parâmetros biológicos são compatíveis com todas as três raças no contexto de cordeiros neonatos.

---

## 4. Arquitetura do Modelo

O pipeline de simulação é dividido em **quatro etapas sequenciais** por rodada:

```
┌───────────────────────────────────────────────────┐
│  ETAPA 1: Geração do Rebanho (gerar_rebanho)      │
│  → Tipo de parto, sexo, paridade, P0, P_opt, z,   │
│    p_morte_nasc, grupo de morbidade               │
└────────────────────┬──────────────────────────────┘
                     ↓
┌───────────────────────────────────────────────────┐
│  ETAPA 2: Trajetória de Peso (gerar_trajetoria)   │
│  → P_t_real[n, d], P_t_alvo[n, d]                │
└────────────────────┬──────────────────────────────┘
                     ↓
┌───────────────────────────────────────────────────┐
│  ETAPA 3: Risco Dinâmico (calcular_risco_dinamico)│
│  → Z_atual[n, d], R_t[n, d], alertas[n, d]       │
└────────────────────┬──────────────────────────────┘
                     ↓
┌───────────────────────────────────────────────────┐
│  ETAPA 4: Avaliação (avaliar_simulacao)           │
│  → sensibilidade, especificidade, detecção, AUC   │
└───────────────────────────────────────────────────┘
```

Cada uma das **1.000 rodadas** executa o pipeline completo de forma independente, com o mesmo gerador de números aleatórios (sequência determinística a partir da semente `seed=42`). A semente garante **reprodutibilidade total**.

---

## 5. Equação 1 — Peso ao Nascer (P₀)

### 5.1 Formulação Matemática

```
P₀ = β₀ + β_parto + β_sexo + β_matriz + η

η ~ N(0, σ_nascimento²)   onde σ_nascimento = 0,66 kg
P₀ = clip(P₀, 0.5, 8.0)
```

### 5.2 Interpretação dos Coeficientes

| Parâmetro | Valor | Interpretação |
|-----------|-------|---------------|
| β₀ = 4,10 kg | Intercepto | Peso esperado de um **macho**, nascido de **parto simples**, filho de **multípara** |
| β_gemelar = −0,65 kg | Penalidade por gemelaridade | Gemelar pesa em média 0,65 kg a menos (Hatcher 2009) |
| β_trigemelar = −1,40 kg | Penalidade progressiva | Trigemelar pesa 1,40 kg a menos (Gardner et al.) |
| β_fêmea = −0,30 kg | Dimorfismo sexual | Fêmeas são mais leves ao nascer |
| β_primípara = −0,35 kg | Efeito da paridade da mãe | Ovelhas de primeira gestação produzem cordeiros mais leves |
| σ_nascimento = 0,66 kg | Desvio padrão do erro aleatório | Variabilidade genética e ambiental não capturada pelos preditores |

### 5.3 Por que Truncar o P₀?

O clipping entre 0,5 e 8,0 kg impõe **limites biologicamente plausíveis**. Sem truncamento, a cauda da normal poderia gerar valores negativos ou absurdamente altos (e.g., 15 kg), o que quebraria a verossimilhança do modelo. É uma forma de impor **restrições de domínio** sobre o espaço amostral.

### 5.4 Conexão com Regressão

Esta é uma **regressão linear múltipla** com preditores categóricos (tipo de parto, sexo, paridade) codificados como desvios a partir do grupo de referência (macho / simples / multípara). É equivalente a um modelo ANOVA de três fatores.

---

## 6. Equação 2 — Ganho Médio Diário (GMD) e Trajetória de Peso

### 6.1 Formulação do GMD Base

```
GMD_base = gmd_base + penalidade_parto + γ·(P₀ − P_opt) + ε

ε ~ N(0, σ_gmd²)   onde σ_gmd = 0,015 kg/dia
```

| Parâmetro | Valor | Interpretação |
|-----------|-------|---------------|
| gmd_base = 0,252 kg/dia | Crescimento esperado para ovinos de corte | Base da literatura para a fase neonatal |
| penalidade_gemelar = −0,025 kg/dia | Competição por leite | Gêmeos crescem mais devagar |
| penalidade_trigemelar = −0,030 kg/dia | Competição intensificada | Trigêmeos crescem ainda mais devagar |
| γ = 0,02 | Sensibilidade do GMD ao desvio do peso ótimo | Animais que nascem abaixo do ótimo tendem a ter recuperação de crescimento mais lenta |
| ε | Ruído genético individual | Variabilidade entre animais dentro do mesmo grupo |

### 6.2 Trajetória de Peso

A partir do GMD, a trajetória **alvo** (o que o sistema espera de um animal saudável) é:

```
P_t_alvo[t] = P₀ + GMD_base × t
```

A trajetória **real observada** é:

```
P_t_real[t] = P₀ + GMD_efetivo × t + ruído_balança

ruído_balança ~ N(0, σ_balança²)   onde σ_balança = 0,10 kg
```

O `GMD_efetivo` é o `GMD_base` multiplicado por um **fator de grupo de morbidade** (ver Seção 11).

### 6.3 O Modelo de Crescimento é Linear — e por quê isso importa

O modelo adota crescimento linear no intervalo de 0 a 90 dias. Esta é uma **aproximação razoável** para a fase neonatal, quando o animal está em crescimento exponencial acelerado e a curva de crescimento real ainda não atingiu o ponto de inflexão sigmoidal. No entanto, para idades mais avançadas (> 90 dias), modelos não-lineares como **von Bertalanffy** ou **Logístico de Richards** seriam mais adequados. A adoção do modelo linear é uma **decisão de parcimônia** justificável no escopo da fase neonatal.

---

## 7. Equação 3 — Risco Logístico Basal ao Nascimento

### 7.1 Formulação

O risco basal de morte é modelado usando a **função logística (sigmoide)**:

```
z_nasc = α₀ + α₁ · (P₀ − P_opt)²

p_morte_nasc = sigmoid(z_nasc) = 1 / (1 + exp(−z_nasc))
```

### 7.2 Interpretação dos Parâmetros

| Parâmetro | Valor | Interpretação |
|-----------|-------|---------------|
| α₀_macho = −2,5 | Intercepto para machos | Na ausência de desvio do peso ótimo, a probabilidade basal de morte de um macho é `sigmoid(−2.5) ≈ 7,6%` |
| α₀_fêmea = −3,0 | Intercepto para fêmeas | Fêmeas têm vantagem de sobrevivência: `sigmoid(−3.0) ≈ 4,7%` |
| α₁ = 1,2 | Coeficiente quadrático | Penaliza **tanto** animais abaixo **quanto** acima do peso ótimo |
| P_opt_simples = 3,96 kg | Peso ótimo para parto simples | Hatcher (2009) |
| P_opt_gemelar = 3,63 kg | Peso ótimo para parto gemelar | Hatcher (2009) |
| P_opt_trigemelar = 3,44 kg | Peso ótimo extrapolado | Gardner et al. |

### 7.3 Geometria da Curva em U

O termo `(P₀ − P_opt)²` é sempre não-negativo, e como α₁ = 1,2 > 0, o score `z_nasc` **aumenta** conforme o animal se afasta do peso ótimo em qualquer direção. Após a sigmoide, isso produz a **curva em U** característica da mortalidade neonatal: probabilidade de morte mínima próxima ao peso ótimo e crescente nas caudas.

```
P(morte)
  │
  │ *                                       *
  │   *                                   *
  │     *         mínimo               *
  │       *                          *
  │         *                      *
  │           ****            *****
  │               ************
  └─────────────────────────────────────── P₀
               P_opt
```

### 7.4 Por que Usar Regressão Logística?

A variável resposta (morte/sobrevivência) é **binária** (0 ou 1). A regressão logística é o modelo canônico para variáveis resposta binárias na família exponencial, garantindo que a probabilidade prevista permaneça em [0, 1]. A introdução do termo quadrático é uma extensão natural (polinomial) para capturar a relação não-linear de forma paramétrica simples.

### 7.5 Conexão com Regressão Logística (vista em sala)

```
logit(P(Y=1)) = ln(p/(1−p)) = α₀ + α₁·X²

onde X = P₀ − P_opt  (desvio do peso ótimo)
```

Essa é exatamente a formulação da regressão logística com um único preditor quadrático. O modelo não é ajustado por MLE (máxima verossimilhança) a dados reais — os parâmetros são calibrados a partir da literatura —, mas sua estrutura matemática é idêntica.

---

## 8. Equação 4 — Risco Dinâmico R(t)

### 8.1 Formulação

O risco dinâmico combina o **risco basal ao nascimento** com o **desempenho de crescimento observado** em cada pesagem:

```
R(t) = p_morte_nasc × 100 × exp(k × max(0, −Z(t)))

R(t) = clip(R(t), 0, 100)   [restrito ao intervalo [0%, 100%]]
```

### 8.2 Interpretação Detalhada

- **`p_morte_nasc`**: risco basal estático calculado ao nascer (Equação 3). É uma constante por animal, não muda ao longo do tempo.
- **`max(0, −Z(t))`**: só entra em ação quando o Z-score é **negativo**, ou seja, quando o animal está **abaixo** do peso esperado. Quando o animal está no alvo ou acima (Z ≥ 0), `max(0,0) = 0`, e `exp(0) = 1`, portanto `R(t) = p_morte_nasc × 100` (risco basal).
- **`exp(k × max(0, −Z))`**: fator de amplificação exponencial. Para Z = −1 e k = 0,2: `exp(0,2) ≈ 1,22` (aumento de 22% no risco). Para Z = −5: `exp(1,0) ≈ 2,72` (risco aumenta 172%).
- **`k = 0,2`** (k_decaimento): parâmetro de sensibilidade. Valores maiores tornam o sistema mais agressivo; valores menores, mais conservador.

### 8.3 Analogia com Funções de Risco em Sobrevivência

A estrutura `R(t) = R₀ · exp(β · X)` é matematicamente idêntica ao **modelo de riscos proporcionais de Cox**, onde:
- `R₀` = risco basal (hazard baseline).
- `X = max(0, −Z(t))` = covariável dinâmica.
- `β = k` = log do hazard ratio.

O que o código implementa é, portanto, uma versão **discreta e simplificada** de um modelo de risco proporcional com covariável tempo-variante.

### 8.4 Assimetria Intencional

O fato de o modelo só amplificar o risco quando Z < 0 (e não quando Z > 0) é uma **decisão de design clínico**: o sistema não penaliza animais que crescem acima do esperado, pois isso geralmente não é um problema na fase neonatal (pelo menos para as raças em questão). Isso é uma **hipótese estatística embutida no modelo**, que deveria ser testada em dados reais.

---

## 9. Equação 5 — Heteroscedasticidade Residual σ(t)

### 9.1 Formulação

```
σ(t) = σ_nascimento + λ_sigma × (t / T_max)²

σ_nascimento = 0,66 kg
λ_sigma = 1,5
T_max = 90 dias
```

### 9.2 Valores em Cada Pesagem

| Dia de pesagem (t) | σ(t) calculado |
|---------------------|---------------|
| t = 15 dias | 0,66 + 1,5 × (15/90)² = 0,66 + 0,042 ≈ **0,70 kg** |
| t = 30 dias | 0,66 + 1,5 × (30/90)² = 0,66 + 0,167 ≈ **0,83 kg** |
| t = 45 dias | 0,66 + 1,5 × (45/90)² = 0,66 + 0,375 ≈ **1,04 kg** |
| t = 60 dias | 0,66 + 1,5 × (60/90)² = 0,66 + 0,667 ≈ **1,33 kg** |

### 9.3 Fundamentação

Sarmento et al. (2010) demonstraram que a variância do peso em ovinos **aumenta com a idade** — fenômeno chamado de **heteroscedasticidade** na literatura de modelos lineares. Isso é esperado biologicamente: animais que nascem com o mesmo peso divergem progressivamente em função de diferenças genéticas, nutricionais e sanitárias que se acumulam com o tempo.

O modelo captura isso com uma função quadrática em `t`, que é a forma mais simples de variância não-constante crescente. Em regressão linear clássica, ignorar a heteroscedasticidade não torna os estimadores viesados, mas torna os **erros padrão incorretos** e os **testes de hipótese inválidos**. O modelo usa σ(t) corretamente no denominador do Z-score para corrigir isso.

### 9.4 Implicação para o Z-score

O Z-score calculado como `(P_real − P_alvo) / σ(t)` é um **resíduo padronizado heteroscedástico**. Um animal que está 1 kg abaixo do alvo aos 15 dias (Z ≈ −1,43) é muito mais preocupante do que um animal que está 1 kg abaixo do alvo aos 60 dias (Z ≈ −0,75), pois σ(t) é maior no dia 60.

---

## 10. Z-score Longitudinal e Disparo de Alerta

### 10.1 Cálculo do Z-score

```
Z(t) = (P_t_real − P_t_alvo) / σ(t)

onde σ(t) é a variância crescente (Equação 5)
```

### 10.2 Regra de Alerta

```
alerta[animal, t] = TRUE   ⟺   R(t) > limiar_alerta_pct

limiar_alerta_pct = 30%
```

O alerta é disparado **individualmente por animal e por pesagem**. Um animal pode ter o alerta disparado em uma pesagem mas não em outra.

### 10.3 Definição de "Animal Alertado"

Para fins de cálculo das métricas binárias (sensibilidade, especificidade), um animal é considerado "alertado" se `alerta.any(axis=1)` for verdadeiro — ou seja, se **pelo menos uma das pesagens** disparou o alerta ao longo de todo o período de monitoramento (dias 15, 30, 45, 60).

### 10.4 Intuição do Z-score

Um Z-score de −2 significa que o animal está **2 desvios padrão abaixo** do esperado para sua idade. Isso corresponde, em distribuição normal, ao **2,3° percentil** — apenas 2,3% dos animais saudáveis estariam tão abaixo do alvo por acaso. O sistema usa isso como evidência de comprometimento sanitário.

---

## 11. Grupos de Morbidade e Multiplicadores de GMD

### 11.1 Os Quatro Grupos

| Grupo | Código | Proporção na pop. | Multiplicador de GMD | Descrição |
|-------|--------|-------------------|----------------------|-----------|
| SAUDAVEL | 0 | 70% | × 1,00 | Sem comprometimento |
| SUBNUTRIDO | 1 | 15% | × 0,70 | Déficit nutricional crônico |
| DOENTE | 2 | 10% | × 0,50 (após dia 20) | Evento infeccioso agudo |
| CRITICO | 3 | 5% | × 0,20 | Falência múltipla |

### 11.2 O Grupo DOENTE tem Comportamento Dinâmico

O grupo DOENTE é o único que tem **comportamento fásico**:
- **Dias 0–20**: cresce normalmente (multiplicador = 1,00).
- **Dias > 20**: o evento infeccioso se manifesta e o GMD cai para 50% do basal.

Matematicamente, a trajetória real do animal doente após o dia 20 é:

```
P_real(t) = P₀ + GMD_base × dia_inicio_doenca + (GMD_base × 0,50) × (t − dia_inicio_doenca)

para t > dia_inicio_doenca = 20
```

Isso cria uma **descontinuidade na derivada** (mudança de inclinação) na trajetória ao dia 20, que o sistema de alertas deve detectar nas pesagens subsequentes (dias 30, 45, 60).

### 11.3 Variável Resposta (Ground Truth)

Para cálculo das métricas de avaliação:

```python
is_problematico = grupos > GrupoMorbidade.SAUDAVEL
# True para grupos 1 (SUBNUTRIDO), 2 (DOENTE), 3 (CRITICO)
# False para grupo 0 (SAUDAVEL)
```

Isto define o **rótulo real** (verdade oculta) de cada animal para o problema de classificação binária.

---

## 12. O Loop Monte Carlo

### 12.1 Estrutura do Loop

```python
rng = np.random.default_rng(seed=42)

for sim in range(1_000):
    rebanho = gerar_rebanho(10_000, rng, p)
    P_t_real, P_t_alvo = gerar_trajetoria(rebanho, rng, p)
    Z_atual, R_t, alertas = calcular_risco_dinamico(rebanho, P_t_real, P_t_alvo, p)
    resultado = avaliar_simulacao(rebanho, R_t, alertas, p)
    # acumula métricas
```

### 12.2 Por que 1.000 Rodadas?

A escolha de 1.000 rodadas é guiada pelo **Teorema do Limite Central** aplicado à Monte Carlo: com n suficientemente grande, a distribuição das médias amostrais converge para a normal, permitindo calcular intervalos de confiança. Para proporções (sensibilidade, especificidade), o erro padrão de Monte Carlo é `σ/√n`; com n=1.000, o erro padrão é reduzido a 3,2% do desvio padrão verdadeiro.

### 12.3 Independência das Rodadas

Cada rodada usa o **mesmo gerador de números aleatórios** (`rng`), mas como o gerador avança seu estado a cada chamada, cada rodada produz amostras **estatisticamente independentes** e **diferentes** entre si. O estado inicial (semente 42) é fixo, garantindo que toda execução do código produza exatamente os mesmos resultados — essa propriedade é chamada de **reprodutibilidade computacional**.

### 12.4 O que é Fixo e o que Varia

- **Fixo**: parâmetros biológicos (β's, α's, σ's, proporções populacionais).
- **Varia**: composição aleatória do rebanho (tipo de parto, sexo, paridade, peso ao nascer, grupo de morbidade) e os ruídos de medição.

---

## 13. Métricas de Avaliação

### 13.1 Matriz de Confusão

Para cada rodada, a matriz de confusão é calculada:

```
                    ┌──────────────────┬──────────────────┐
                    │  Alerta = TRUE   │  Alerta = FALSE  │
┌───────────────────┼──────────────────┼──────────────────┤
│ Problemático = TRUE│       VP         │        FN        │
├───────────────────┼──────────────────┼──────────────────┤
│ Problemático = FALSE│      FP        │        VN        │
└───────────────────┴──────────────────┴──────────────────┘

VP = Verdadeiro Positivo   FN = Falso Negativo
VN = Verdadeiro Negativo   FP = Falso Positivo
```

### 13.2 Sensibilidade (Recall / Taxa de Verdadeiros Positivos)

```
Sensibilidade = VP / (VP + FN)
```

Responde à pergunta: **"De todos os animais problemáticos, qual proporção o sistema alertou corretamente?"**

- Alta sensibilidade é crucial em medicina veterinária: **falso negativo tem consequências clínicas** (animal doente não tratado morre).

### 13.3 Especificidade (Taxa de Verdadeiros Negativos)

```
Especificidade = VN / (VN + FP)
```

Responde à pergunta: **"De todos os animais saudáveis, qual proporção o sistema silenciou corretamente?"**

- Alta especificidade evita desperdício de recursos com animais falsamente classificados como doentes.

### 13.4 Trade-off Sensibilidade × Especificidade

O limiar de alerta `limiar_alerta_pct = 30%` é um **ponto de corte** na distribuição de R(t). Abaixar o limiar aumenta a sensibilidade (mais alertas disparados) mas reduz a especificidade (mais falsos positivos). Esse trade-off é visualizado pela **curva ROC**.

### 13.5 AUC-ROC

```
AUC = ∫₀¹ TPR(FPR) dFPR
```

A AUC mede a **capacidade discriminatória** do modelo: a probabilidade de que, dado um animal problemático e um saudável escolhidos aleatoriamente, o modelo atribua um score de risco maior ao problemático. AUC = 0,5 é equivalente a sorte; AUC > 0,8 é considerado bom para aplicações clínicas.

No código, o score de risco por animal é `max(R_t)` — o máximo risco observado ao longo de todas as pesagens.

### 13.6 Tempo de Detecção (Lead Time)

```
lead_time[animal] = T_max − dia_do_primeiro_alerta

onde T_max = 90 dias (data da desmama)
```

Essa métrica responde: **"Com quantos dias de antecedência em relação à desmama o sistema detectou o animal?"** Lead time maior = mais tempo para intervenção veterinária.

Apenas animais **verdadeiros positivos** (is_problematico = True **e** alertou = True) são incluídos no cálculo.

### 13.7 Curva de Detecção Acumulada

Para cada simulação, calcula-se a fração acumulada de animais problemáticos detectados em função do dia de pesagem:

```python
primeiro_alerta_idx = np.argmax(alertas[vp_mask], axis=1)
for j in range(len(p.dias_pesagem)):
    det_ate_agora = np.sum(primeiro_alerta_idx <= j)
    cum_det_fractions[sim, j] = det_ate_agora / total_problematicos
```

A curva resultante é uma **função de distribuição empírica acumulada** (ECDF) do tempo de primeira detecção.

### 13.8 Curva de Calibração

A calibração mede se as probabilidades declaradas pelo modelo correspondem às frequências observadas. Para avaliá-la:

1. Os animais são agrupados em **10 decis** pelo `max(R_t)`.
2. Para cada decil, calcula-se a média do risco previsto e a proporção real de animais problemáticos.
3. Um modelo perfeitamente calibrado gera pontos sobre a diagonal do gráfico (45°).

Desvios abaixo da diagonal indicam **superestimação do risco**; acima, **subestimação**.

---

## 14. Os Cinco Gráficos e o que Revelam

### Gráfico 1 — Histogramas: Sensibilidade e Especificidade

- **O que mostra**: distribuição empírica das métricas ao longo das 1.000 rodadas Monte Carlo.
- **O que interpretar**: largura do histograma = variabilidade do sistema. Uma distribuição concentrada (baixo desvio padrão) indica que o desempenho é **robusto** à variação aleatória dos rebanhos. A linha KDE (Kernel Density Estimate) suaviza a distribuição para facilitar a leitura.
- **Pergunta esperada da turma**: "Por que há variabilidade se os parâmetros são fixos?" **Resposta**: porque cada rebanho de 10.000 animais é uma amostra aleatória diferente — a composição por tipo de parto, sexo e grupo de morbidade varia entre rodadas.

### Gráfico 2 — Curva ROC com Intervalo de Confiança

- **O que mostra**: a curva ROC **média** (média das TPR ao longo de todas as rodadas) e a banda de ± 1 desvio padrão em cinza.
- **O que interpretar**: a área sob a curva média (AUC) quantifica o poder discriminatório. A banda cinza indica a **variabilidade** do desempenho — uma banda estreita indica estabilidade.
- **Como a curva média é construída**: todas as curvas ROC individuais são interpoladas sobre uma grade comum de FPR (100 pontos entre 0 e 1) e depois somadas elemento a elemento. A média dessas curvas interpoladas é plotada.

### Gráfico 3 — Curva de Calibração

- **O que mostra**: risco previsto médio por decil (eixo X) versus proporção real de animais problemáticos por decil (eixo Y).
- **O que interpretar**: pontos acima da diagonal indicam que o modelo **subestima o risco** (mais animais doentes do que o modelo prevê). Pontos abaixo indicam **superestimação**.
- **Importante**: os decis são construídos **acumulando** as contribuições de todas as 1.000 rodadas, então os bins com baixa contagem individual podem ter contagem acumulada suficiente para ser representativos.

### Gráfico 4 — Trajetória Temporal do Peso e Alertas

- **O que mostra**: um exemplo real (última rodada da simulação) com três animais representativos: saudável, subnutrido e crítico.
- **O que interpretar**: a linha preta tracejada é a **curva esperada** (alvo do sistema). Animais abaixo dela geram Z negativo e têm R(t) amplificado. O marcador circular vermelho indica o **momento em que o alerta foi disparado pela primeira vez**.
- **Detalhe técnico**: o código busca um animal de cada grupo que satisfaça a condição `(grupo == X) & (alertas.any(axis=1))`, garantindo que os exemplos escolhidos sejam representativos da dinâmica de alerta.

### Gráfico 5 — Curva de Detecção Acumulada (Step Function)

- **O que mostra**: média da proporção de animais problemáticos detectados em função do dia de pesagem.
- **Por que usar degraus (step function)**: a detecção só acontece nos dias de pesagem (15, 30, 45, 60) — não existe informação entre eles. A função em degraus reflete corretamente essa natureza **discreta** do processo.
- **O que interpretar**: uma curva que sobe rapidamente nos primeiros dias indica que o sistema detecta precocemente. A assíntota máxima é a sensibilidade global do sistema (nunca chega a 100% se há falsos negativos).

---

## 15. Limitações e Possíveis Extensões

### 15.1 Limitações do Modelo Atual

| Limitação | Impacto | Extensão Possível |
|-----------|---------|------------------|
| Crescimento linear (GMD constante) | Subestima a curva sigmoidal real para idades > 60 dias | Usar von Bertalanffy, Richards ou Brody |
| σ(t) quadrática simples | Pode não capturar a variância real que cresce mais lentamente e depois acelera | Ajustar σ(t) com dados longitudinais reais por MLE |
| Parâmetros calibrados por literatura, não por dados locais | Parâmetros podem não representar o rebanho específico do produtor | Estimar por MLE ou Bayesiano com dados do rebanho |
| Grupos de morbidade independentes do peso ao nascer | Em biologia, animais mais leves tendem a ter maior chance de ser subnutridos ou críticos | Modelar grupos de morbidade como função do P₀ (modelo de seleção) |
| Sem correlação entre sexo e tipo de parto | Gêmeos têm distribuição de sexo ligeiramente diferente de partos simples | Usar distribuição multinomial conjunta |
| AUC estática (max risk) | Perde a informação temporal de como o risco evolui | Usar AUC dinâmica (cumulative/dynamic AUC de Hung & Chiang) |
| Calibração por decis simples | Bins de baixa contagem podem ser instáveis | LOESS calibration, Platt scaling, isotonic regression |
| k e limiar arbitrários | Desempenho sensível a esses parâmetros sem validação | Análise de sensibilidade e validação cruzada com dados reais |

### 15.2 Como Validar o Modelo com Dados Reais

Um protocolo de validação proposto:

1. **Coleta**: dados de peso em múltiplos momentos de rebanhos reais com registros de mortalidade.
2. **Calibração**: ajustar β's, α's e σ's aos dados locais por MLE.
3. **Split**: dividir em treino (calibração) e teste (validação).
4. **Validação**: calcular sensibilidade, especificidade e AUC no conjunto de teste e comparar com os valores simulados.
5. **Calibração**: construir a curva de calibração com dados reais e verificar aderência.

---

## 16. Conexão Explícita com Modelos de Regressão I

### 16.1 Regressão Linear Múltipla (Equação 1 — Peso ao Nascer)

```
P₀ = β₀ + β₁·gemeo + β₂·trigemeo + β₃·femea + β₄·primipara + η

η ~ N(0, σ²)
```

Esta é a forma matricial `Y = Xβ + ε` da regressão linear múltipla com preditores binários (codificação dummy). O estimador de mínimos quadrados ordinários (MQO) seria `β̂ = (X'X)⁻¹X'Y`. No código, os parâmetros são conhecidos (calibrados da literatura), mas a estrutura é idêntica.

### 16.2 Regressão Logística com Preditor Quadrático (Equação 3)

```
log(p/(1−p)) = α₀ + α₁·(P₀ − P_opt)²
```

Esta é exatamente a forma de um **GLM com família Binomial e função de ligação logit**, com a covariável sendo o quadrado do desvio do peso ótimo. O modelo pode ser reescrito:

```
log(p/(1−p)) = α₀ + α₁·P₀² − 2α₁·P_opt·P₀ + α₁·P_opt²
                    ─────────────────────────────────────
                    coeficientes lineares em P₀ e P₀²
```

Ou seja, é uma regressão logística polinomial de grau 2.

### 16.3 Modelo de Variância Heteroscedástica (Equação 5)

```
Var(ε_t) = σ(t)² = [σ_nascimento + λ·(t/T_max)²]²
```

Em regressão linear clássica, assumimos homocedasticidade: `Var(ε) = σ² (constante)`. Quando essa suposição é violada, os estimadores MQO continuam não-viesados, mas perdem eficiência e os testes de hipótese são inválidos. O modelo usa σ(t) como parte do denominador do Z-score — equivalente a uma **ponderação** que corrige a heteroscedasticidade (WLS, Weighted Least Squares).

### 16.4 O Z-score como Resíduo Padronizado

```
Z(t) = (observado − esperado) / desvio_padrão_esperado
```

Em diagnóstico de regressão linear, os **resíduos padronizados** (ou studentizados) seguem essa mesma lógica: dividimos o resíduo pelo desvio padrão estimado para torná-los comparáveis. Um resíduo padronizado |Z| > 2 é um sinal de outlier em modelos de regressão — exatamente a lógica usada aqui para identificar animais que "fogem" da trajetória esperada.

### 16.5 A Curva ROC e o AUC como Avaliação de Modelo

A curva ROC é o padrão ouro para avaliação de **modelos de classificação binária**. Ela é particularmente relevante em Regressão I porque:

1. A regressão logística produz um **score contínuo** (probabilidade prevista).
2. Escolher um ponto de corte para transformar esse score em classificação binária é uma decisão operacional.
3. A curva ROC mostra o desempenho para **todos os possíveis pontos de corte** simultaneamente.
4. A AUC é equivalente à estatística de Wilcoxon-Mann-Whitney: a probabilidade de um modelo ranquear corretamente um positivo acima de um negativo.

---

## 17. Glossário Técnico

| Termo | Definição |
|-------|-----------|
| **Monte Carlo** | Método numérico que usa amostragem aleatória repetida para obter resultados estatísticos |
| **GMD** | Ganho Médio Diário — taxa de crescimento em kg/dia |
| **Sigmoid / Logística** | Função `1/(1+e⁻ˣ)` que mapeia qualquer real para (0,1) |
| **Z-score** | Resíduo padronizado: quantas unidades de desvio padrão um valor está distante da média |
| **ROC** | Receiver Operating Characteristic — curva que plota TPR × FPR para todos os limiares |
| **AUC** | Area Under the Curve — área sob a curva ROC; mede poder discriminatório |
| **Sensibilidade** | Proporção de positivos verdadeiros detectados (= Recall = TPR) |
| **Especificidade** | Proporção de negativos verdadeiros silenciados corretamente (= TNR) |
| **Calibração** | Correspondência entre probabilidade prevista e frequência observada |
| **KDE** | Kernel Density Estimation — suavizador não-paramétrico de distribuição |
| **Heteroscedasticidade** | Variância do erro que não é constante (depende de t ou de X) |
| **Lead time** | Antecedência com que um evento é detectado antes de ocorrer |
| **Hazard / Risco** | Probabilidade instantânea de ocorrência de um evento em um dado momento |
| **ECDF** | Empirical Cumulative Distribution Function — distribuição acumulada empírica |
| **WLS** | Weighted Least Squares — mínimos quadrados ponderados para heteroscedasticidade |
| **MLE** | Maximum Likelihood Estimation — método de estimação dos parâmetros |
| **GLM** | Generalized Linear Model — família de modelos que inclui regressão logística |
| **Seed** | Semente do gerador de números aleatórios — garante reprodutibilidade |

---

## 18. Referências

| Referência | Relevância no modelo |
|-----------|----------------------|
| Freitas et al. (1980) | Peso médio ao nascer de Morada Nova: 3,1 kg |
| McMillan et al. (1983) | Faixa ótima de sobrevivência: 3,3–4,1 kg |
| Hatcher (2009) | P_opt por tipo de parto; β₀ para machos simples; pesos gemelar e trigemelar |
| Gardner et al. | Efeitos de parto, sexo e paridade sobre o peso; β_trigemeo |
| Sarmento et al. (2010) | Variância heterogênea crescente σ(t) ao longo da vida |
| Hosmer & Lemeshow (2000) | *Applied Logistic Regression* — base teórica da regressão logística usada |
| Harrell (2015) | *Regression Modeling Strategies* — avaliação de modelos preditivos, AUC, calibração |
| Fawcett (2006) | *An Introduction to ROC Analysis* — construção e interpretação da curva ROC |
| Cox & Snell (1989) | *Analysis of Binary Data* — modelos binários e calibração |
| Hung & Chiang (2010) | AUC dinâmica para dados de sobrevivência com covariáveis tempo-variantes |

---

> **Dica de apresentação**: Comece pela **Seção 1 (Motivação)**, vá para a **Seção 4 (Arquitetura)**, e então percorra as equações (5 a 10) mostrando o código ao lado. Termine com os **cinco gráficos (Seção 14)** e a **conexão com Regressão I (Seção 16)**. Reserve pelo menos 10 minutos para perguntas e use o **Glossário (Seção 17)** como referência rápida durante a arguição.

> **Arquivos-fonte do projeto:**
> - `mortality_simulation_view_v8.py` — núcleo do modelo
> - `mortality_simulation_charts_v8.py` — visualizações
