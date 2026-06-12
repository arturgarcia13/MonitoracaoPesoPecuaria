# Sistema de Monitoramento Biometrico de Ovinos

Sistema preditivo baseado em **tres equacoes sequenciais** extraidas do documento `regressao.md`,
estruturado seguindo os principios de Engenharia de Software (RUP, SOLID, TDD).

---

## 1. As Duas Equacoes do Documento (+ Risco)

O documento define um fluxo de **duas regressoes distintas e sequenciais**, mais um modelo de risco:

| # | Equacao | Secao | Objetivo | Features | Target |
|---|---|---|---|---|---|
| 1 | `P0 = 4.10 + beta_parto + beta_sexo + beta_matriz + eta` | 3.1 | Estimar **peso ao nascer** | Sexo, Tipo_Parto, Ordem_Parto | Peso_Nascer |
| 2 | `Pt = P0 + GMD_i * t + eta_t` | 3.2 | Projetar **trajetoria de peso** | `P0`, `t`, `P0*t`, `Multiplo*t` | Peso_Atual |
| 3 | `P(Y=1) = sigmoid(alpha0 + alpha1*(P0-4)^2)` | 3.3 | Avaliar **risco de obito** | (P0 - 4.0)^2 | Y_Morto (0/1) |

### Por que as equacoes sao separadas?

A Equacao 1 estima o P0 com base em **quem e o animal** (caracteristicas fixas de parto).
A Equacao 2 usa esse P0 como **ancora** para projetar o crescimento ao longo do tempo.
A Equacao 3 avalia o **risco imediato** logo apos o nascimento, antes de qualquer projecao.

Isso reflete o passo a passo biologico real:
```
    Nascimento do animal
          |
    [PASSO 1] Estimar P0 pelas caracteristicas do parto (Eq. 1)
          |
    [PASSO 2] Projetar curva de crescimento Pt = f(P0, t)   (Eq. 2)
          |
    [PASSO 3] Avaliar risco neonatal P(Y=1) = f(P0)         (Eq. 3)
          |
    [PASSO 4] Monitoramento continuo: Z-Score individual a cada pesagem
```

---

## 2. Arquitetura do Projeto (RUP)

```text
caprinos_ovinos/
|
+-- docs/
|   +-- requirements/dataset_simulado/    # CSVs gerados (animais + pesagens)
|   +-- architecture/graficos/            # 7 PNGs exportados
|
+-- deploy/modelos/                        # 3 arquivos .pkl treinados
|   +-- modelo_equacao1_peso_nascer.pkl
|   +-- modelo_equacao2_trajetoria_peso.pkl
|   +-- modelo_equacao3_risco_neonatal.pkl
|
+-- src/main/
|   +-- domain/
|   |   +-- calculadora_biometrica.py     # Efeitos fixos: beta_sexo, beta_parto, beta_ordem, GMD
|   |   +-- avaliador_risco.py            # Curva logistica de mortalidade neonatal
|   +-- simulation/
|   |   +-- data_generator.py             # Simula 500 animais + pesagens (0 a 90 dias)
|   +-- monitoring/
|   |   +-- monitorador_rebanho.py        # Z-Score individual + alertas (Normal/Atencao/Critico)
|   +-- ml/
|   |   +-- trainer.py                    # 3 metodos, 1 por equacao do documento
|   |   +-- predictor.py                  # Fluxo sequencial Passo 1 -> 2 -> 3
|   +-- visualization/
|   |   +-- plotter.py                    # 7 graficos para Zootecnia e Setor Privado
|   +-- main.py                           # Pipeline com 5 fases explicitas
|
+-- test/verification/
|   +-- test_sistema_especialista.py      # 6 testes: formulas P0, GMD, risco
|   +-- test_data_generator.py            # 1 teste: integridade das tabelas
|   +-- test_ml.py                        # 11 testes: 1 por equacao + pipeline sequencial
|   +-- test_monitorador.py               # 6 testes: Z-Score e alertas
|
+-- requirements.txt
+-- regressao.md
```

---

## 3. Modulos de ML em Detalhe

### `trainer.py` — TreinadorModelos

Tres metodos, um por equacao do documento:

```python
# Equacao 1 (Secao 3.1): features categoricas -> P0
treinador.treinar_equacao_p0(df_animais)

# Equacao 2 (Secao 3.2): Aprende o GMD individual usando interacoes (P0*t e Multiplo*t) -> Pt
treinador.treinar_equacao_pt(df_animais, df_pesagens)

# Equacao 3 (Secao 3.3): (P0 - 4.0)^2 -> P(Y=1)
treinador.treinar_equacao_risco(df_animais)
```

Metricas obtidas na ultima execucao:

| Equacao | Metrica | Valor |
|---|---|---|
| Equacao 1 (P0) | RMSE | 0.4961 kg |
| Equacao 1 (P0) | R2 | 0.38 |
| Equacao 2 (Pt) | RMSE | 0.00 kg (Ajuste perfeito) |
| Equacao 2 (Pt) | R2 | 1.000 |
| Equacao 3 (Risco) | Accuracy | 74.8% |
| Equacao 3 (Risco) | AUC-ROC | 0.70 |

### `predictor.py` — PreditivoModelos

Encapsula o fluxo sequencial com metodos nomeados conforme os passos do documento:

```python
preditivo = PreditivoModelos()

# Passo 1: Estimar P0
p0 = preditivo.estimar_peso_nascer("M", "Simples", "Multipara")

# Passo 2: Projetar peso em t=90 dias (passando tipo_parto para calcular GMD corretamente)
pt = preditivo.projetar_peso_em_t(p0, dias_vida=90, tipo_parto="Simples")

# Passo 3: Avaliar risco de obito
risco = preditivo.avaliar_risco_neonatal(p0_real)

# Ou executar tudo de uma vez:
resultado = preditivo.executar_pipeline_completo(
    sexo="M", tipo_parto="Simples", ordem_parto="Multipara",
    peso_nascer_real=4.10, dias_projecao=90
)
```

---

## 4. Sistema de Monitoramento Z-Score (`monitoring/`)

Implementa a regra de negocio principal do documento (Secao 1):

```
Z_atual = (P_t_real - mu_Pt) / sigma_t
```

onde:
- `mu_Pt = P0 + GMD * t` (curva individual do animal)
- `sigma_t = 0.35 + 1.5 * (t/90)^2` (variancia crescente)

| Status | Condicao | Acao |
|---|---|---|
| [NORMAL] | Z >= -1.0 | Rotina |
| [ATENCAO] | -2.0 < Z < -1.0 | Observacao intensificada |
| [CRITICO] | Z <= -2.0 | Notificacao imediata |

---

## 5. Visualizacoes (7 graficos)

**Para Zootecnistas:**
- `curva_crescimento_percentis.png` — Corredor P5/P50/P95 do rebanho
- `risco_neonatal_curva_u.png` — Taxa de mortalidade empirica por faixa de peso
- `curva_logistica_mortalidade.png` — Curva continua teorica (Cenario B do documento)
- `zscore_temporal_rebanho.png` — Trajetoria Z-Score de 20 animais com faixas de alerta

**Para o Setor Privado:**
- `correlacao_peso_gmd_negocios.png` — Correlacao P0 x GMD com coeficiente r
- `distribuicao_peso_90d.png` — Histograma do peso na desmama
- `distribuicao_peso_por_tipo_parto.png` — KDE comparando Simples vs Gemeo vs Trigemeos

---

## 6. Testes (23 passed)

```
test_sistema_especialista.py   6 testes  -- formulas do dominio
test_data_generator.py         1 teste   -- integridade da simulacao
test_ml.py                    11 testes  -- equacoes 1, 2, 3 + pipeline sequencial
test_monitorador.py            6 testes  -- Z-Score, sigma_t, faixas de alerta
```

---

## 7. Como Executar

```powershell
# Configurar ambiente
$env:PYTHONPATH = "."

# Pipeline completo (5 fases)
.\.venv\Scripts\python.exe src\main\main.py

# Suite de testes
.\.venv\Scripts\pytest.exe test\verification\ -v
```
