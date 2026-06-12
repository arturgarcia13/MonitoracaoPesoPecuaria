# Sistema de Monitoramento Biométrico de Ovinos

Sistema preditivo e analítico baseado nas equações biométricas documentadas em `regressao.md`, estruturado segundo os princípios de Engenharia de Software (RUP, SOLID, TDD).

---

## 1. Visão Geral

O sistema implementa três funcionalidades centrais, diretamente mapeadas às seções do documento técnico:

| Funcionalidade | Seção do Documento | Módulo |
|---|---|---|
| Cálculo de Peso ao Nascer (P₀) com efeitos fixos | 3.1 | `domain/calculadora_biometrica.py` |
| Curva de crescimento longitudinal com GMD | 3.1, 3.2 | `domain/calculadora_biometrica.py` |
| Variância residual crescente (σ_t) | 3.2 | `simulation/data_generator.py`, `monitoring/` |
| Z-Score individual + alertas Verde/Amarelo/Vermelho | Seção 1 (Arquitetura) | `monitoring/monitorador_rebanho.py` |
| Curva logística de risco neonatal (Curva em U) | 3.3 Cenário B | `domain/avaliador_risco.py` |
| Modelos de ML treináveis com backup `.pkl` | Seção 1 (Backlog) | `ml/trainer.py`, `ml/predictor.py` |

---

## 2. Arquitetura e Padrões

- **RUP**: Estrutura de pastas mapeia os workflows de implementação, teste e deploy.
- **SOLID / SRP**: Cada classe possui uma única responsabilidade.
- **TDD**: Toda funcionalidade de domínio possui testes que passam antes do código.
- **Clean Code**: Nomes em português alinhados à linguagem ubíqua da pecuária.

### Estrutura de Diretórios

```text
caprinos_ovinos/
│
├── docs/
│   ├── requirements/dataset_simulado/    # CSVs gerados pelo simulador
│   └── architecture/graficos/            # 7 PNGs exportados
│
├── deploy/modelos/                        # .pkl dos modelos de ML
│
├── src/main/
│   ├── domain/
│   │   ├── calculadora_biometrica.py     # Efeitos fixos: β_sexo, β_parto, β_ordem, GMD
│   │   └── avaliador_risco.py            # Curva logística de mortalidade neonatal
│   ├── simulation/
│   │   └── data_generator.py             # Simula 500 animais + pesagens (0 a 90 dias)
│   ├── monitoring/
│   │   └── monitorador_rebanho.py        # Z-Score individual + alertas (Normal/Atenção/Crítico)
│   ├── ml/
│   │   ├── trainer.py                    # Regressão Linear (crescimento) + Logística (mortalidade)
│   │   └── predictor.py                  # Inferência via modelos salvos
│   ├── visualization/
│   │   └── plotter.py                    # 7 gráficos para Zootecnia e Setor Privado
│   └── main.py                           # Orquestrador do pipeline completo
│
├── test/verification/
│   ├── test_sistema_especialista.py      # 6 testes: fórmulas de P₀, GMD e risco
│   ├── test_data_generator.py            # 1 teste: integridade das tabelas simuladas
│   ├── test_ml.py                        # 2 testes: treinamento e persistência dos .pkl
│   └── test_monitorador.py              # 6 testes: Z-Score, σ_t, faixas de alerta
│
├── requirements.txt
└── regressao.md                          # Documento técnico base
```

---

## 3. Detalhamento dos Módulos

### 3.1. Domínio (`src/main/domain/`)

**`calculadora_biometrica.py`** — Implementa as equações da Seção 3.1:

- `calcular_peso_esperado(sexo, tipo_parto, ordem_parto)` → aplica os coeficientes β da regressão:
  - Intercepto base: **β₀ = 4.10 kg** (macho, parto simples, mãe multípara)
  - β_sexo = -0.30 kg (fêmea)
  - β_parto = -0.65 kg (gêmeo) / -1.40 kg (trigêmeos)
  - β_ordem = -0.35 kg (primípara)

- `calcular_gmd_esperado(peso_nascer, tipo_parto)` → implementa a correlação GMD ↔ P₀:
  - GMD = 0.272 + γ(P₀ - 4.0), onde γ = 0.02
  - Penalização de -0.025 kg/dia para parto múltiplo

**`avaliador_risco.py`** — Implementa o Cenário B (Seção 3.3):

- Curva logística quadrática: z = α₀ + α₁(P₀ - P_opt)²
- α₀ = -2.5, α₁ = 1.2, P_opt = 4.0 kg
- Produz a Curva em U: penaliza tanto baixo peso quanto macrossomia

### 3.2. Monitoramento (`src/main/monitoring/`)

**`monitorador_rebanho.py`** — Núcleo do sistema de alertas (Seção 1 do documento):

Fórmula implementada:
```
Z_atual = (P_t_real - μ_Pt) / σ_t
```

Onde:
- **μ_Pt** = P₀ + GMD × t  (curva condicional individual do animal)
- **σ_t** = σ_nascer + λ × (t / t_max)²  (variância crescente, σ_nascer=0.35, λ=1.5)

Faixas de alerta:

| Status | Condição | Ação |
|---|---|---|
| `[NORMAL]` | Z ≥ -1.0 | Monitoramento de rotina |
| `[ATENCAO]` | -2.0 < Z < -1.0 | Observação intensificada |
| `[CRITICO]` | Z ≤ -2.0 | Notificação imediata ao produtor |

### 3.3. Machine Learning (`src/main/ml/`)

**`trainer.py`** — Dois modelos treináveis via scikit-learn:

1. **Regressão Linear** (`modelo_crescimento.pkl`): prevê **Peso_Atual** em um dia t, tendo como features: `Peso_Nascer`, `Dias_Vida`, `Sexo`, `Tipo_Parto`, `Ordem_Parto`.
2. **Regressão Logística** (`modelo_mortalidade.pkl`): prevê probabilidade de óbito usando o desvio quadrático `(Peso_Nascer - 4.0)²` para capturar a forma em U.

**`predictor.py`** — Carrega os `.pkl` em memória para inferência em tempo real (base para integração futura com API REST).

### 3.4. Visualizações (`src/main/visualization/`)

7 gráficos exportados em `.png` (300 DPI) divididos por público:

**Para Zootecnistas:**
| Arquivo | Conteúdo |
|---|---|
| `curva_crescimento_percentis.png` | Corredor P5 / P50 / P95 do crescimento |
| `risco_neonatal_curva_u.png` | Taxa de mortalidade empírica por faixa de peso |
| `curva_logistica_mortalidade.png` | Curva contínua teórica (Cenário B do documento) |
| `zscore_temporal_rebanho.png` | Trajetória do Z-Score de 20 animais com faixas de alerta |

**Para o Setor Privado:**
| Arquivo | Conteúdo |
|---|---|
| `correlacao_peso_gmd_negocios.png` | Correlação P₀ × GMD com coeficiente r |
| `distribuicao_peso_90d.png` | Histograma do peso na desmama (média + P10 refugo) |
| `distribuicao_peso_por_tipo_parto.png` | Curvas KDE comparando Simples vs Gêmeo vs Trigêmeos |

---

## 4. Testes (15 passed)

```
test/verification/test_sistema_especialista.py   6 testes - formulas do dominio
test/verification/test_data_generator.py         1 teste  - integridade da simulacao
test/verification/test_ml.py                     2 testes - treinamento e backup pkl
test/verification/test_monitorador.py            6 testes - Z-Score e alertas
```

---

## 5. Como Executar

```powershell
# Ativar o ambiente virtual e configurar o PYTHONPATH
$env:PYTHONPATH="."

# Rodar o pipeline completo (simula dados, treina modelos, gera graficos)
.\.venv\Scripts\python.exe src\main\main.py

# Rodar a suite de testes
.\.venv\Scripts\pytest.exe test\verification\ -v
```
