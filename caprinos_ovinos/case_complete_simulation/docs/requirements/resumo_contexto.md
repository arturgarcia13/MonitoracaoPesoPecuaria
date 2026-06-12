# Resumo de Contexto - Modelagem Biométrica (Caprinos/Ovinos)

**Data da Refatoração:** Junho de 2026
**Objetivo:** Alinhar o código-fonte estritamente com a fundamentação teórica de `regressao.md`.

## 1. O Problema Identificado
O pipeline de ML original misturava features de nascimento e projeção de crescimento em um único modelo, violando a regra de negócios (biológica) que define processos em equações separadas.

## 2. Ações de Correção Realizadas
- **`trainer.py` e `predictor.py`:** Refatorados para refletir exatamente os 3 modelos do documento:
  - **Equação 1 ($P_0$):** Estima o peso ao nascer baseado em Efeitos Fixos (Sexo, Tipo de Parto, Ordem).
  - **Equação 2 ($P_t$):** Projeta a trajetória de crescimento usando $P_0$ e $t$ (Dias de Vida) como âncoras.
  - **Equação 3 (Risco Neonatal):** Avalia probabilidade de óbito neonatal usando a curva em U centrada no $P_{opt}$ de 4.0 kg.
- **Testes (`test_ml.py`):** Suíte atualizada para testar as equações em etapas separadas (23 de 23 testes passando).
- **Diagnósticos:** Criado o `diagnostico_modelos.py` para gerar painéis de validação estatística (R², RMSE, Curva ROC) documentando a performance de cada modelo isoladamente.

## 3. Validação Teórica
Confirmou-se que a premissa fundamental matemática $\eta \sim N(0, 0.5^2)$ da Equação 1 está sendo rigorosamente gerada e respeitada pela biblioteca `numpy` via `np.random.normal(0, 0.5)` em `data_generator.py` (linha 38), e subsequentemente modelada pelo OLS (`LinearRegression`).

## 4. Próximos Passos (Backlog Restante)
O arcabouço central está 100% aderente ao documento. As etapas futuras do projeto incluem integração com frontend e estruturação de pipelines de retreino.
