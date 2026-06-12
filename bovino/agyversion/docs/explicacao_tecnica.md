# Explicação Técnica da Solução

## 1. Simulação Genético-Quantitativa
Baseado nas abordagens de *Test-Day Models* e Modelos Mistos Animais (`Y = Xb + Za + Mm + Wp + e`), a simulação computacional modela fatores biológicos:
- **Genética Direta ($a$) e Materna ($m$)**: O peso ao nascer e ganho pré-desmama dependem da genética materna e da capacidade própria, refletindo-se em taxas diárias.
- **Tendência Genética**: Animais mais jovens tendem a performar melhor devido à evolução genética dos rebanhos sobre os anos.

## 2. Ajuste Estatístico Rigoroso (MQO)
A estimação dos parâmetros de regressão seguiu estritamente as Mínimos Quadrados Ordinários (MQO) adotando-se o teorema de Gauss-Markov. O código `treinamento_modelo.py` foca no diagnóstico estrutural e não apenas no valor de $R^2$:
- **Testes de Hipóteses e Distribuições**: Ao treinar o modelo `OLS` via `statsmodels`, validamos os resíduos através do Teste de Jarque-Bera (normalidade) e Durbin-Watson (autocorrelação).
- **Detecção de Influência (Alavancagem e Masking)**: Calculamos as matrizes Hat-diags ($h_{ii}$) e a Distância de Cook. A "regra de ouro" de Cook $> 4/n$ é usada de forma sistêmica para detectar anomalias (potencial de mascaramento).
- **Prevenção à Multicolinearidade**: O cálculo do VIF (Variance Inflation Factor) está inserido nos relatórios em formato JSON, prevendo distorções de predição entre variáveis proximais (ex: Colostro vs Peso ao Nascer).

## 3. Direcionamento Visual (Comunicação)
A fim de atender os requisitos exigidos, abandonou-se a exibição gráfica excessivamente p-valor cêntrica e optou-se por gráficos de alta clareza de negócio.
- **Gráficos Executivos**: Mostram, numa leitura unifocal, como a eficácia do GPD (Ganho de Peso Diário) se reverte em Lucro e a respectiva margem de viabilidade por sexo e peso ao desmamar.
