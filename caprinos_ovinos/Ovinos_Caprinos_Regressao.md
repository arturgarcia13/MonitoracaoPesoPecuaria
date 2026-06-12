Estruturação do Modelo de Regressão e Sistema de Alertas

### 1. Pergunta de Negócio

"Como podemos prever antecipadamente o desenvolvimento ponderal e o risco de mortalidade neonatal de ovinos e caprinos, utilizando dados simples coletados logo ao nascimento (peso ao nascer, sexo, tipo de parto e ordem de parto da matriz), a fim de emitir alertas dinâmicos que otimizem o manejo nutricional, reduzam perdas e maximizem a rentabilidade da propriedade?"

### 2. Contextualização do Problema: O Impacto no Negócio

A viabilidade econômica da ovinocultura e caprinocultura depende diretamente do número de animais viáveis desmamados por matriz e do ganho de peso até o abate. Um dos maiores gargalos do setor é a alta taxa de mortalidade no período neonatal. Segundo Ameghino et al. (1984), a mortalidade de cordeiros é um dos principais fatores que reduz a produtividade e a competitividade da ovinocultura em nível mundial.

O peso ao nascer atua como o principal fator preditivo para a sobrevivência e o futuro crescimento do animal. Estudos demonstram que existe uma forte relação curvilínea entre o peso ao nascer e a sobrevivência: a mortalidade é mais alta nos extremos (pesos muito baixos ou excessivos) e é otimizada em valores intermediários. McMillan et al. (1983) indicam que o peso ótimo ao nascimento para garantir a mínima mortalidade situa-se entre 3,3 e 4,1 kg.

- O problema do baixo peso: Cordeiros ou cabritos leves ao nascer nascem com menores reservas de energia e possuem uma grande área de superfície corporal relativa, perdendo calor rapidamente. Estão fortemente predispostos ao complexo "inanição/exposição", que responde por grande parte dos óbitos nas primeiras 72 horas de vida.
    
- O problema do peso excessivo: Por outro lado, pesos ao nascer muito elevados (especialmente em partos simples) aumentam significativamente os riscos de distocia (partos difíceis), podendo levar tanto a cria quanto a matriz a óbito.
    

Além da questão vital, animais mais leves ao nascer tendem a continuar mais leves. Eles apresentam uma taxa de ganho de peso médio diário (GMD) substancialmente menor, resultando em desmame com baixo peso e atraso na idade de abate, o que eleva os custos de alimentação e ocupação da fazenda.

Impacto no Negócio: A implementação de um aplicativo com um motor estatístico não servirá apenas como um "caderno digital". Ao registrar os dados do parto, o sistema calculará o risco de óbito e a curva de crescimento projetada de forma personalizada para cada animal. Isso permite que a equipe de campo atue proativamente — fornecendo colostro extra, abrigo térmico ou direcionando os animais para o creep-feeding — reduzindo a mortalidade e garantindo que as metas de peso de abate sejam atingidas no prazo ideal.

---

### 3. Descrevem-se os dados e as variáveis envolvidas

#### 3.1 Apresentação das Variáveis

Variável: Peso ao Nascer

- Como é utilizada: Atua como ponto de partida da trajetória (Intercepto da Equação) e influencia o ganho de peso subsequente e o cálculo do risco de mortalidade logística.
    
- Referências e Coeficientes:
    

- Ovinos: FREITAS et al. (1980) apontam média de 3,1 kg (Morada Nova). S. Hatcher (2009) encontrou média de 3,63 kg (Merino), variando de 2,90 a 4,00 kg dependendo do tipo de parto. Gardner et al. apontam até 5,47 kg para partos simples.
    
- Caprinos: Medeiros et al. (2012) encontraram média geral de 3,96 kg para mestiços.
    

[ESPAÇO PARA TABELA 1] Inserir a "Tabela 1" do artigo de Medeiros et al. (2012) mostrando os pesos ao nascimento, à desmama e ao abate para ilustrar as diferenças numéricas reais.

Variável: Sexo

- Referências e Coeficientes:
    

- Ovinos: EVERTS et al. (1985) notam fêmeas 0,19 kg mais leves que machos. Gardner et al. mostram machos sendo 0,363 kg mais pesados.
    
- Caprinos: Medeiros et al. encontram diferença de 0,32 kg a favor dos machos (4,12 kg vs 3,80 kg).
    

Variável: Coeficiente do Tipo de Parto (Simples vs. Múltiplo)

- Referências e Coeficientes:
    

- Ovinos: Nunes et al. (1980) apontam diferença média de 0,60 kg a menos para nascimentos múltiplos. Hatcher (2009) encontrou penalização de -0,65 kg para gêmeos e -1,10 kg para trigêmeos em relação a partos simples. Gardner et al. observaram gêmeos 0,692 kg mais leves e trigêmeos 1,40 kg mais leves.
    
- Caprinos: Diferença de 0,42 kg a favor de partos simples.
    

Variável: Idade/Ordem de Parto da Matriz

- Referências e Coeficientes: O estudo de Gardner et al. (Ovinos) demonstrou que as crias do primeiro parto são as mais leves. O salto de peso do primeiro (mãe primípara) para o segundo parto é de, em média, 0,351 kg a mais.
    

Variável: Ganho Médio Diário (GMD)

- Referências e Coeficientes:
    

- Ovinos: Animais cruzados em terminação atingem de 0,252 a 0,292 kg/dia. Animais afetados por parto múltiplo extremo podem ganhar tão pouco quanto 0,076 kg/dia.
    
- Caprinos: A média na fase pré-desmama foi de 0,140 kg/dia, caindo para 0,119 kg/dia pós-desmama. Animais de parto simples ganharam 0,153 kg/dia, enquanto gêmeos ganharam 0,127 kg/dia.
    

[ESPAÇO PARA TABELA 2] Inserir a "Tabela 2" do artigo de Medeiros et al. (2012) ou "Tabela 3" do artigo de Gardner et al., que evidenciam o efeito direto das variáveis independentes (Sexo, Parto, Paridade) nos ganhos de peso.

#### 3.2 Mostrar modelo completo

Consolidação do Sistema de Equações Teóricas:

Equação 1: Peso ao Nascer Esperado (P0) P0 = 4.10 + βparto + βsexo + βmatriz + η (dado que η ∼ N(0, σ²_t))

- β0 (Intercepto Base): 4.10 kg (Macho, Parto Simples, Mãe Multípara).
    
- β1 (Tipo de Parto): -0.65 se gêmeo; -1.40 se trigêmeos.
    
- β2 (Sexo): -0.30 se fêmea.
    
- β3 (Matriz): -0.35 se primípara.
    

Equação 2: Trajetória de Peso Atual (Pt) Pt = P0 + GMD⋅t Onde o GMD é ajustado pelo peso ao nascer real do animal: GMD = GMDbase + γ(P0 − 4.0) onde γ ≈ 0.02

Equação 3: Risco de Mortalidade Neonatal (Y) z = α0 + α1(P0 − Popt)² P(Y=1) = 1 / (1 + e^-z)

- Popt = 4,0 kg (peso ótimo para mortalidade mínima).
    

[ESPAÇO PARA GRÁFICO 1] Inserir a "Figure 1" ou "Figure 2" do artigo de Hatcher et al. (2009) "Phenotypic aspects of lamb survival...". Este gráfico ilustra perfeitamente a Equação 3: a relação curvilínea (U invertido) entre o peso ao nascer e a sobrevivência.

#### 3.3 Exemplos para elucidar as escolhas feitas

- Cenário de Crescimento (Eq. 2): Se o GMD base for 0,140 kg/dia, um animal que nasce forte (4,5 kg) terá GMD ajustado para 0,150 kg/dia. Um animal que nasce fraco (2,5 kg) terá GMD de 0,110 kg/dia. Em 60 dias, a divergência de pesos criará o abismo observado nos currais reais.
    
- Cenário de Mortalidade (Eq. 3): Um animal de 4,0 kg resulta em risco basal de $\approx 7,5%$. Um animal de apenas 2,0 kg dispara a probabilidade de mortalidade para $\approx 90,9%$ (inanição/hipotermia). Um filhote de 5,5 kg sobe o risco para $\approx 55%$ (distocia).
    

#### 3.4 O Problema da Heterogeneidade Residual

A variância fenotípica não é constante ao longo da vida do animal. Em vez de somar um erro fixo $\eta \sim N(0, 0.5^2)$, o modelo utiliza uma Variância Residual Heterogênea em função do tempo ($\sigma_t$).

Por que isso é crítico para o negócio? O artigo Modelos de regressão aleatória na avaliação genética... (Sarmento et al., 2010) comprova que assumir variância residual constante (homogênea) gera sub e superestimativas graves de valores genéticos e variações de peso. Ao assumir variância heterogênea crescente (via polinômios ordinários FO6 ou de Legendre FL5):

1. Evitamos Falsos Alertas: O sistema "sabe" que uma variação de 500 gramas aos 5 dias de vida é alarmante, mas a mesma variação aos 90 dias de vida é estatisticamente normal.
    
2. Preparação Genética: Deixa os dados padronizados para rodar avaliações genéticas de reprodutores (GBLUP) no futuro pela propriedade.
    

#### 3.5 Arquitetura de Dados e Componentes da Solução

- Camada Back-End: Motor de inferência (Python/FastAPI) e Banco de Dados Relacional (PostgreSQL) fatiado em Animais e Pesagens.
    
- Bandas de Confiança (Z-Score Adaptativo): Alertas baseados em $Z = (P_{real} - \mu_{Pt}) / \sigma_t$. Se $Z \ge -1.0$ (Normal), $-2.0 < Z < -1.0$ (Atenção - Lote amarelo), $Z \le -2.0$ (Crítico - Alerta Push vermelho).
    
- Checklist Técnico: Modo offline, pipeline de retreino trimestral dos coeficientes locais e truncamento causal de mortalidade.
    

---

### 4. Apresentação dos Gráficos das Relações Identificadas

(Esta seção deverá ser populada com os gráficos gerados pela UI do software)

- Gráfico 1: Risco de Mortalidade Neonatal (Logístico): Uma curva exibindo a probabilidade de óbito plotada em relação aos pesos reais aferidos nas primeiras 24 horas na propriedade, validando o "U invertido" e a faixa ideal de peso local.
    
- Gráfico 2: Trajetórias de Crescimento (Bandas de Confiança): Gráfico de linha longitudinal. O eixo X representa os Dias de Vida e o Y o Peso Vivo. A "linha alvo" central deve ser envolta por um sombreamento translúcido em forma de cone horizontal (representando a variância heterogênea crescendo com a idade). Os pontos reais das pesagens devem aparecer sobrepostos, com alteração de cor caso cruzem o percentil de alerta (Z-score negativo).
    

### 5. Interpretação Direta dos Resultados e Efeitos das Variáveis

(Exemplo de como o sistema/relatório interpretará os dados automaticamente)

- Impacto do Parto Múltiplo: Os dados processados devem evidenciar estatisticamente a penalização imposta pelos partos gemelares. A interpretação apontará que: "Animais de parto duplo nasceram $\beta$ kg mais leves e apresentam curva de aceleração de peso inferior aos de parto simples. O efeito justifica-se pela restrição física intrauterina e competição por leite."
    
- Efeito da Primiparidade: "As matrizes de primeira cria entregaram cordeiros sistematicamente mais leves ao nascimento ($\Delta = -0,35$ kg). O fenômeno fisiológico ocorre devido à menor vascularização uterina da ovelha primípara."
    
- Avaliação do Risco Individual: "Os animais cujos pesos reais ao nascimento divergiram em mais de 1,5 kg da média ótima (4,0 kg) concentraram $80\%$  dos óbitos na primeira semana de operação, validando a alta acurácia do modelo logístico preditivo."
    

### 6. Conclusões e Implicações Práticas

Conclusão: A aplicação de modelos de regressão bem fundamentados permite transformar a simples pesagem de rotina em uma ferramenta de predição robusta. O modelo confirma que o peso ao nascer é a pedra angular da lucratividade zootécnica, afetando em efeito cascata tanto o risco de mortalidade precoce quanto o GMD até o abate. Ao implementar uma abordagem com variância heterogênea adaptativa ao tempo, o sistema modela com precisão o crescimento biológico, filtrando ruídos estatísticos e gerando escores de alerta precisos.

Implicações Práticas (I.A. Explicável no Campo):

- Ação Neonatal Imediata: Ao inserir o peso e dados de um cordeiro no aplicativo (ex: 1,9 kg, Gêmeo, Mãe Primípara), a tela exibirá um diagnóstico: "Alto Risco de Hipotermia/Inanição (Risco: 90%). Proceder com abrigamento e banco de colostro."
    
- Manejo Nutricional Dinâmico: Animais que apresentarem Z-score em constante queda no gráfico de trajetórias devem ser imediatamente diagnosticados pela interface para manejo corretivo. "Lote apartável para suplementação via creep-feeding devido ao déficit uterino diagnosticado".
    
- Melhoramento Genético: Com a estabilização do banco de dados longitudinal e o controle dos resíduos heterogêneos, a fazenda possuirá infraestrutura técnica pronta para rodar modelos animais de predição de valor genético (DEP), permitindo descartar matrizes e reprodutores que sistematicamente geram pesos incompatíveis com as faixas de viabilidade estipuladas pelo aplicativo.


Pasta com os artigos:

[Ovinos e Caprinos](https://drive.google.com/drive/folders/1oBWQwcDItHcRYsb51gf-powIouMBY6XT?usp=drive_link)