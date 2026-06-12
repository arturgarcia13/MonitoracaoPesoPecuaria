# Documento Técnico: Modelagem Biométrica e Arquitetura de Monitoramento

3.1 Apresentação de Variáveis e Justificativas Teóricas 

O objetivo desta seção é detalhar como cada variável biológica está incorporada no modelo matemático , sustentando a importância de sua inclusão e os valores de seus coeficientes a partir de revisões da literatura científica de ovinocultura e caprinocultura.

Peso ao Nascer ($P_0$) 

O peso ao nascer é modelado como uma função linear de efeitos fixos (tipo de parto, sexo e matriz) acrescido de um erro aleatório gaussiano:

$$P_0 = 4.10 + \beta_{\text{parto}} + \beta_{\text{sexo}} + \beta_{\text{matriz}} + \eta$$

Dado que:


$$\eta \sim N(0, 0.5^2)$$

---

1. Intercepto Base ($\beta_0 = 4.10\text{ kg}$) 

O valor de referência representa o peso esperado para um animal que seja **Macho, oriundo de Parto Simples e de uma Mãe Multípara**.

#### Referências Base:

* **Cordeiro (Ovinos):**
* FREITAS et al. (1980): Em estudo com 330 cordeiros Morada Nova (linhagem Branca III), observou-se peso médio ao nascer de $3,1\text{ kg}$.


* *MCMILLAN et al. (1983):* Na Nova Zelândia, identificou-se que o peso de nascimento é um fator crítico para a sobrevivência. O intervalo ótimo para garantir a mínima mortalidade foi estimado entre $3,3\text{ kg}$ e $4,1\text{ kg}$.


* S. HATCHER (2009): Relatou média geral de $3,63\text{ kg}$, apontando que cordeiros de parto simples alcançaram média de $4,00 \pm 0.01\text{ kg}$.




* **Cabritos (Caprinos):**
* A média geral de peso ao nascimento identificada para todas as crias avaliadas foi de $3,96\text{ kg}$.





---

2. Efeito de Sexo ($\beta_{\text{sexo}}$) 

* 
**Abordagem no Modelo:** $\beta_{\text{sexo}} = -0.30\text{ kg}$ se o animal for **fêmea** (calculado através da média consensual das literaturas).



#### Referências Base:

* **Ovinos:**
* EVERTS et al. (1985): Identificou que o sexo influencia o peso ao nascer, sendo as fêmeas $0,19\text{ kg}$ mais leves que os machos.


* 
*GARDNER et al.:* Apontou média de $4,92\text{ kg}$ para machos e $4,57\text{ kg}$ para fêmeas (uma diferença de $0,363\text{ kg}$ a favor dos machos).




* **Caprinos:**
* Dados experimentais registram cabritos machos com média de $4,12\text{ kg}$ e fêmeas com $3,80\text{ kg}$ (machos $8,4\%$ ou $0,32\text{ kg}$ mais pesados).





---

3. Efeito do Tipo de Parto ($\beta_{\text{parto}}$) 

* 
**Abordagem no Modelo:** $\beta_{\text{parto}} = -0.65\text{ kg}$ se **gêmeo**; e $\beta_{\text{parto}} = -1.40\text{ kg}$ se **trigêmeos** (tendo o parto simples como base zero).



#### Referências Base:

* **Ovinos:**
* FREITAS et al. (1980): Partos simples geraram cordeiros com média de $3,4\text{ kg}$, enquanto partos gemelares resultaram em $2,7\text{ kg}$ (diferença de $0,70\text{ kg}$).


* NUNES et al. (1980): Pesos de $2,92 \pm 0.66\text{ kg}$ (simples) contra $2,32 \pm 0.59\text{ kg}$ (múltiplos), gerando uma diferença de $0,60\text{ kg}$.


* S. HATCHER (2009): Simples ($4,00\text{ kg}$), gêmeos ($3,35\text{ kg}$) e múltiplos ($2,90\text{ kg}$). Diferença de $0,65\text{ kg}$ (simples vs. gêmeos) e $1,10\text{ kg}$ (simples vs. múltiplos).


* 
*GARDNER et al.:* Apresentou decréscimo progressivo: simples ($5,47\text{ kg}$), gêmeos ($-0,692\text{ kg}$), trigêmeos ($-1,40\text{ kg}$) e quadrigêmeos ($-2,08\text{ kg}$).




* **Caprinos:**
* Partos simples registraram média de $4,17\text{ kg}$ e partos duplos $3,75\text{ kg}$ (diferença de $0,42\text{ kg}$).





---

4. Idade/Ordem de Parto da Matriz ($\beta_{\text{matriz}}$) 

* 
**Abordagem no Modelo:** $\beta_{\text{matriz}} = -0.35\text{ kg}$ se a mãe for **primípara**.



#### Referências Base:

* O peso médio dos cordeiros cresce da 1ª até a 4ª gestação, declinando posteriormente. O maior incremento biológico ocorre da primeira para a segunda gestação, onde os cordeiros nascem, em média, $0,351\text{ kg}$ mais pesados.



---

Ganho Médio Diário (GMD) e Crescimento Longitudinal 

O peso ponderal ao longo do tempo $t$ é calculado agregando o Ganho Médio Diário ao peso inicial:

$$P_t = P_0 + \text{GMD} \cdot t$$

Onde o $\text{GMD}$ está intrinsecamente correlacionado ao peso de nascimento:

$$\text{GMD} = \text{GMD}_{\text{base}} + \gamma(P_0 - 4.0)$$

Parâmetros estipulados:

* 
$\gamma \approx 0.02$ 


* **$\text{GMD}_{\text{base}}$ por Aptidão Comercial:**
* Ovinos de corte puros (ex: Dorper/Santa Inês): $0.252$ a $0.292\text{ kg/dia}$.


* Caprinos comerciais (Mestiços Anglo): $0.140$ a $0.153\text{ kg/dia}$.




* 
**Penalização por Tipo de Parto:** Subtrair $0.025\text{ kg/dia}$ do $\text{GMD}$ se a cria for de parto múltiplo.



> 
> **Justificativa Biológica da Correlação:** Estudos estatísticos (Medeiros et al.; Santos et al.) demonstram uma correlação positiva forte de $0,832$ entre o peso ao nascer e o peso à desmama , e de $0,847$ com o ganho de peso diário. Animais com restrição uterina severa nascem com menor quantidade de fibras musculares (miongênese limitada) e menor capacidade de ingestão de colostro/leite.
> 
> 

---

3.2 O Modelo Completo e Heterogeneidade Residual 

Para evitar alarmes falsos em idades jovens e simular o comportamento biológico real (onde a dispersão do peso aumenta com a idade), a variância residual não é tratada como homogênea. Substitui-se o erro fixo por uma variância residual dependente do tempo $t$:

$$P_{ti} = P_{0i} + (\text{GMD}_i \cdot t) + \eta_t$$

$$\eta_t \sim N(0, \sigma_t)$$

A modelagem do desvio padrão residual ($\sigma_t$) é descrita pela função quadrática suave abaixo:

Onde:

* 
$\sigma_{\text{nascer}} = 0.35$ (variabilidade padrão no nascimento).


* 
$t_{\text{max}} = 90\text{ dias}$ (idade limite estabelecida na desmama).


* 
$\lambda = 1.5$ (fator de escala biométrica).



---

3.3 Exemplos de Aplicação Prática dos Modelos 

Cenário A: Simulação de Ganho Ponderal Dinâmico 

Tomando como base um caprino mestiço ($\text{GMD}_{\text{base}} = 0.140\text{ kg/dia}$):

1. 
**Animal Forte ($P_0 = 4.5\text{ kg}$):** 


$$\text{GMD} = 0.140 + 0.02 \cdot (4.5 - 4.0) = 0.150\text{ kg/dia}$$





2. 
**Animal Fraco ($P_0 = 2.5\text{ kg}$):** 


$$\text{GMD} = 0.140 + 0.02 \cdot (2.5 - 4.0) = 0.110\text{ kg/dia}$$






Aos 60 dias de vida, a divergência de curvas ampliará drasticamente o distanciamento entre os animais.

Cenário B: Predição Epidemiológica de Risco Neonatal 

A probabilidade de mortalidade é calculada a partir de uma função logística baseada no Score $Z$ de distanciamento do peso ótimo ($P_{\text{opt}} = 4.0\text{ kg}$):

$$z = \alpha_0 + \alpha_1(P_0 - P_{\text{opt}})^2$$

Adotando os parâmetros calibrados ($\alpha_0 = -2,5$ e $\alpha_1 = 1,2$):

* 
**Peso Ideal ($P_0 = 4.0\text{ kg}$):** $z = -2.5 \rightarrow P(Y=1) \approx \mathbf{7,5\%}$ (Mortalidade Mínima).


* 
**Baixo Peso ($P_0 = 2.0\text{ kg}$):** $z = 2.3 \rightarrow P(Y=1) \approx \mathbf{90,9\%}$ (Hipotermia/Inanição).


* 
**Sobrepeso/Macrossomia ($P_0 = 5.5\text{ kg}$):** $z = 0.2 \rightarrow P(Y=1) \approx \mathbf{55,0\%}$ (Distocia/Trauma).



> A relação resulta em uma **Curva em U**, penalizando tanto a desnutrição fetal quanto os partos distócicos por fetos excessivamente grandes.
> 
> 

---

1. Arquitetura de Dados e Componentes da Solução 

Para sustentação de mercado e escalabilidade em propriedades privadas, o sistema é projetado sob a seguinte engenharia de software:

```
[Aplicativo Mobile (Offline Coleta)] 
         │ (Sincronização Postgres)
         ▼
[Banco de Dados Relacional SQL] ──► [Motor de Inferência (Python API)]
   - Tabela Animais                     - FastAPI / Flask
   - Tabela Pesagens                    - Z-Score Dinâmico & XAI

```

### Componentes de Infraestrutura:

1. 
**Motor de Inferência (API em Python):** Modelos expostos via microserviços em FastAPI ou Flask para responder a rota de predição em tempo de coleta.


2. 
**Banco de Dados Longitudinal:** Estrutura relacional clássica (PostgreSQL) dividida em:


* 
`Animais`: `ID`, `Sexo`, `Tipo_Parto`, `Ordem_Parto_Matriz`, `Data_Nascimento`, `Peso_Nascer`.


* 
`Pesagens`: `ID_Animal`, `Data_Pesagem`, `Dias_de_Vida (t)`, `Peso_Atual`.





### Regra de Negócio e Disparo de Alertas Estatísticos:

Ao invés de limites estáticos, o sistema calcula o Z-score do peso real contra a curva condicional individual calculada para as características do animal ($\mu_{P_t}$):

$$Z_{\text{atual}} = \frac{P_{t\_{\text{real}}} - \mu_{P_t}}{\sigma_t}$$

* 🟢 **Status Normal ($Z \ge -1.0$):** Animal operando dentro ou acima da faixa esperada.


* 🟡 **Atenção ($-2.0 < Z < -1.0$):** Alerta visual. Animal cruzou o percentil inferior e desacelerou o crescimento.


* 🔴 **Crítico ($Z \le -2.0$):** Notificação *Push* imediata ao produtor. O peso está mais de 2 desvios padrões abaixo do limiar do perfil biológico.



---

## Requisitos de Implementação do Sistema (Backlog)

* [ ] **Módulo Coleta Offline:** Permitir digitação de biometrias em currais sem conectividade com a internet, aplicando sincronização assíncrona posterior com o PostgreSQL.


* [ ] **Pipeline de Retreino Automatizado:** Criação de rotinas (via Apache Airflow ou Cron jobs) a cada 3 ou 6 meses para reestimar os parâmetros fixos e logísticos com os dados históricos da própria fazenda.


* [ ] **Tratamento de Dados de Morte (Truncamento Causal):** Ativação da flag `Y_morto == 1` para mover automaticamente o registro para a aba "Histórico/Óbitos", removendo o animal das projeções ativas e cobranças de metas.