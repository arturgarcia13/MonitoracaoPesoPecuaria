# Decisões Tomadas com Base nos Relatórios

## 1. Do Especialista Acadêmico
- **Decisão**: A simulação matemática abandonou "randômicos simples" e abraçou o conceito de "Ganho Genético Aditivo" e "Habilidade Materna". Variáveis de "Ambiente pré e pós desmama" foram incluídas imitando um autêntico Modelo Misto (`Y = Xb + Za + Mm + e`).
- **Justificativa**: Em *1-s2.0-S0022030210003905-main.txt* e *2027-4297-recia-10-01-00068.txt*, os autores enfatizam que a correlação entre ganhos prematuros e precocidade só existe pela carga genética sobreposta à nutrição.

## 2. Do Especialista da Disciplina
- **Decisão**: Substituímos os treinamentos "Scikit-Learn" padrões por "Statsmodels" puro. Implementamos o cálculo obrigatório de Resíduos Estudentizados Externos, Distância de Cook, VIF e Teste de Jarque-Bera.
- **Justificativa**: Conforme os textos *8 Diagnostico e Avaliacao no MRLS.txt* e *10 Comparacao de Modelos.txt*, o professor rejeita análises que miram apenas $R^2$. O rigor das premissas via testes formais e diagnóstico de "alavancagem" e "mascaramento" são cruciais para a aprovação.

## 3. Do Analista de Requisitos (Comunicação e Pitch)
- **Decisão**: Adotamos como público-alvo o **Setor Privado**. Geramos gráficos focados inteiramente em impacto de Lucro ($R\$$) e tomada de decisão preditiva ("Sistema de Alertas" para descarte precoce).
- **Justificativa**: O texto *comunicacao_e_pitch.txt* determina a eliminação de ruído matemático nas apresentações e requer a tradução direta do modelo para o alvo definido. Não haverá p-valores nos eixos; os resíduos ficam nos anexos técnicos.
