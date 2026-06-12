# Tutorial de Uso

## Visão Geral
Este projeto simula dados de crescimento bovino e ajusta um Modelo de Regressão Linear Múltipla (MRLM) para prever o lucro projetado de cada animal com base no seu desempenho pré-desmama e características de nascimento. A execução deste sistema gera relatórios estatísticos e gráficos voltados a um **Pitch Executivo para o Setor Privado**.

## Requisitos de Ambiente
- Python 3.8+
- Pacotes exigidos:
  - `pandas`
  - `numpy`
  - `statsmodels`
  - `matplotlib`
  - `seaborn`

Instalação de dependências:
```bash
pip install pandas numpy statsmodels matplotlib seaborn
```

## Como Executar
1. Navegue até o diretório `v1/src/`:
   ```bash
   cd E:/MYAREA/AREA_DEV/Faculdade/ModelosRegressaoI/MonitaracaoPesoBovinoWorkflow/v1/src
   ```
2. Execute o orquestrador principal:
   ```bash
   python main.py
   ```

## Estrutura de Saída (`v1/outputs`)
Ao finalizar, o pipeline gera:
- `dados_com_predicoes.csv`: Os dados simulados enriquecidos com resíduos estudentizados, predição do modelo, Distância de Cook e Leverages.
- `diagnostico_estatistico.json`: Sumário programático dos testes Durbin-Watson, Jarque-Bera e fatores VIF.
- `sumario_modelo.txt`: Tabela completa OLS do `statsmodels`.
- `pitch_impacto_desmama.png` e `pitch_viabilidade_desmama.png`: Gráficos limpos criados para a apresentação de negócio.
- `diagnostico_residuos.png`: Gráfico técnico exigido pelo professor para comprovar homocedasticidade.
- `animais_em_risco.csv`: Base de alerta com os IDs que apresentam projeção de lucro negativo (potencial descarte).
