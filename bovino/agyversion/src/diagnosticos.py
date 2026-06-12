import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuração de estilo visual "Business" (Limpo, Cores Sóbrias, Direto ao Ponto)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("muted")

def gerar_graficos_apresentacao(caminho_dados_pred, caminho_saida):
    df = pd.read_csv(caminho_dados_pred)
    os.makedirs(caminho_saida, exist_ok=True)
    
    # 1. Gráfico de Diagnóstico Técnico (Para relatório/professor)
    # Resíduos Estudentizados vs Fitted Values (Homocedasticidade)
    plt.figure(figsize=(8, 6))
    plt.scatter(df['predicao_lucro'], df['residuos_estudentizados'], alpha=0.5, color='gray')
    plt.axhline(0, color='red', linestyle='--')
    plt.axhline(3, color='orange', linestyle=':')
    plt.axhline(-3, color='orange', linestyle=':')
    plt.title('Diagnóstico de Homocedasticidade: Resíduos vs Valores Ajustados', fontsize=12)
    plt.xlabel('Predição de Lucro (Valores Ajustados)')
    plt.ylabel('Resíduos Estudentizados Externos')
    plt.tight_layout()
    plt.savefig(f"{caminho_saida}/diagnostico_residuos.png", dpi=300)
    plt.close()

    # 2. Gráfico para o Pitch Executivo (Setor Privado) - Impacto do Ganho Pré-desmama no Lucro
    # "Clean, Business, 1 Mensagem"
    plt.figure(figsize=(8, 6))
    sns.regplot(x='gpd_pre_desmama', y='lucro', data=df, scatter_kws={'alpha':0.3, 'color':'#2b5b84'}, line_kws={'color':'#e74c3c', 'linewidth':3})
    plt.title('Eficácia Pré-Desmama dita a Lucratividade Final', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Ganho de Peso Diário Pré-Desmama (Kg/Dia)', fontsize=12)
    plt.ylabel('Lucro Projetado por Animal (R$)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{caminho_saida}/pitch_impacto_desmama.png", dpi=300)
    plt.close()

    # 3. Identificação de Animais em Risco de Prejuízo (Alavancagem de Decisão)
    # Mostra a relação Peso a Desmama vs Lucratividade, destacando machos e fêmeas
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='peso_desmama', y='lucro', hue='sexo_M', data=df, palette=['#34495e', '#e67e22'], alpha=0.7)
    plt.axhline(0, color='red', linestyle='-', linewidth=2, label='Limiar de Prejuízo')
    plt.title('Viabilidade Financeira vs Peso na Desmama', fontsize=16, fontweight='bold')
    plt.xlabel('Peso ao Desmamar (Kg)', fontsize=12)
    plt.ylabel('Lucro (R$)', fontsize=12)
    plt.legend(title='Sexo (1=Macho, 0=Fêmea)')
    plt.tight_layout()
    plt.savefig(f"{caminho_saida}/pitch_viabilidade_desmama.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    caminho_dados_pred = 'E:/MYAREA/AREA_DEV/Faculdade/ModelosRegressaoI/MonitaracaoPesoBovinoWorkflow/v1/outputs/dados_com_predicoes.csv'
    caminho_saida = 'E:/MYAREA/AREA_DEV/Faculdade/ModelosRegressaoI/MonitaracaoPesoBovinoWorkflow/v1/outputs'
    gerar_graficos_apresentacao(caminho_dados_pred, caminho_saida)
    print("Gráficos gerados e salvos para o pitch.")
