import os
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, accuracy_score

class TreinadorModelos:
    """
    Responsabilidade: Treinar modelos de Machine Learning para recuperar/aproximar 
    as regras do sistema especialista. Gera um modelo para Peso ao Nascer e outro 
    para predição de Mortalidade.
    """
    
    def __init__(self, caminho_backup: str = "deploy/modelos/"):
        self.caminho_backup = caminho_backup
        if not os.path.exists(self.caminho_backup):
            os.makedirs(self.caminho_backup)
            
    def treinar_modelo_crescimento(self, df_animais: pd.DataFrame, df_pesagens: pd.DataFrame) -> Pipeline:
        """Treina um modelo linear para prever peso em um tempo t baseado no peso ao nascer e dias de vida."""
        # Integrando as tabelas
        df = pd.merge(df_pesagens, df_animais, on="ID_Animal", how="inner")
        
        # Features e Target: Peso_Nascer agora é feature, Peso_Atual é target.
        X = df[['Peso_Nascer', 'Dias_Vida', 'Sexo', 'Tipo_Parto', 'Ordem_Parto']]
        y = df['Peso_Atual']
        
        # Pipeline: OneHotEncoding para categóricas + Regressao Linear
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first', sparse_output=False), ['Sexo', 'Tipo_Parto', 'Ordem_Parto']),
                ('num', 'passthrough', ['Peso_Nascer', 'Dias_Vida'])
            ])
            
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
        
        pipeline.fit(X, y)
        
        # Opcional: Avaliar
        y_pred = pipeline.predict(X)
        mse = mean_squared_error(y, y_pred)
        print(f"[Treinamento] MSE Curva de Crescimento (Peso_Atual): {mse:.4f}")
        
        # Salvar
        joblib.dump(pipeline, os.path.join(self.caminho_backup, 'modelo_crescimento.pkl'))
        return pipeline

    def treinar_modelo_mortalidade(self, df: pd.DataFrame) -> Pipeline:
        """Treina um modelo de Regressao Logistica para mortalidade com base no Z-score do peso."""
        # Vamos treinar diretamente no Peso_Nascer como feature e Y_Morto como target.
        # Mas para capturar a curva em 'U' (peso muito baixo ou muito alto mata),
        # precisaremos adicionar feature polinomial ou usar o desvio quadratico do peso otimo.
        
        df_model = df.copy()
        # Criando feature customizada: desvio quadratico do peso otimo (4.0)
        df_model['Peso_Quad'] = (df_model['Peso_Nascer'] - 4.0) ** 2
        
        X = df_model[['Peso_Quad']]
        y = df_model['Y_Morto']
        
        pipeline = Pipeline(steps=[
            ('classifier', LogisticRegression(class_weight='balanced'))
        ])
        
        pipeline.fit(X, y)
        
        y_pred = pipeline.predict(X)
        acc = accuracy_score(y, y_pred)
        print(f"[Treinamento] Accuracy Mortalidade: {acc:.4f}")
        
        joblib.dump(pipeline, os.path.join(self.caminho_backup, 'modelo_mortalidade.pkl'))
        return pipeline
