import pytest
import os
import pandas as pd
from src.main.ml.trainer import TreinadorModelos

class TestTreinadorModelos:
    def setup_method(self):
        self.treinador = TreinadorModelos(caminho_backup="test_models/")

    def test_deve_treinar_modelo_crescimento_e_salvar_pkl(self):
        # Arrange
        df_animais = pd.DataFrame({
            "ID_Animal": [1, 2, 3],
            "Sexo": ["M", "F", "M"],
            "Tipo_Parto": ["Simples", "Gemeo", "Simples"],
            "Ordem_Parto": ["Multipara", "Primipara", "Multipara"],
            "Peso_Nascer": [4.10, 2.80, 4.05],
            "Y_Morto": [0, 1, 0]
        })
        
        df_pesagens = pd.DataFrame({
            "ID_Animal": [1, 1, 2, 3, 3],
            "Dias_Vida": [0, 15, 0, 0, 30],
            "Peso_Atual": [4.10, 8.5, 2.80, 4.05, 12.0]
        })
        
        # Act
        pipeline = self.treinador.treinar_modelo_crescimento(df_animais, df_pesagens)
        
        # Assert
        assert pipeline is not None
        assert os.path.exists("test_models/modelo_crescimento.pkl")

    def test_deve_treinar_modelo_mortalidade_e_salvar_pkl(self):
        # Arrange
        df_fake = pd.DataFrame({
            "Peso_Nascer": [4.0, 2.0, 6.0, 4.1],
            "Y_Morto": [0, 1, 1, 0]
        })
        
        # Act
        pipeline = self.treinador.treinar_modelo_mortalidade(df_fake)
        
        # Assert
        assert pipeline is not None
        assert os.path.exists("test_models/modelo_mortalidade.pkl")
