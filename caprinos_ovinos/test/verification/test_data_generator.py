import pytest
from src.main.simulation.data_generator import GeradorDados

class TestGeradorDados:
    def setup_method(self):
        self.gerador = GeradorDados(n_animais=50)

    def test_deve_gerar_duas_tabelas_com_registros(self):
        # Act
        df_animais, df_pesagens = self.gerador.gerar_dados()
        
        # Assert
        assert len(df_animais) == 50
        assert "ID_Animal" in df_animais.columns
        assert "Peso_Nascer" in df_animais.columns
        assert len(df_pesagens) >= 50 # Pelo menos 1 pesagem (dia 0) por animal
