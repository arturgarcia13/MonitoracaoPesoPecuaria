import pytest
import math

# Importações relativas simuladas - as classes reais serão implementadas em src.main.domain
from src.main.domain.calculadora_biometrica import CalculadoraBiometrica
from src.main.domain.avaliador_risco import AvaliadorRiscoNeonatal

class TestCalculadoraBiometrica:
    def setup_method(self):
        # Arrange global para os testes
        self.calculadora = CalculadoraBiometrica()

    def test_deve_calcular_peso_nascimento_macho_simples_multipara(self):
        # Arrange
        sexo = "M"
        tipo_parto = "Simples"
        ordem_parto = "Multipara"
        
        # Act
        peso = self.calculadora.calcular_peso_esperado(sexo, tipo_parto, ordem_parto)
        
        # Assert
        assert math.isclose(peso, 4.10, rel_tol=1e-5)

    def test_deve_calcular_peso_nascimento_femea_gemeo_primipara(self):
        # Arrange
        sexo = "F" # -0.30
        tipo_parto = "Gemeo" # -0.65
        ordem_parto = "Primipara" # -0.35
        # Total esperado: 4.10 - 0.30 - 0.65 - 0.35 = 2.80
        
        # Act
        peso = self.calculadora.calcular_peso_esperado(sexo, tipo_parto, ordem_parto)
        
        # Assert
        assert math.isclose(peso, 2.80, rel_tol=1e-5)

    def test_deve_calcular_gmd_ovino_simples_peso_base(self):
        # Arrange
        peso_nascimento = 4.0
        tipo_parto = "Simples"
        
        # Act
        gmd = self.calculadora.calcular_gmd_esperado(peso_nascimento, tipo_parto)
        
        # Assert (0.272 + 0 - 0)
        assert math.isclose(gmd, 0.272, rel_tol=1e-5)

    def test_deve_calcular_gmd_ovino_gemeo_baixo_peso(self):
        # Arrange
        peso_nascimento = 3.0 # delta = -1.0 -> -0.02
        tipo_parto = "Gemeo" # delta = -0.025
        # Base ovino = 0.272
        # Total: 0.272 - 0.02 - 0.025 = 0.227
        
        # Act
        gmd = self.calculadora.calcular_gmd_esperado(peso_nascimento, tipo_parto)
        
        # Assert
        assert math.isclose(gmd, 0.227, rel_tol=1e-5)

class TestAvaliadorRiscoNeonatal:
    def setup_method(self):
        self.avaliador = AvaliadorRiscoNeonatal()

    def test_deve_calcular_mortalidade_peso_ideal(self):
        # Arrange
        peso_nascimento = 4.0
        # z = -2.5 + 1.2(0) = -2.5. Prob = 1 / (1 + exp(2.5)) = 0.0758
        
        # Act
        prob_morte = self.avaliador.calcular_probabilidade_obito(peso_nascimento)
        
        # Assert
        assert 0.07 < prob_morte < 0.08

    def test_deve_calcular_mortalidade_baixo_peso(self):
        # Arrange
        peso_nascimento = 2.0
        # z = -2.5 + 1.2(4) = 2.3. Prob = 1 / (1 + exp(-2.3)) = 0.9088
        
        # Act
        prob_morte = self.avaliador.calcular_probabilidade_obito(peso_nascimento)
        
        # Assert
        assert 0.90 < prob_morte < 0.92
