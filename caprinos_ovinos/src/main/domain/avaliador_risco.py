import math

class AvaliadorRiscoNeonatal:
    """
    Responsabilidade: Avaliar a probabilidade de mortalidade neonatal
    baseado no peso ao nascer usando curva logística (Z-Score).
    """
    
    PESO_OTIMO = 4.0
    ALPHA_0 = -2.5
    ALPHA_1 = 1.2
    
    def calcular_probabilidade_obito(self, peso_nascimento: float) -> float:
        """Calcula P(Y=1) baseado no afastamento do peso ideal."""
        z_score = self._calcular_z_score(peso_nascimento)
        return self._funcao_logistica(z_score)
        
    def _calcular_z_score(self, peso_nascimento: float) -> float:
        """Aplica a penalidade quadrática para variação do peso ótimo."""
        delta = peso_nascimento - self.PESO_OTIMO
        return self.ALPHA_0 + (self.ALPHA_1 * (delta ** 2))
        
    def _funcao_logistica(self, z_score: float) -> float:
        """Aplica a transformação sigmoide no Z-score."""
        return 1.0 / (1.0 + math.exp(-z_score))
