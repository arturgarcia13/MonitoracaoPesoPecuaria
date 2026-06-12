class CalculadoraBiometrica:
    """
    Responsabilidade: Encapsular as fórmulas matemáticas do sistema especialista
    para cálculo do peso ao nascimento e GMD.
    """
    
    PESO_BASE = 4.10
    
    def calcular_peso_esperado(self, sexo: str, tipo_parto: str, ordem_parto: str) -> float:
        """Calcula a expectativa de peso com base nos efeitos fixos."""
        peso = self.PESO_BASE
        peso += self._obter_fator_sexo(sexo)
        peso += self._obter_fator_parto(tipo_parto)
        peso += self._obter_fator_ordem_parto(ordem_parto)
        return peso
        
    def _obter_fator_sexo(self, sexo: str) -> float:
        """Obtém coeficiente para sexo."""
        if sexo.upper() == "F":
            return -0.30
        return 0.0
        
    def _obter_fator_parto(self, tipo_parto: str) -> float:
        """Obtém coeficiente para tipo de parto."""
        tipo_upper = tipo_parto.upper()
        if tipo_upper == "GEMEO":
            return -0.65
        elif tipo_upper == "TRIGEMEOS":
            return -1.40
        return 0.0
        
    def _obter_fator_ordem_parto(self, ordem: str) -> float:
        """Obtém coeficiente para idade da matriz."""
        if ordem.upper() == "PRIMIPARA":
            return -0.35
        return 0.0

    def calcular_gmd_esperado(self, peso_nascer: float, tipo_parto: str) -> float:
        """Calcula o Ganho Médio Diário baseado no peso de nascimento para Ovinos."""
        gmd_base = 0.272  # Valor empírico médio para Ovinos de corte (ex: Dorper)
        delta_peso = peso_nascer - 4.0
        gmd = gmd_base + (0.02 * delta_peso)
        
        if tipo_parto.upper() in ["GEMEO", "TRIGEMEOS"]:
            gmd -= 0.025
            
        return gmd
