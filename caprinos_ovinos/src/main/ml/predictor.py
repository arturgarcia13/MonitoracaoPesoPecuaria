import joblib
import os
import pandas as pd

class PreditivoModelos:
    """
    Responsabilidade: Carregar os modelos persistidos (pkl) e fazer predições
    sobre novos dados.
    """
    
    def __init__(self, caminho_backup: str = "deploy/modelos/"):
        self.caminho_backup = caminho_backup
        
        # Carregando modelos em memória se existirem
        path_crescimento = os.path.join(self.caminho_backup, 'modelo_crescimento.pkl')
        path_morte = os.path.join(self.caminho_backup, 'modelo_mortalidade.pkl')
        
        self.modelo_crescimento = joblib.load(path_crescimento) if os.path.exists(path_crescimento) else None
        self.modelo_morte = joblib.load(path_morte) if os.path.exists(path_morte) else None
        
    def prever_peso_futuro(self, peso_nascer: float, dias_vida: int, sexo: str, tipo_parto: str, ordem: str) -> float:
        """Faz predição do peso esperado do animal no tempo (t) em dias de vida."""
        if not self.modelo_crescimento:
            raise ValueError("Modelo de curva de crescimento não treinado/encontrado.")
            
        df = pd.DataFrame([{
            'Peso_Nascer': peso_nascer,
            'Dias_Vida': dias_vida,
            'Sexo': sexo,
            'Tipo_Parto': tipo_parto,
            'Ordem_Parto': ordem
        }])
        
        return self.modelo_crescimento.predict(df)[0]
        
    def prever_risco_mortalidade(self, peso_nascimento: float) -> float:
        """Prevê probabilidade (0 a 1) de morte."""
        if not self.modelo_morte:
            raise ValueError("Modelo de mortalidade não treinado/encontrado.")
            
        df = pd.DataFrame([{
            'Peso_Quad': (peso_nascimento - 4.0) ** 2
        }])
        
        # [:, 1] pega a probabilidade da classe 1 (Morte)
        prob = self.modelo_morte.predict_proba(df)[0][1]
        return prob
