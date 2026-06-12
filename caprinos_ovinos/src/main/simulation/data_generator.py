import pandas as pd
import numpy as np
from src.main.domain.calculadora_biometrica import CalculadoraBiometrica
from src.main.domain.avaliador_risco import AvaliadorRiscoNeonatal

class GeradorDados:
    """
    Responsabilidade: Simular o banco de dados longitudinal de animais e 
    suas pesagens ao longo do tempo (0 a 90 dias).
    """

    def __init__(self, n_animais: int = 500):
        self.n_animais = n_animais
        self.calculadora = CalculadoraBiometrica()
        self.avaliador = AvaliadorRiscoNeonatal()
        
    def gerar_dados(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Gera as tabelas de Animais e Pesagens."""
        df_animais = self._gerar_animais()
        df_pesagens = self._gerar_pesagens(df_animais)
        return df_animais, df_pesagens

    def _gerar_animais(self) -> pd.DataFrame:
        np.random.seed(42)
        ids = np.arange(1, self.n_animais + 1)
        # Fixando espécie apenas para Ovinos
        especies = ["Ovino"] * self.n_animais
        sexos = np.random.choice(["M", "F"], size=self.n_animais, p=[0.48, 0.52])
        tipos_parto = np.random.choice(["Simples", "Gemeo", "Trigemeos"], size=self.n_animais, p=[0.70, 0.25, 0.05])
        ordens = np.random.choice(["Primipara", "Multipara"], size=self.n_animais, p=[0.30, 0.70])
        
        pesos_nascer = []
        gmds = []
        mortos = []
        
        for i in range(self.n_animais):
            peso_base = self.calculadora.calcular_peso_esperado(sexos[i], tipos_parto[i], ordens[i])
            peso_real = peso_base + np.random.normal(0, 0.5) # erro aleatorio de nascimento
            
            # Truncando valores absurdos
            peso_real = max(1.0, peso_real)
            pesos_nascer.append(peso_real)
            
            gmd = self.calculadora.calcular_gmd_esperado(peso_real, tipos_parto[i])
            gmds.append(gmd)
            
            prob_morte = self.avaliador.calcular_probabilidade_obito(peso_real)
            morte = 1 if np.random.random() < prob_morte else 0
            mortos.append(morte)
            
        df = pd.DataFrame({
            "ID_Animal": ids,
            "Especie": especies,
            "Sexo": sexos,
            "Tipo_Parto": tipos_parto,
            "Ordem_Parto": ordens,
            "Peso_Nascer": pesos_nascer,
            "GMD_Real": gmds,
            "Y_Morto": mortos
        })
        return df

    def _gerar_pesagens(self, df_animais: pd.DataFrame) -> pd.DataFrame:
        pesagens = []
        # Dias de pesagem: 0, 15, 30, 45, 60, 75, 90
        dias_coleta = [0, 15, 30, 45, 60, 75, 90]
        
        sigma_nascer = 0.35
        lambda_escala = 1.5
        t_max = 90
        
        for _, animal in df_animais.iterrows():
            # Se morreu, assumimos que morreu nos primeiros 5 dias (nao tem pesagens futuras)
            dias = [0] if animal["Y_Morto"] == 1 else dias_coleta
            
            for t in dias:
                sigma_t = sigma_nascer + lambda_escala * ((t / t_max)**2)
                eta_t = np.random.normal(0, sigma_t) if t > 0 else 0
                
                peso_atual = animal["Peso_Nascer"] + (animal["GMD_Real"] * t) + eta_t
                peso_atual = max(peso_atual, 0.5) # Segurança lógica
                
                pesagens.append({
                    "ID_Animal": animal["ID_Animal"],
                    "Dias_Vida": t,
                    "Peso_Atual": peso_atual
                })
                
        return pd.DataFrame(pesagens)
