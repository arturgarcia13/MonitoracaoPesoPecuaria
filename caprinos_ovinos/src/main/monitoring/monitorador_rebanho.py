from dataclasses import dataclass
from enum import Enum
from src.main.domain.calculadora_biometrica import CalculadoraBiometrica


class StatusAlerta(Enum):
    """
    Enum para os tres estados de alerta definidos no documento regressao.md.
    Secao 1 - Arquitetura de Dados / Regra de Negocio e Disparo de Alertas.
    """
    NORMAL  = "[NORMAL]  Z >= -1.0 : Animal operando dentro ou acima do esperado"
    ATENCAO = "[ATENCAO] -2 < Z < -1: Desaceleracao do crescimento"
    CRITICO = "[CRITICO] Z <= -2.0  : Mais de 2dp abaixo do perfil biologico"


@dataclass
class ResultadoMonitoramento:
    """
    Contém o resultado completo de uma avaliação de pesagem individual.
    Segue o padrão de 'objeto de valor' (Value Object) do Domain-Driven Design.
    """
    id_animal: int
    dias_vida: int
    peso_real: float
    peso_esperado: float   # μ_Pt — média condicional da curva individual
    desvio_padrao_t: float # σ_t — variância residual no instante t
    z_score: float
    status: StatusAlerta
    interpretacao: str


class MonitoradorRebanho:
    """
    Responsabilidade: Calcular o Z-Score individual de cada animal em relação
    à sua curva de crescimento esperada condicional, gerando alertas estatísticos.

    Fórmula (regressao.md, Seção 1):
        Z_atual = (P_t_real - μ_Pt) / σ_t

    Faixas de alerta:
        🟢 Normal    → Z >= -1.0
        🟡 Atenção   → -2.0 < Z < -1.0
        🔴 Crítico   → Z <= -2.0
    """

    # Parâmetros da variância residual (regressao.md, Seção 3.2)
    SIGMA_NASCER: float = 0.35
    LAMBDA_ESCALA: float = 1.5
    T_MAX: float = 90.0

    def __init__(self):
        self._calculadora = CalculadoraBiometrica()

    def avaliar_pesagem(
        self,
        id_animal: int,
        peso_real: float,
        dias_vida: int,
        sexo: str,
        tipo_parto: str,
        ordem_parto: str,
        peso_nascer: float,
    ) -> ResultadoMonitoramento:
        """
        Avalia a pesagem de um animal individual, retornando o Z-Score e o alerta.

        Args:
            id_animal: Identificador único do animal.
            peso_real: Peso aferido no curral (kg).
            dias_vida: Idade do animal em dias (0 a 90).
            sexo: "M" ou "F".
            tipo_parto: "Simples", "Gemeo" ou "Trigemeos".
            ordem_parto: "Primipara" ou "Multipara".
            peso_nascer: Peso ao nascer (P0), usado como âncora da curva.

        Returns:
            ResultadoMonitoramento com Z-Score e status de alerta.
        """
        mu_pt = self._calcular_peso_esperado_em_t(peso_nascer, dias_vida, tipo_parto)
        sigma_t = self._calcular_sigma_t(dias_vida)
        z_score = (peso_real - mu_pt) / sigma_t

        status = self._classificar_alerta(z_score)
        interpretacao = self._gerar_interpretacao(status, z_score, peso_real, mu_pt)

        return ResultadoMonitoramento(
            id_animal=id_animal,
            dias_vida=dias_vida,
            peso_real=peso_real,
            peso_esperado=mu_pt,
            desvio_padrao_t=sigma_t,
            z_score=z_score,
            status=status,
            interpretacao=interpretacao,
        )

    def _calcular_peso_esperado_em_t(
        self, peso_nascer: float, dias_vida: int, tipo_parto: str
    ) -> float:
        """Calcula μ_Pt: o peso esperado individual em um dado dia t."""
        gmd = self._calculadora.calcular_gmd_esperado(peso_nascer, tipo_parto)
        return peso_nascer + (gmd * dias_vida)

    def _calcular_sigma_t(self, dias_vida: int) -> float:
        """
        Calcula σ_t: variância residual crescente ao longo do tempo.
        Fórmula (regressao.md, Seção 3.2):
            σ_t = σ_nascer + λ * (t / t_max)²
        """
        t_normalizado = dias_vida / self.T_MAX
        return self.SIGMA_NASCER + self.LAMBDA_ESCALA * (t_normalizado ** 2)

    def _classificar_alerta(self, z_score: float) -> StatusAlerta:
        """Classifica o Z-Score nas três faixas de alerta do documento."""
        if z_score <= -2.0:
            return StatusAlerta.CRITICO
        elif z_score < -1.0:
            return StatusAlerta.ATENCAO
        return StatusAlerta.NORMAL

    def _gerar_interpretacao(
        self,
        status: StatusAlerta,
        z_score: float,
        peso_real: float,
        peso_esperado: float,
    ) -> str:
        """Gera texto interpretativo para o produtor."""
        diferenca = peso_real - peso_esperado
        sinal = "acima" if diferenca >= 0 else "abaixo"
        detalhe = f"Peso real: {peso_real:.2f}kg | Esperado: {peso_esperado:.2f}kg | Z={z_score:.2f}"

        if status == StatusAlerta.CRITICO:
            return (
                f"[CRÍTICO] Animal {abs(diferenca):.2f}kg {sinal} do esperado. "
                f"Notificação imediata recomendada. {detalhe}"
            )
        elif status == StatusAlerta.ATENCAO:
            return (
                f"[ATENÇÃO] Animal {abs(diferenca):.2f}kg {sinal} do esperado. "
                f"Monitorar na próxima semana. {detalhe}"
            )
        return (
            f"[NORMAL] Animal dentro da faixa esperada. "
            f"{detalhe}"
        )
