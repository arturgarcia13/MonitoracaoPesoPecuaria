import pytest
from src.main.monitoring.monitorador_rebanho import MonitoradorRebanho, StatusAlerta


class TestMonitoradorRebanho:
    """
    Testes do sistema de monitoramento de Z-Score individual.
    Baseado na seção 'Regra de Negócio e Disparo de Alertas' do documento.
    """

    def setup_method(self):
        self.monitor = MonitoradorRebanho()

    def test_deve_retornar_status_normal_quando_z_acima_de_menos_um(self):
        # Arrange: animal com peso ideal (esperado=esperado → Z=0)
        # Arrange
        id_animal = 1
        peso_nascer = 4.10
        dias_vida = 30
        # GMD esperado = 0.272 + 0.02*(4.10-4.0) = 0.274
        # mu_t = 4.10 + 0.274*30 = 12.32
        peso_real = 12.32  # peso exatamente no esperado → Z=0

        # Act
        resultado = self.monitor.avaliar_pesagem(
            id_animal, peso_real, dias_vida, "M", "Simples", "Multipara", peso_nascer
        )

        # Assert
        assert resultado.status == StatusAlerta.NORMAL
        assert resultado.z_score >= -1.0

    def test_deve_retornar_status_atencao_quando_z_entre_menos_dois_e_menos_um(self):
        # Arrange: animal com desvio moderado
        id_animal = 2
        peso_nascer = 4.10
        dias_vida = 60
        # sigma_60 = 0.35 + 1.5*(60/90)^2 = 0.35 + 1.5*0.444 = 1.016
        # mu_60 = 4.10 + 0.274*60 = 20.54
        # Para Z=-1.5: peso_real = 20.54 - 1.5*1.016 = 20.54 - 1.524 = 19.016
        peso_real = 19.02

        # Act
        resultado = self.monitor.avaliar_pesagem(
            id_animal, peso_real, dias_vida, "M", "Simples", "Multipara", peso_nascer
        )

        # Assert
        assert resultado.status == StatusAlerta.ATENCAO
        assert -2.0 < resultado.z_score < -1.0

    def test_deve_retornar_status_critico_quando_z_abaixo_de_menos_dois(self):
        # Arrange: animal muito abaixo do esperado
        id_animal = 3
        peso_nascer = 4.10
        dias_vida = 90
        # sigma_90 = 0.35 + 1.5*(90/90)^2 = 0.35 + 1.5 = 1.85
        # mu_90 = 4.10 + 0.272*90 = 28.58 (usando peso_nascer=4.10, delta=0.1, gmd=0.274)
        # Para Z=-3.0: peso_real = mu_90 - 3*sigma_90
        sigma_90 = 0.35 + 1.5 * (1.0 ** 2)
        gmd = 0.272 + 0.02 * (4.10 - 4.0)  # 0.274
        mu_90 = 4.10 + gmd * 90
        peso_real = mu_90 - 3.0 * sigma_90  # Z=-3 garantido

        # Act
        resultado = self.monitor.avaliar_pesagem(
            id_animal, peso_real, dias_vida, "M", "Simples", "Multipara", peso_nascer
        )

        # Assert
        assert resultado.status == StatusAlerta.CRITICO
        assert resultado.z_score <= -2.0

    def test_deve_calcular_sigma_t_crescente_com_o_tempo(self):
        # Arrange & Act
        sigma_0 = self.monitor._calcular_sigma_t(0)
        sigma_45 = self.monitor._calcular_sigma_t(45)
        sigma_90 = self.monitor._calcular_sigma_t(90)

        # Assert: σ deve crescer com o tempo (heterogeneidade residual)
        assert sigma_0 < sigma_45 < sigma_90

    def test_deve_calcular_sigma_nascimento_igual_a_constante(self):
        # Assert: ao nascer, σ_t = σ_nascer = 0.35
        sigma_0 = self.monitor._calcular_sigma_t(0)
        assert abs(sigma_0 - 0.35) < 1e-10

    def test_resultado_contem_campos_obrigatorios(self):
        # Arrange
        resultado = self.monitor.avaliar_pesagem(
            1, 10.0, 30, "F", "Gemeo", "Primipara", 3.5
        )

        # Assert: todos os campos do dataclass presentes
        assert hasattr(resultado, "id_animal")
        assert hasattr(resultado, "z_score")
        assert hasattr(resultado, "status")
        assert hasattr(resultado, "interpretacao")
        assert len(resultado.interpretacao) > 0
