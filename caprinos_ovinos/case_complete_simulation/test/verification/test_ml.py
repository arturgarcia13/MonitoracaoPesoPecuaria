import pytest
import os
import math
import pandas as pd
from src.main.ml.trainer import TreinadorModelos
from src.main.ml.predictor import PreditivoModelos

CAMINHO_TEST = "test_models_v2/"


def _criar_df_animais() -> pd.DataFrame:
    """Cria um DataFrame minimo de animais para testes."""
    return pd.DataFrame({
        "ID_Animal": [1, 2, 3, 4, 5, 6],
        "Sexo": ["M", "F", "M", "F", "M", "F"],
        "Tipo_Parto": ["Simples", "Gemeo", "Simples", "Trigemeos", "Simples", "Gemeo"],
        "Ordem_Parto": ["Multipara", "Primipara", "Multipara", "Multipara", "Primipara", "Multipara"],
        "Peso_Nascer": [4.10, 2.80, 4.05, 2.30, 3.75, 3.40],
        "GMD_Real": [0.272, 0.234, 0.271, 0.210, 0.265, 0.255],
        "Y_Morto": [0, 1, 0, 1, 0, 0],
    })


def _criar_df_pesagens() -> pd.DataFrame:
    """Cria um DataFrame minimo de pesagens longitudinais para testes."""
    registros = []
    dias = [0, 15, 30, 45, 60, 75, 90]
    for id_animal, peso_nascer, gmd in [(1, 4.10, 0.272), (3, 4.05, 0.271), (5, 3.75, 0.265), (6, 3.40, 0.255)]:
        for t in dias:
            registros.append({
                "ID_Animal": id_animal,
                "Dias_Vida": t,
                "Peso_Atual": peso_nascer + gmd * t,
            })
    return pd.DataFrame(registros)


class TestTreinadorEquacaoP0:
    """
    Testa o Modelo 1 — Equacao P0 (Secao 3.1 do documento).
    Verifica que o treinamento aprende os efeitos fixos beta e persiste o .pkl.
    """

    def setup_method(self):
        self.treinador = TreinadorModelos(caminho_backup=CAMINHO_TEST)
        self.df_animais = _criar_df_animais()

    def test_deve_treinar_e_salvar_modelo_p0(self):
        # Act
        pipeline = self.treinador.treinar_equacao_p0(self.df_animais)

        # Assert
        assert pipeline is not None
        assert os.path.exists(os.path.join(CAMINHO_TEST, TreinadorModelos.NOME_MODELO_P0))

    def test_modelo_p0_deve_prever_peso_numerico_positivo(self):
        # Arrange
        pipeline = self.treinador.treinar_equacao_p0(self.df_animais)

        # Act
        df_novo = pd.DataFrame([{"Sexo": "M", "Tipo_Parto": "Simples", "Ordem_Parto": "Multipara"}])
        pred = pipeline.predict(df_novo)[0]

        # Assert
        assert pred > 0
        assert isinstance(float(pred), float)


class TestTreinadorEquacaoPt:
    """
    Testa o Modelo 2 — Equacao Pt (Secao 3.2 do documento).
    Verifica que P0 e t (Dias_Vida) sao as unicas features e que o modelo e salvo.
    """

    def setup_method(self):
        self.treinador = TreinadorModelos(caminho_backup=CAMINHO_TEST)
        self.df_animais = _criar_df_animais()
        self.df_pesagens = _criar_df_pesagens()

    def test_deve_treinar_e_salvar_modelo_pt(self):
        # Act
        pipeline = self.treinador.treinar_equacao_pt(self.df_animais, self.df_pesagens)

        # Assert
        assert pipeline is not None
        assert os.path.exists(os.path.join(CAMINHO_TEST, TreinadorModelos.NOME_MODELO_PT))

    def test_modelo_pt_deve_crescer_com_o_tempo(self):
        # Arrange: animal com P0 fixo, verificar que Pt cresce quando t aumenta
        pipeline = self.treinador.treinar_equacao_pt(self.df_animais, self.df_pesagens)

        pred_t0 = pipeline.predict(pd.DataFrame([{
            "Peso_Nascer": 4.0, "Dias_Vida": 0, "GMD_P0_t": 0.0, "GMD_Multiplo_t": 0.0
        }]))[0]
        pred_t90 = pipeline.predict(pd.DataFrame([{
            "Peso_Nascer": 4.0, "Dias_Vida": 90, "GMD_P0_t": 0.0, "GMD_Multiplo_t": 0.0
        }]))[0]

        # Assert: Pt deve ser maior que P0 em t=90
        assert pred_t90 > pred_t0


class TestTreinadorEquacaoRisco:
    """
    Testa o Modelo 3 — Risco Neonatal (Secao 3.3, Cenario B do documento).
    """

    def setup_method(self):
        self.treinador = TreinadorModelos(caminho_backup=CAMINHO_TEST)
        self.df_animais = _criar_df_animais()

    def test_deve_treinar_e_salvar_modelo_risco(self):
        # Act
        pipeline = self.treinador.treinar_equacao_risco(self.df_animais)

        # Assert
        assert pipeline is not None
        assert os.path.exists(os.path.join(CAMINHO_TEST, TreinadorModelos.NOME_MODELO_RISCO))


class TestPipelineSequencial:
    """
    Testa o PreditivoModelos: verifica que o fluxo Passo 1 -> 2 -> 3
    funciona de ponta a ponta apos o treinamento.
    """

    def setup_method(self):
        treinador = TreinadorModelos(caminho_backup=CAMINHO_TEST)
        treinador.treinar_equacao_p0(_criar_df_animais())
        treinador.treinar_equacao_pt(_criar_df_animais(), _criar_df_pesagens())
        treinador.treinar_equacao_risco(_criar_df_animais())
        self.preditivo = PreditivoModelos(caminho_backup=CAMINHO_TEST)

    def test_passo1_deve_estimar_peso_nascer(self):
        peso = self.preditivo.estimar_peso_nascer("M", "Simples", "Multipara")
        assert peso > 0.0

    def test_passo2_deve_projetar_peso_futuro_maior_que_p0(self):
        p0 = 4.10
        pt = self.preditivo.projetar_peso_em_t(p0, 90)
        assert pt > p0

    def test_passo3_deve_retornar_probabilidade_valida(self):
        prob = self.preditivo.avaliar_risco_neonatal(4.0)
        assert 0.0 <= prob <= 1.0

    def test_pipeline_completo_deve_retornar_dicionario_com_todos_campos(self):
        resultado = self.preditivo.executar_pipeline_completo(
            sexo="M", tipo_parto="Simples", ordem_parto="Multipara",
            peso_nascer_real=4.1, dias_projecao=90
        )
        assert "p0_estimado_modelo" in resultado
        assert "p0_usado" in resultado
        assert "pt_projetado_dia_90" in resultado
        assert "prob_obito" in resultado

    def test_pipeline_com_p0_real_deve_usar_valor_real(self):
        p0_real = 3.5
        resultado = self.preditivo.executar_pipeline_completo(
            sexo="F", tipo_parto="Gemeo", ordem_parto="Primipara",
            peso_nascer_real=p0_real
        )
        assert resultado["p0_usado"] == p0_real
