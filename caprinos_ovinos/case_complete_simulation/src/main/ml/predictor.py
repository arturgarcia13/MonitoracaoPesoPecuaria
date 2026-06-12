import joblib
import os
import pandas as pd


class PreditivoModelos:
    """
    Responsabilidade: Carregar os modelos persistidos e executar o pipeline
    de inferencia sequencial conforme o fluxo definido no documento regressao.md.

    Fluxo de uso correto (passo a passo do documento):

        PASSO 1 — Estimar P0:
            Dado um novo animal (sexo, tipo de parto, ordem da mae),
            o Modelo 1 estima o P0 esperado.

        PASSO 2 — Projetar crescimento:
            Com o P0 estimado (ou real, se ja aferido) e o dia t desejado,
            o Modelo 2 projeta o Peso_Atual esperado (Pt).

        PASSO 3 — Avaliar risco neonatal:
            Com o P0 (real ou estimado), o Modelo 3 calcula P(Y=1),
            a probabilidade de obito neonatal.

    Esta classe encapsula os tres modelos e garante que o fluxo seja
    executado na ordem correta.
    """

    def __init__(self, caminho_backup: str = "deploy/modelos/"):
        self.caminho_backup = caminho_backup

        path_p0 = os.path.join(caminho_backup, "modelo_equacao1_peso_nascer.pkl")
        path_pt = os.path.join(caminho_backup, "modelo_equacao2_trajetoria_peso.pkl")
        path_risco = os.path.join(caminho_backup, "modelo_equacao3_risco_neonatal.pkl")

        self._modelo_p0 = joblib.load(path_p0) if os.path.exists(path_p0) else None
        self._modelo_pt = joblib.load(path_pt) if os.path.exists(path_pt) else None
        self._modelo_risco = joblib.load(path_risco) if os.path.exists(path_risco) else None

    # ─── PASSO 1 ─────────────────────────────────────────────────────────────

    def estimar_peso_nascer(self, sexo: str, tipo_parto: str, ordem_parto: str) -> float:
        """
        PASSO 1 — Equacao P0 (Secao 3.1).

        Estima o peso esperado ao nascer a partir dos efeitos fixos de parto.
        Usar quando o animal ainda nao nasceu e queremos prever o P0.

        Args:
            sexo: "M" ou "F"
            tipo_parto: "Simples", "Gemeo" ou "Trigemeos"
            ordem_parto: "Primipara" ou "Multipara"

        Returns:
            Peso estimado ao nascer (kg).
        """
        self._verificar_modelo(self._modelo_p0, "Equacao 1 (P0)")
        df = pd.DataFrame([{
            "Sexo": sexo,
            "Tipo_Parto": tipo_parto,
            "Ordem_Parto": ordem_parto,
        }])
        return float(self._modelo_p0.predict(df)[0])

    # ─── PASSO 2 ─────────────────────────────────────────────────────────────

    def projetar_peso_em_t(self, peso_nascer: float, dias_vida: int, tipo_parto: str = "Simples") -> float:
        """
        PASSO 2 — Equacao Pt (Secao 3.2).

        Projeta o peso esperado do animal em um determinado dia t,
        usando o P0 e os fatores que compoem o GMD (tipo_parto).

        Args:
            peso_nascer: P0 do animal (real ou estimado).
            dias_vida: Dia t de interesse (0 a 90).
            tipo_parto: "Simples", "Gemeo" ou "Trigemeos". Necessario para penalidade do GMD.

        Returns:
            Peso projetado no dia t (kg).
        """
        self._verificar_modelo(self._modelo_pt, "Equacao 2 (Pt)")
        
        is_multiplo = 1.0 if tipo_parto in ["Gemeo", "Trigemeos"] else 0.0
        
        df = pd.DataFrame([{
            "Peso_Nascer": peso_nascer,
            "Dias_Vida": dias_vida,
            "GMD_P0_t": (peso_nascer - 4.0) * dias_vida,
            "GMD_Multiplo_t": is_multiplo * dias_vida
        }])
        return float(self._modelo_pt.predict(df)[0])

    # ─── PASSO 3 ─────────────────────────────────────────────────────────────

    def avaliar_risco_neonatal(self, peso_nascer: float) -> float:
        """
        PASSO 3 — Risco Neonatal Logistico (Secao 3.3, Cenario B).

        Calcula a probabilidade de obito neonatal P(Y=1) dado o P0.
        Deve ser executado logo apos o nascimento, antes das projecoes de crescimento.

        Args:
            peso_nascer: P0 real do animal (kg).

        Returns:
            Probabilidade de obito entre 0.0 e 1.0.
        """
        self._verificar_modelo(self._modelo_risco, "Equacao 3 (Risco)")
        desvio_quad = (peso_nascer - 4.0) ** 2
        df = pd.DataFrame([{"Desvio_Quad_P0": desvio_quad}])
        return float(self._modelo_risco.predict_proba(df)[0][1])

    # ─── Pipeline completo ───────────────────────────────────────────────────

    def executar_pipeline_completo(
        self,
        sexo: str,
        tipo_parto: str,
        ordem_parto: str,
        peso_nascer_real: float = None,
        dias_projecao: int = 90,
    ) -> dict:
        """
        Executa o fluxo completo do documento em sequencia:
            Passo 1 -> Passo 2 -> Passo 3

        Se peso_nascer_real for fornecido, usa o valor real para P0.
        Caso contrario, usa a estimativa do Modelo 1.

        Returns:
            Dicionario com P0_estimado, P0_usado, Pt_projetado e prob_obito.
        """
        p0_estimado = self.estimar_peso_nascer(sexo, tipo_parto, ordem_parto)
        p0_usado = peso_nascer_real if peso_nascer_real is not None else p0_estimado

        pt_projetado = self.projetar_peso_em_t(p0_usado, dias_projecao, tipo_parto)
        prob_obito = self.avaliar_risco_neonatal(p0_usado)

        return {
            "p0_estimado_modelo": round(p0_estimado, 3),
            "p0_usado": round(p0_usado, 3),
            "pt_projetado_dia_{}".format(dias_projecao): round(pt_projetado, 3),
            "prob_obito": round(prob_obito, 4),
        }

    # ─── Auxiliar ────────────────────────────────────────────────────────────

    def _verificar_modelo(self, modelo, nome: str):
        if modelo is None:
            raise ValueError(
                f"Modelo '{nome}' nao encontrado em '{self.caminho_backup}'. "
                "Execute o treinamento primeiro via TreinadorModelos."
            )
