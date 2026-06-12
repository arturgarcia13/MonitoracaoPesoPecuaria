import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score


class TreinadorModelos:
    """
    Responsabilidade: Treinar e persistir os dois modelos de regressao do documento
    regressao.md, que representam etapas sequenciais do processo biologico:

    ETAPA 1 — Equacao P0 (Secao 3.1):
        P0 = 4.10 + beta_parto + beta_sexo + beta_matriz + eta
        Features: Sexo, Tipo_Parto, Ordem_Parto
        Target  : Peso_Nascer

    ETAPA 2 — Equacao Pt (Secao 3.2):
        P_ti = P0i + (GMD_i * t) + eta_t
        Features: Peso_Nascer (P0), Dias_Vida (t)
        Target  : Peso_Atual

    ETAPA 3 — Risco Neonatal (Secao 3.3 — Cenario B):
        z = alpha0 + alpha1 * (P0 - P_opt)^2   [logistica]
        Features: (Peso_Nascer - 4.0)^2
        Target  : Y_Morto (0/1)

    Nota: O fluxo sequencial E1 -> E2 reflete o passo a passo biologico real:
    1. Estimamos o P0 esperado pelas caracteristicas do parto.
    2. Usamos esse P0 como ancora para projetar a curva de crescimento.
    3. Avaliamos o risco de obito antes de qualquer projecao de crescimento.
    """

    NOME_MODELO_P0 = "modelo_equacao1_peso_nascer.pkl"
    NOME_MODELO_PT = "modelo_equacao2_trajetoria_peso.pkl"
    NOME_MODELO_RISCO = "modelo_equacao3_risco_neonatal.pkl"

    def __init__(self, caminho_backup: str = "deploy/modelos/"):
        self.caminho_backup = caminho_backup
        os.makedirs(self.caminho_backup, exist_ok=True)

    # ─── ETAPA 1: Equacao P0 ─────────────────────────────────────────────────

    def treinar_equacao_p0(self, df_animais: pd.DataFrame) -> Pipeline:
        """
        Treina o Modelo 1 — Equacao do Peso ao Nascer (P0).

        Aprende os coeficientes beta a partir dos efeitos fixos categoricos.
        Target: Peso_Nascer (variavel continua).
        Features: Sexo, Tipo_Parto, Ordem_Parto.

        Equacao do documento (Secao 3.1):
            P0 = 4.10 + beta_parto + beta_sexo + beta_matriz + eta,  eta ~ N(0, 0.52)
        """
        X = df_animais[["Sexo", "Tipo_Parto", "Ordem_Parto"]]
        y = df_animais["Peso_Nascer"]

        preprocessador = ColumnTransformer(transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False),
             ["Sexo", "Tipo_Parto", "Ordem_Parto"]),
        ])

        pipeline = Pipeline(steps=[
            ("preprocessador", preprocessador),
            ("regressor", LinearRegression()),
        ])
        pipeline.fit(X, y)

        metricas = self._avaliar_regressao(pipeline, X, y, "Equacao 1 — P0 (Peso ao Nascer)")
        joblib.dump(pipeline, os.path.join(self.caminho_backup, self.NOME_MODELO_P0))
        return pipeline

    # ─── ETAPA 2: Equacao Pt ─────────────────────────────────────────────────

    def treinar_equacao_pt(self, df_animais: pd.DataFrame, df_pesagens: pd.DataFrame) -> Pipeline:
        """
        Treina o Modelo 2 — Equacao da Trajetoria Longitudinal do Peso (Pt).

        Usa P0 como ancora e t (dias de vida) para projetar o crescimento.
        Equacao do documento (Secao 3.2):
            P_ti = P0i + (GMD_i * t) + eta_t
        Onde (Secao 3.1):
            GMD_i = GMD_base + gamma*(P0i - 4.0) - Penalidade(Tipo_Parto)

        Para que a Regressao Linear aprenda o GMD_i individual corretamente,
        fornecemos as interacoes de `t` com P0 e Tipo_Parto.
        """
        # Juncao para ter P0 e Tipo_Parto de cada animal
        df = pd.merge(df_pesagens, df_animais[["ID_Animal", "Peso_Nascer", "Tipo_Parto"]], on="ID_Animal")

        # Feature Engenharia: Criamos as interacoes explicitas com 't' (Dias_Vida)
        # O P0 é o intercepto individual real (peso no dia 0).
        df_model = df.copy()
        
        # 1. Indicador de parto multiplo
        df_model["Is_Multiplo"] = df_model["Tipo_Parto"].isin(["Gemeo", "Trigemeos"]).astype(float)
        
        # 2. Interações com o tempo (t) - compõem o GMD_i
        df_model["GMD_P0_t"] = (df_model["Peso_Nascer"] - 4.0) * df_model["Dias_Vida"]
        df_model["GMD_Multiplo_t"] = df_model["Is_Multiplo"] * df_model["Dias_Vida"]

        # Features finais
        X = df_model[["Peso_Nascer", "Dias_Vida", "GMD_P0_t", "GMD_Multiplo_t"]]
        y = df_model["Peso_Atual"]

        pipeline = Pipeline(steps=[
            ("regressor", LinearRegression())
        ])
        pipeline.fit(X, y)

        metricas = self._avaliar_regressao(pipeline, X, y, "Equacao 2 — Pt (Trajetoria de Peso)")
        joblib.dump(pipeline, os.path.join(self.caminho_backup, self.NOME_MODELO_PT))
        return pipeline

    # ─── ETAPA 3: Risco Neonatal ─────────────────────────────────────────────

    def treinar_equacao_risco(self, df_animais: pd.DataFrame) -> Pipeline:
        """
        Treina o Modelo 3 — Risco Neonatal Logistico (Cenario B, Secao 3.3).

        Captura a curva em U de mortalidade: tanto baixo quanto alto peso ao nascer
        aumentam o risco de obito (hipotermia/inacao vs distocia).

        Equacao do documento:
            z = alpha0 + alpha1 * (P0 - P_opt)^2
            P(Y=1) = sigmoid(z)
        """
        df_model = df_animais.copy()
        # Feature engenharia: distancia quadratica do peso otimo (4.0 kg)
        df_model["Desvio_Quad_P0"] = (df_model["Peso_Nascer"] - 4.0) ** 2

        X = df_model[["Desvio_Quad_P0"]]
        y = df_model["Y_Morto"]

        pipeline = Pipeline(steps=[
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=500)),
        ])
        pipeline.fit(X, y)

        y_pred = pipeline.predict(X)
        y_prob = pipeline.predict_proba(X)[:, 1]
        acc = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_prob)
        print(f"  [Equacao 3 — Risco Neonatal] Accuracy: {acc:.4f} | AUC-ROC: {auc:.4f}")

        joblib.dump(pipeline, os.path.join(self.caminho_backup, self.NOME_MODELO_RISCO))
        return pipeline

    # ─── Auxiliares ──────────────────────────────────────────────────────────

    def _avaliar_regressao(self, pipeline: Pipeline, X, y, nome: str) -> dict:
        """Calcula e imprime MSE e R2 para modelos de regressao."""
        y_pred = pipeline.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mse)
        print(f"  [{nome}] RMSE: {rmse:.4f} | R2: {r2:.4f}")
        return {"mse": mse, "rmse": rmse, "r2": r2}
