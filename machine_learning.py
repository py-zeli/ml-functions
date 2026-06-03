# -*- coding: utf-8 -*-
"""
Biblioteca Local de Machine Learning (machine_learning.py)
-----------------------------------------------------------
Esta biblioteca agrupa os modelos preditivos e algoritmos de ajuste (OLS, GD),
delegando primitivas estatísticas e matriciais para o módulo 'algebra.py'.
"""

import algebra as alg
import estatistica as est

# Alias de conveniência para compatibilidade com os scripts de teste
criar_matriz = alg.criar_matriz

# --- MODELO 1: REGRESSÃO LINEAR SIMPLES ---

class RegressaoLinear:
    """Modelo de Regressão Linear Simples."""
    def __init__(self):
        self.coef_angular = None
        self.coef_linear = None

    def fit(self, x: list[float], y: list[float], metodo="mqo", taxa_aprendizado=0.01, epocas=5000):
        """
        Ajusta o modelo calibrando os coeficientes.
        Métodos:
        - 'mqo': Mínimos Quadrados Ordinários (analítico).
        - 'gd' ou 'descendente': Gradiente Descendente (iterativo).
        """
        # Validação de Vetores
        if len(x) != len(y):
            raise ValueError("Os vetores x e y devem ter o mesmo tamanho.")

        # Escolha do método de calibração
        if metodo == "mqo":
            self.coef_angular, self.coef_linear = minimos_quadrados(x, y)
        elif metodo in ("gd", "descendente"):
            coef_l = 0.0
            coef_a = 0.0
            for _ in range(epocas):
                coef_a, coef_l = gradient_descent_passo(x, y, taxa_aprendizado, coef_a, coef_l, modelo="linear")
            self.coef_linear = coef_l
            self.coef_angular = coef_a
        else:
            raise ValueError("Método desconhecido. Escolha entre 'mqo' ou 'gd' (descendente).")
            
        return self

    def predict(self, x: list[float]) -> list[float]:
        """Gera previsões para novos dados de entrada."""
        if self.coef_angular is None or self.coef_linear is None:
            raise RuntimeError("O modelo precisa ser treinado com .fit() antes de prever.")
            
        return realizar_previsao(x, self.coef_angular, self.coef_linear)


# --- MODELO 2: REGRESSÃO LINEAR MÚLTIPLA ---

class RegressaoLinearMultipla:
    """Modelo de Regressão Linear Múltipla via Equação Normal matricial."""
    def __init__(self):
        self.coeficientes = None

    def fit(self, X: list[list[float]], y: list[float]):
        """
        Ajusta o modelo resolvendo a Equação Normal matricial: beta = (X^T * X)^-1 * X^T * y.
        X: Matriz de design na representação de lista de colunas (incluindo vetor de intercepto).
        y: Vetor com a variável alvo.
        """
        # X é a matriz transposta de design na convenção usual
        X_transposta = X
        X_normal = alg.transpor_matriz(X_transposta)
        
        # 1. Covariância: X^T * X
        XtX = alg.multiplicar_matrizes(X_normal, X_transposta)
        
        # 2. Inversão de XtX
        inversa = alg.inverter_matriz(XtX)
        
        # 3. Produto de X^T por y
        Xt_y = alg.multiplicar_matriz_vetor(X_transposta, y)
        
        # 4. Cálculo final dos coeficientes (beta)
        self.coeficientes = alg.multiplicar_matriz_vetor(inversa, Xt_y)
        return self

    def predict(self, X: list[list[float]]) -> list[float]:
        """Gera previsões para novos dados de entrada na mesma representação matricial."""
        if self.coeficientes is None:
            raise RuntimeError("O modelo precisa ser treinado com .fit() antes de prever.")
        
        X_normal = alg.transpor_matriz(X)
        return alg.multiplicar_matriz_vetor(X_normal, self.coeficientes)


# --- MODELO 3: REGRESSÃO LOGÍSTICA ---

class RegressaoLogistica:
    """Modelo de Regressão Logística Binária."""
    def __init__(self):
        self.peso = None
        self.bias = None

    def fit(self, X: list[float], y: list[float], taxa_aprendizado=0.1, epocas=1000):
        """Ajusta o modelo calibrando o peso e o bias via Gradiente Descendente."""
        self.peso, self.bias = treinar_logistica(X, y, taxa_aprendizado, epocas)
        return self

    def predict_proba(self, X: list[float]) -> list[float]:
        """Prediz a probabilidade de pertencer à classe 1 para cada amostra."""
        if self.peso is None or self.bias is None:
            raise RuntimeError("O modelo precisa ser treinado com .fit() antes de prever.")
        return realizar_previsao(X, self.peso, self.bias, modelo="logistica")

    def predict(self, X: list[float], limiar=0.5) -> list[int]:
        """Classifica as amostras em 0 ou 1 com base no limiar de probabilidade."""
        probas = self.predict_proba(X)
        return [1 if p >= limiar else 0 for p in probas]


# --- FUNÇÕES DE AJUSTE E CÁLCULO ---

def minimos_quadrados(x: list[float], y: list[float]) -> tuple[float, float]:
    """
    Ajusta o modelo de Regressão Linear Simples via Mínimos Quadrados.
    Fórmula do Coeficiente Angular: Covariância(X, Y) / Variância(X)
    Fórmula do Coeficiente Linear: média(Y) - Coeficiente Angular * média(X)
    """
    x_media = est.media(x)
    y_media = est.media(y)
    
    covariancia_xy = est.covariancia(x, y)
    variancia_x = est.variancia(x)
    
    coef_angular = covariancia_xy / variancia_x
    coef_linear = y_media - (coef_angular * x_media)
    
    return coef_angular, coef_linear


def gradient_descent_passo(x: list[float], y: list[float], taxa_aprendizado: float, peso: float, bias: float, modelo="linear") -> tuple[float, float]:
    """
    Realiza um único passo de atualização de peso (coef_angular) e bias (coef_linear) usando Gradiente Descendente.
    Funciona de forma unificada tanto para a Regressão Linear (MSE) quanto para a Regressão Logística (Log Loss).
    
    Paralelo Matemático dos Modelos:
    A descida de gradiente atualiza os parâmetros na direção oposta ao gradiente da perda:
      peso_novo = peso - taxa_aprendizado * gradiente_peso
      bias_novo = bias - taxa_aprendizado * gradiente_bias
      
    Onde a derivada da perda em relação aos parâmetros tem a mesma estrutura:
      gradiente_bias = escala * media(y_pred - y)
      gradiente_peso = escala * media((y_pred - y) * x)
      
    Convergência Matemática:
      - Regressão Linear (Perda MSE): escala = 2, y_pred = peso * x + bias
      - Regressão Logística (Perda Log Loss): escala = 1, y_pred = sigmoide(peso * x + bias)
    """
    y_pred = realizar_previsao(x, peso, bias, modelo=modelo)
    escala = 2.0 if modelo == "linear" else 1.0

    # Erro residual: y_pred - y
    erro_residual = alg.subtrair_vetores(y_pred, y)

    # Cálculo unificado dos gradientes usando médias da biblioteca de estatística e álgebra
    gradiente_bias = escala * est.media(erro_residual)
    gradiente_peso = escala * est.media(alg.produto_vetor(erro_residual, x))

    # Atualização
    novo_bias = bias - (taxa_aprendizado * gradiente_bias)
    novo_peso = peso - (taxa_aprendizado * gradiente_peso)

    return novo_peso, novo_bias


def gradient_descent(x: list[float], y: list[float], taxa_de_aprendizado: float, coef_linear=0.0, coef_angular=0.0) -> tuple[float, float]:
    """
    Realiza uma única etapa do Gradiente Descendente para regressão linear simples.
    Mantido para retrocompatibilidade; delega o passo para a função unificada gradient_descent_passo.
    """
    novo_peso, novo_bias = gradient_descent_passo(x, y, taxa_de_aprendizado, coef_angular, coef_linear, modelo="linear")
    return novo_bias, novo_peso


def realizar_previsao(x: list[float], peso: float, bias: float, modelo="linear") -> list[float]:
    """
    Gera previsões para um vetor de entrada x dado um peso e bias.
    - Para 'linear': y_pred = peso * x_i + bias (Regressão Linear)
    - Para 'logistica': y_pred = sigmoide(peso * x_i + bias) (Regressão Logística)
    """
    if modelo == "linear":
        return [(peso * xi) + bias for xi in x]
    elif modelo == "logistica":
        return [alg.sigmoide((peso * xi) + bias) for xi in x]
    else:
        raise ValueError("Modelo desconhecido. Escolha 'linear' ou 'logistica'.")


def calcular_erro(y: list[float], y_pred: list[float]) -> list[float]:
    """Calcula os resíduos simples entre o valor real e a previsão."""
    return alg.subtrair_vetores(y, y_pred)


def rmse(y: list[float], y_pred: list[float]) -> dict[str, float]:
    """Calcula métricas estatísticas de erro: RMSE, MSE e Erro Percentual Médio."""
    n_y = len(y)
    if n_y == 0:
        raise ValueError("O vetor de dados não pode estar vazio.")

    y_media = est.media(y)
    y_erro = calcular_erro(y, y_pred)
    
    mse = est.media(alg.vetor_expoente(y_erro, 2))
    rmse_val = mse ** 0.5
    erro_percentual_medio = (rmse_val / y_media) * 100 if y_media != 0 else 0.0

    return {
        "RMSE": rmse_val,
        "MSE": mse,
        "Erro Percentual Médio": erro_percentual_medio
    }


# --- MODELO 3: REGRESSÃO LOGÍSTICA ---

def treinar_logistica(X: list[float], y: list[float], taxa_aprendizado: float, epocas: int) -> tuple[float, float]:
    """
    Treina o modelo de Regressão Logística do zero.
    Corrige o peso e o bias via Gradiente Descendente.
    """
    peso = 0.0
    bias = 0.0
    for _ in range(epocas):
        peso, bias = gradient_descent_passo(X, y, taxa_aprendizado, peso, bias, modelo="logistica")
        
    return peso, bias


# --- METRICAS DE AVALIAÇÃO DE CLASSIFICAÇÃO ---

def matriz_confusao(y_real: list[int], y_pred: list[int]) -> dict[str, int]:
    """
    Calcula os elementos da matriz de confusão:
    - Verdadeiros Positivos (VP)
    - Verdadeiros Negativos (VN)
    - Falsos Positivos (FP)
    - Falsos Negativos (FN)
    """
    if len(y_real) != len(y_pred):
        raise ValueError("Os vetores y_real e y_pred devem ter o mesmo tamanho.")
        
    vp = sum(1 for r, p in zip(y_real, y_pred) if r == 1 and p == 1)
    vn = sum(1 for r, p in zip(y_real, y_pred) if r == 0 and p == 0)
    fp = sum(1 for r, p in zip(y_real, y_pred) if r == 0 and p == 1)
    fn = sum(1 for r, p in zip(y_real, y_pred) if r == 1 and p == 0)
    
    return {"VP": vp, "VN": vn, "FP": fp, "FN": fn}


def acuracia(y_real: list[int], y_pred: list[int]) -> float:
    """Calcula a acurácia (fração de previsões corretas)."""
    mc = matriz_confusao(y_real, y_pred)
    total = sum(mc.values())
    if total == 0:
        return 0.0
    return (mc["VP"] + mc["VN"]) / total


def precisao(y_real: list[int], y_pred: list[int]) -> float:
    """Calcula a precisão (VP / (VP + FP))."""
    mc = matriz_confusao(y_real, y_pred)
    denominador = mc["VP"] + mc["FP"]
    if denominador == 0:
        return 0.0
    return mc["VP"] / denominador


def revocacao(y_real: list[int], y_pred: list[int]) -> float:
    """Calcula a revocação/recall (VP / (VP + FN))."""
    mc = matriz_confusao(y_real, y_pred)
    denominador = mc["VP"] + mc["FN"]
    if denominador == 0:
        return 0.0
    return mc["VP"] / denominador


def f1_score(y_real: list[int], y_pred: list[int]) -> float:
    """Calcula o F1-Score (média harmônica de precisão e revocação)."""
    prec = precisao(y_real, y_pred)
    rec = revocacao(y_real, y_pred)
    if (prec + rec) == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)