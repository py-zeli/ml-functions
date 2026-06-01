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
                coef_l, coef_a = gradient_descent(x, y, taxa_aprendizado, coef_l, coef_a)
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


def gradient_descent(x: list[float], y: list[float], taxa_de_aprendizado: float, coef_linear=0.0, coef_angular=0.0) -> tuple[float, float]:
    """Realiza uma única etapa do Gradiente Descendente para atualizar os pesos."""

    # Previsão: y = ax + b
    y_pred = realizar_previsao(x, coef_angular, coef_linear)

    # Cálculo dos erros e gradientes
    y_erro = calcular_erro(y, y_pred)
    soma_erro = sum(y_erro)
    
    # Primitiva de produto escalar elemento a elemento (expressão geradora)
    soma_prod_erro_xi = sum(err * xi for err, xi in zip(y_erro, x))

    gradiente_coef_angular = (-2 / len(x)) * soma_prod_erro_xi
    gradiente_coef_linear = (-2 / len(x)) * soma_erro

    # Atualização dos coeficientes
    novo_coef_linear = coef_linear - (taxa_de_aprendizado * gradiente_coef_linear)
    novo_coef_angular = coef_angular - (taxa_de_aprendizado * gradiente_coef_angular)

    return novo_coef_linear, novo_coef_angular


def realizar_previsao(x: list[float], coef_angular: float, coef_linear: float) -> list[float]:
    """Gera previsões para regressão simples: y_pred = ax + b."""
    return [(coef_angular * xi) + coef_linear for xi in x]


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
    
    mse = est.media([err ** 2 for err in y_erro])
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
    tamanho_vetor = len(X)
    
    for _ in range(epocas):
        erros = [alg.predizer_probabilidade(xi, peso, bias) - yi for xi, yi in zip(X, y)]
        
        gradiente_peso = sum(erro * xi for erro, xi in zip(erros, X)) / tamanho_vetor
        gradiente_bias = sum(erros) / tamanho_vetor
        
        # Correção dos pesos
        peso -= taxa_aprendizado * gradiente_peso
        bias -= taxa_aprendizado * gradiente_bias
        
    return peso, bias