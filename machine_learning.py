# -*- coding: utf-8 -*-
"""
Biblioteca Local de Machine Learning (machine_learning.py)
-----------------------------------------------------------
Esta biblioteca agrupa os modelos preditivos e algoritmos de ajuste (OLS, GD),
delegando primitivas estatísticas e matriciais para o módulo 'algebra.py'.
"""

import algebra as alg

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
                ajustes = gradient_descent(x, y, taxa_aprendizado, coef_l, coef_a)
                coef_l = ajustes["novo_coef_linear"]
                coef_a = ajustes["novo_coef_angular"]
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
        XtX = alg.multiplicar_matrizes(X_transposta, X_normal)
        
        # 2. Resolução do determinante e adjunta para inversão
        matriz_diagonal = alg.transformar_em_triangular(XtX)
        determinante = alg.multiplicar_diagonal(matriz_diagonal)
        
        if determinante == 0:
            raise ValueError("A matriz XtX não é invertível (determinante igual a zero).")
            
        adjunta = alg.matriz_adjunta(XtX)
        inversa = alg.primeiro_fator(determinante, adjunta)
        
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
    x_media = alg.media(x)
    y_media = alg.media(y)
    
    # Desvios em relação à média (reutilizando a média calculada)
    x_desvios = alg.desvio(x, x_media)
    y_desvios = alg.desvio(y, y_media)
    
    # Inclinação da reta
    covariancia_xy = sum(alg.produto_vetor(x_desvios, y_desvios))
    variancia_x = sum(alg.vetor_expoente(x_desvios, 2))
    
    coef_angular = covariancia_xy / variancia_x
    coef_linear = y_media - (coef_angular * x_media)
    
    return coef_angular, coef_linear


def gradient_descent(x: list[float], y: list[float], taxa_de_aprendizado: float, coef_linear=0.0, coef_angular=0.0) -> dict[str, float]:
    """Realiza uma única etapa do Gradiente Descendente para atualizar os pesos."""
    tamanho_vetor = len(x)

    # Previsão: y = ax + b
    y_pred = realizar_previsao(x, coef_angular, coef_linear)

    # Cálculo dos erros e gradientes
    y_erro = calcular_erro(y, y_pred)
    soma_erro = sum(y_erro)
    
    # Primitiva de produto escalar elemento a elemento
    prod_erro_xi = alg.produto_vetor(y_erro, x)
    soma_prod_erro_xi = sum(prod_erro_xi)

    gradiente_coef_angular = (-2 / tamanho_vetor) * soma_prod_erro_xi
    gradiente_coef_linear = (-2 / tamanho_vetor) * soma_erro

    # Atualização dos coeficientes
    novo_coef_linear = coef_linear - (taxa_de_aprendizado * gradiente_coef_linear)
    novo_coef_angular = coef_angular - (taxa_de_aprendizado * gradiente_coef_angular)

    return {
        "antigo_coef_linear": coef_linear,
        "novo_coef_linear": novo_coef_linear,
        "antigo_coef_angular": coef_angular,
        "novo_coef_angular": novo_coef_angular
    }


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

    y_media = alg.media(y)
    y_erro = calcular_erro(y, y_pred)
    y_erro_quadrado = alg.vetor_expoente(y_erro, 2)

    mse = alg.media(y_erro_quadrado)
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
        soma_gradiente_peso = 0.0
        soma_gradiente_bias = 0.0
        
        for xi, yi in zip(X, y):
            y_predito = alg.predizer_probabilidade(xi, peso, bias)
            erro = y_predito - yi
            
            soma_gradiente_peso += erro * xi
            soma_gradiente_bias += erro
            
        gradiente_peso = soma_gradiente_peso / tamanho_vetor
        gradiente_bias = soma_gradiente_bias / tamanho_vetor
        
        # Correção dos pesos
        peso -= taxa_aprendizado * gradiente_peso
        bias -= taxa_aprendizado * gradiente_bias
        
    return peso, bias