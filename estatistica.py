# -*- coding: utf-8 -*-
"""
Biblioteca de Estatística Descritiva e Primitivas Amostrais (estatistica.py)
--------------------------------------------------------------------------
Centraliza as funções de média, desvios, variância, covariância e correlação.
"""

def media(vetor: list[float]) -> float:
    """Calcula a média aritmética simples de um vetor."""
    if not vetor:
        raise ValueError("O vetor não pode ser vazio.")
    return sum(vetor) / len(vetor)


def desvio(vetor: list[float], m=None) -> list[float]:
    """Retorna o resíduo/desvio de cada escalar do vetor em relação à sua média."""
    if m is None:
        m = media(vetor)
    return [x - m for x in vetor]


def variancia(vetor: list[float]) -> float:
    """Calcula a variância amostral de um conjunto de dados (com Correção de Bessel)."""
    if len(vetor) < 2:
        raise ValueError("Para variância amostral, o vetor deve ter pelo menos 2 elementos.")
    m = media(vetor) 
    return sum((x - m) ** 2 for x in vetor) / (len(vetor) - 1)


def desvio_padrao(vetor: list[float]) -> float:
    """Calcula o desvio padrão amostral de um vetor."""
    return variancia(vetor) ** 0.5


def covariancia(vetor_x: list[float], vetor_y: list[float]) -> float:
    """Calcula a covariância amostral entre dois vetores."""
    if len(vetor_x) != len(vetor_y):
        raise ValueError("Os vetores x e y devem ter o mesmo tamanho.")
    if len(vetor_x) < 2:
        raise ValueError("Os vetores devem ter pelo menos 2 elementos para covariância amostral.")
    
    m_x = media(vetor_x)
    m_y = media(vetor_y)
    return sum((xi - m_x) * (yi - m_y) for xi, yi in zip(vetor_x, vetor_y)) / (len(vetor_x) - 1)


def coeficiente_pearson(vetor_x: list[float], vetor_y: list[float]) -> float:
    """Calcula o coeficiente de correlação linear de Pearson entre dois vetores."""
    n = len(vetor_x)
    if n != len(vetor_y):
        raise ValueError("Os vetores x e y devem ter o mesmo tamanho.")
        
    soma_x = sum(vetor_x)
    soma_y = sum(vetor_y)
    
    soma_x2 = sum(xi ** 2 for xi in vetor_x)
    soma_y2 = sum(yi ** 2 for yi in vetor_y)
    soma_xy = sum(xi * yi for xi, yi in zip(vetor_x, vetor_y))

    numerador = (n * soma_xy) - (soma_x * soma_y)
    denominador = ((n * soma_x2 - soma_x ** 2) * (n * soma_y2 - soma_y ** 2)) ** 0.5

    if denominador == 0:
        return 0.0
    return numerador / denominador


def coeficiente_angular(vetor_x: list[float], vetor_y: list[float]) -> float:
    """Calcula o coeficiente angular (declividade) da reta de regressão linear."""
    return (coeficiente_pearson(vetor_x, vetor_y) * desvio_padrao(vetor_y)) / desvio_padrao(vetor_x)
