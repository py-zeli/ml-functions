# -*- coding: utf-8 -*-
"""
Biblioteca de Distribuições de Probabilidade (probabilidades.py)
---------------------------------------------------------------
Centraliza as funções de cálculo de distribuições Discretas (Binomial e Poisson).
"""
import math

def fatorial(n: int) -> int:
    """
    Calcula o fatorial de um número inteiro não-negativo de forma iterativa.
    Exemplo: fatorial(5) = 5 * 4 * 3 * 2 * 1 = 120
    """
    if n < 0:
        raise ValueError("O fatorial não está definido para números negativos.")
    resultado = 1
    for i in range(2, n + 1):
        resultado *= i
    return resultado


def combinacao(n: int, k: int) -> int:
    """
    Calcula a combinação simples n escolhe k (coeficiente binomial).
    Fórmula: C(n, k) = n! / (k! * (n - k)!)
    """
    if k < 0 or k > n:
        return 0
    return fatorial(n) // (fatorial(k) * fatorial(n - k))


# --- DISTRIBUIÇÃO BINOMIAL ---

def binomial_pmf(k: int, n: int, p: float) -> float:
    """
    Calcula a Função de Massa de Probabilidade (PMF) da distribuição Binomial.
    Mapeia a probabilidade de obter exatamente k sucessos em n tentativas independentes,
    cada uma com probabilidade de sucesso p.
    Fórmula: P(X = k) = C(n, k) * p^k * (1 - p)^(n - k)
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("A probabilidade p deve estar no intervalo [0, 1].")
    if n < 0:
        raise ValueError("O número de tentativas n deve ser não-negativo.")
    if k < 0 or k > n:
        return 0.0
    return combinacao(n, k) * (p ** k) * ((1.0 - p) ** (n - k))


def binomial_cdf(k: int, n: int, p: float) -> float:
    """
    Calcula a Função de Distribuição Acumulada (CDF) da distribuição Binomial.
    Mapeia a probabilidade de obter no máximo k sucessos (X <= k) em n tentativas.
    Fórmula: P(X <= k) = sum_{i=0}^k C(n, i) * p^i * (1 - p)^(n - i)
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("A probabilidade p deve estar no intervalo [0, 1].")
    if n < 0:
        raise ValueError("O número de tentativas n deve ser não-negativo.")
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(binomial_pmf(i, n, p) for i in range(k + 1))


# --- DISTRIBUIÇÃO DE POISSON ---

def poisson_pmf(k: int, lambd: float) -> float:
    """
    Calcula a Função de Massa de Probabilidade (PMF) da distribuição de Poisson.
    Mapeia a probabilidade de ocorrer exatamente k eventos em um intervalo fixo,
    dada uma taxa média de ocorrência lambda.
    Fórmula: P(X = k) = (lambda^k * e^(-lambda)) / k!
    """
    if lambd <= 0.0:
        raise ValueError("A taxa média lambda deve ser estritamente maior que zero.")
    if k < 0:
        return 0.0
    return (lambd ** k * math.exp(-lambd)) / fatorial(k)


def poisson_cdf(k: int, lambd: float) -> float:
    """
    Calcula a Função de Distribuição Acumulada (CDF) da distribuição de Poisson.
    Mapeia a probabilidade de ocorrer no máximo k eventos (X <= k).
    Fórmula: P(X <= k) = sum_{i=0}^k (lambda^i * e^(-lambda)) / i!
    """
    if lambd <= 0.0:
        raise ValueError("A taxa média lambda deve ser estritamente maior que zero.")
    if k < 0:
        return 0.0
    return sum(poisson_pmf(i, lambd) for i in range(k + 1))


# --- DISTRIBUIÇÃO NORMAL ---

def normal_cdf(x: float, mu: float, sigma: float) -> float:
    """
    Calcula a Função de Distribuição Acumulada (CDF) da distribuição Normal.
    Mapeia a probabilidade de obter um valor menor ou igual a x (X <= x).
    Fórmula: P(X <= x) = 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2))))
    """
    if sigma <= 0.0:
        raise ValueError("O desvio padrão sigma deve ser estritamente maior que zero.")
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))

