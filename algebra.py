# -*- coding: utf-8 -*-
"""
Biblioteca de Álgebra Linear e Operações Matriciais (algebra.py)
--------------------------------------------------------------
Esta biblioteca centraliza todas as primitivas matemáticas de vetores
e matrizes (representadas de forma orientada a colunas).
"""
import math

# --- OPERAÇÕES VETORIAIS ---

def vetor_expoente(vetor: list[float], n: float) -> list[float]:
    """Eleva cada elemento do vetor à potência n."""
    return [i ** n for i in vetor]


def produto_vetor(vetor_a: list[float], vetor_b: list[float]) -> list[float]:
    """Calcula o produto elemento a elemento entre dois vetores."""
    if len(vetor_a) != len(vetor_b):
        raise ValueError("Os vetores devem ter o mesmo tamanho para o produto elemento a elemento.")
    return [vetor_a_i * vetor_b_i for vetor_a_i, vetor_b_i in zip(vetor_a, vetor_b)]


def subtrair_vetores(vetor_a: list[float], vetor_b: list[float]) -> list[float]:
    """Subtrai vetor_b de vetor_a elemento a elemento."""
    if len(vetor_a) != len(vetor_b):
        raise ValueError("Os vetores devem ter o mesmo tamanho para subtração.")
    return [a - b for a, b in zip(vetor_a, vetor_b)]


# --- FUNÇÕES DE ATIVAÇÃO ---

def sigmoide(z: float) -> float:
    """Função de ativação sigmoide estável (evita overflow para z muito negativos)."""
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    else:
        exp_z = math.exp(z)
        return exp_z / (1 + exp_z)


def predizer_probabilidade(x: float, peso: float, bias: float) -> float:
    """Calcula o logit (z) e retorna a probabilidade correspondente (curva sigmoide)."""
    z = (x * peso) + bias
    return sigmoide(z)


# --- OPERAÇÕES MATRICIAIS (Representação de lista de colunas) ---

def criar_matriz(*args: list[float]) -> list[list[float]]:
    """Cria uma matriz na representação de lista de colunas."""
    return [list(vetor) for vetor in args]


def dimensionar_matriz(matriz: list[list[float]]) -> tuple[int, int]:
    """Retorna a quantidade de colunas e linhas da matriz."""
    if not matriz or not matriz[0]:
        return 0, 0
    return len(matriz), len(matriz[0])


def transpor_matriz(matriz: list[list[float]]) -> list[list[float]]:
    """Transpõe a matriz invertendo colunas por linhas em uma única linha Pythônica."""
    return [list(coluna) for coluna in zip(*matriz)]


def multiplicar_matrizes(matriz1: list[list[float]], matriz2: list[list[float]]) -> list[list[float]]:
    """Multiplica duas matrizes na representação de colunas."""
    n_colunas_m1, n_linhas_m1 = dimensionar_matriz(matriz1)
    n_colunas_m2, n_linhas_m2 = dimensionar_matriz(matriz2)
    
    if n_colunas_m1 != n_linhas_m2:
        raise ValueError("O número de colunas de matriz1 deve ser igual ao número de linhas de matriz2.")

    linhas_m1 = list(zip(*matriz1))
    return [
        [sum(a * b for a, b in zip(linha_a, coluna_b)) for linha_a in linhas_m1]
        for coluna_b in matriz2
    ]


def multiplicar_matriz_vetor(matriz: list[list[float]], vetor: list[float]) -> list[float]:
    """Multiplica cada coluna da matriz pelo vetor (calculando o produto escalar de cada coluna com o vetor)."""
    if not matriz:
        return []
    if len(matriz[0]) != len(vetor):
        raise ValueError("O tamanho das colunas da matriz deve ser igual ao tamanho do vetor.")

    return [sum(a * b for a, b in zip(coluna, vetor)) for coluna in matriz]


def diagonal_matriz(matriz: list[list[float]]) -> list[list[int]]:
    """Obtém as coordenadas da diagonal principal de uma matriz quadrada."""
    n_colunas, n_linhas = dimensionar_matriz(matriz)
    limite = min(n_colunas, n_linhas)
    indices = list(range(limite))
    return [indices, indices]


def transformar_em_triangular(matriz: list[list[float]]) -> list[list[float]]:
    """Realiza eliminação Gaussiana para obter uma matriz triangular superior."""
    n_colunas = len(matriz) 
    A = [[float(x) for x in coluna] for coluna in matriz]
    n_linhas = len(A[0])

    for j in range(n_colunas):
        pivo = A[j][j]
        if pivo == 0:
            continue

        for i in range(j + 1, n_linhas):
            fator = A[j][i] / pivo
            for k in range(j, n_colunas):
                A[k][i] -= fator * A[k][j]
                
    return A


def multiplicar_diagonal(matriz_diagonal: list[list[float]]) -> float:
    """Calcula o produto dos elementos na diagonal principal (determinante de triangular)."""
    coordenadas = diagonal_matriz(matriz_diagonal)
    produto = 1.0
    for col, lin in zip(coordenadas[0], coordenadas[1]):
        produto *= matriz_diagonal[col][lin]
    return produto


def determinante_2x2(matriz_menor: list[list[float]]) -> float:
    """Calcula o determinante de uma matriz de tamanho 2x2."""
    (a, b), (c, d) = matriz_menor
    return (a * d) - (b * c)


def matriz_cofatores(matriz: list[list[float]]) -> list[list[float]]:
    """Gera a matriz de cofatores para uma matriz quadrada 3x3."""
    n = len(matriz)
    cofatores = []
    
    for i in range(n):
        linha_cofatores = []
        for j in range(n):
            submatriz = []
            for linha_idx in range(n):
                if linha_idx != i:
                    nova_linha = matriz[linha_idx][:j] + matriz[linha_idx][j+1:]
                    submatriz.append(nova_linha)
            
            menor = determinante_2x2(submatriz)
            sinal = (-1) ** (i + j)
            linha_cofatores.append(sinal * menor)
        cofatores.append(linha_cofatores)
        
    return cofatores


def matriz_adjunta(matriz: list[list[float]]) -> list[list[float]]:
    """Obtém a matriz adjunta (transposta da matriz de cofatores)."""
    cofatores = matriz_cofatores(matriz)
    return transpor_matriz(cofatores)


def primeiro_fator(determinante: float, matriz_adjunta: list[list[float]]) -> list[list[float]]:
    """Multiplica a matriz adjunta pelo inverso do determinante para obter a inversa."""
    if determinante == 0:
        raise ValueError("A matriz não pode ser invertida pois o determinante é zero.")
    det_inverso = 1.0 / determinante
    return [[det_inverso * elemento for elemento in linha] for linha in matriz_adjunta]


def inverter_matriz(matriz: list[list[float]]) -> list[list[float]]:
    """Calcula a inversa de uma matriz quadrada (3x3 ou menor) usando cofatores e adjunta."""
    matriz_diagonal = transformar_em_triangular(matriz)
    determinante = multiplicar_diagonal(matriz_diagonal)
    
    if determinante == 0:
        raise ValueError("A matriz não pode ser invertida pois o determinante é zero.")
        
    adj = matriz_adjunta(matriz)
    det_inverso = 1.0 / determinante
    return [[det_inverso * elemento for elemento in linha] for linha in adj]
