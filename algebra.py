# -*- coding: utf-8 -*-
"""
Biblioteca de Álgebra Linear e Estatística (algebra.py)
------------------------------------------------------
Esta biblioteca centraliza todas as primitivas matemáticas de vetores
e matrizes (representadas de forma orientada a colunas).
"""

# --- OPERAÇÕES VETORIAIS ---

def media(vetor: list[float]) -> float:
    """Calcula a média aritmética simples de um vetor."""
    if not vetor:
        raise ValueError("O vetor não pode ser vazio.")
    return float(sum(vetor) / len(vetor))


def vetor_expoente(vetor: list[float], n: float) -> list[float]:
    """Eleva cada elemento do vetor à potência n."""
    return [i ** n for i in vetor]


def produto_vetor(vetor_a: list[float], vetor_b: list[float]) -> list[float]:
    """Calcula o produto elemento a elemento entre dois vetores."""
    if len(vetor_a) != len(vetor_b):
        raise ValueError("Os vetores devem ter o mesmo tamanho para o produto elemento a elemento.")
    return [vetor_a_i * vetor_b_i for vetor_a_i, vetor_b_i in zip(vetor_a, vetor_b)]


def variancia(vetor: list[float]) -> float:
    """Calcula a variância amostral de um conjunto de dados."""
    m = media(vetor) 
    list_variancia = [(i - m)**2 for i in vetor]
    return sum(list_variancia) / len(vetor)


def desvio(vetor: list[float], m=None) -> list[float]:
    """Retorna o resíduo/desvio de cada escalar do vetor em relação à sua média."""
    if m is None:
        m = media(vetor)
    return [(i - m) for i in vetor]


def desvio_padrao(vetor: list[float]) -> float:
    """Calcula o desvio padrão de um vetor."""
    return variancia(vetor) ** 0.5


def subtrair_vetores(vetor_a: list[float], vetor_b: list[float]) -> list[float]:
    """Subtrai vetor_b de vetor_a elemento a elemento."""
    if len(vetor_a) != len(vetor_b):
        raise ValueError("Os vetores devem ter o mesmo tamanho para subtração.")
    return [a - b for a, b in zip(vetor_a, vetor_b)]


def coeficiente_pearson(vetor_x: list[float], vetor_y: list[float]) -> float:
    """Calcula o coeficiente de correlação linear de Pearson entre dois vetores."""
    n = len(vetor_x)
    if n != len(vetor_y):
        raise ValueError("Os vetores x e y devem ter o mesmo tamanho.")
        
    soma_x = sum(vetor_x)
    soma_y = sum(vetor_y)
    
    soma_x2 = sum(xi ** 2 for xi in vetor_x)
    soma_y2 = sum(yi ** 2 for yi in vetor_y)
    soma_xy = sum(i * j for i, j in zip(vetor_x, vetor_y))

    numerador = (n * soma_xy) - (soma_x * soma_y)
    denominador = ((n * soma_x2 - soma_x ** 2) * (n * soma_y2 - soma_y ** 2)) ** 0.5

    if denominador == 0:
        return 0.0
    return numerador / denominador


def coeficiente_angular(vetor_x: list[float], vetor_y: list[float]) -> float:
    """Calcula o coeficiente angular (declividade) da reta de regressão linear."""
    return (coeficiente_pearson(vetor_x, vetor_y) * desvio_padrao(vetor_y)) / desvio_padrao(vetor_x)


# --- FUNÇÕES DE ATIVAÇÃO ---

def sigmoide(z: float) -> float:
    """Função de ativação sigmoide (mapeia qualquer real para o intervalo [0, 1])."""
    EULER = 2.718281828459045
    return 1 / (1 + (EULER ** (-z)))


def predizer_probabilidade(x: float, peso: float, bias: float) -> float:
    """Calcula o logit (z) e retorna a probabilidade correspondente (curva sigmoide)."""
    z = (x * peso) + bias
    return sigmoide(z)


# --- OPERAÇÕES MATRICIAIS (Representação de lista de colunas) ---

def criar_matriz(*args: list[float]) -> list[list[float]]:
    """Cria uma matriz na representação de lista de colunas."""
    return [list(vetor) for vetor in args]


def dimensionar_matriz(matriz: list[list[float]]) -> tuple[range, range]:
    """Retorna ranges correspondentes à quantidade de colunas e linhas."""
    return range(len(matriz)), range(len(matriz[0]))


def transpor_matriz(matriz: list[list[float]]) -> list[list[float]]:
    """Transpõe a matriz invertendo colunas por linhas em uma única linha Pythônica."""
    return [list(coluna) for coluna in zip(*matriz)]


def multiplicar_matrizes(matriz1: list[list[float]], matriz2: list[list[float]]) -> list[list[float]]:
    """Multiplica duas matrizes na representação de colunas."""
    if len(matriz1[0]) != len(matriz2):
        raise ValueError("O número de colunas/linhas das matrizes é incompatível para multiplicação.")

    matriz_produto = []
    i_m1_range, j_m1_range = dimensionar_matriz(matriz1)
    i_m2_range, j_m2_range = dimensionar_matriz(matriz2)

    for im1 in i_m1_range:
        for j_m2 in j_m2_range:
            produtos = []
            for j_m1, im2 in zip(j_m1_range, i_m2_range):
                while len(matriz_produto) < j_m2_range[-1] + 1:
                    matriz_produto.append([])
                fator1 = matriz1[im1][j_m1]
                fator2 = matriz2[im2][j_m2]
                produtos.append(fator1 * fator2)
            matriz_produto[j_m2].insert(im2, sum(produtos))

    return matriz_produto


def multiplicar_matriz_vetor(matriz: list[list[float]], vetor: list[float]) -> list[float]:
    """Multiplica uma matriz orientada a colunas por um vetor."""
    if len(matriz[0]) != len(vetor):
        raise ValueError("O número de linhas por coluna deve ser igual ao tamanho do vetor.")

    resultado = []
    for linha in matriz:
        soma = sum(a * b for a, b in zip(linha, vetor))
        resultado.append(soma)
        
    return resultado


def diagonal_matriz(matriz: list[list[float]]) -> list[list[int]]:
    """Obtém as coordenadas da diagonal principal de uma matriz quadrada."""
    tamanho_i, tamanho_j = dimensionar_matriz(matriz)
    limite = min(tamanho_i[-1], tamanho_j[-1]) + 1
    return [list(range(limite)), list(range(limite))]


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
    a = matriz_menor[0][0]
    b = matriz_menor[0][1]
    c = matriz_menor[1][0]
    d = matriz_menor[1][1]
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
