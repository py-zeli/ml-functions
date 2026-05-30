import algebra as alg


class RegressaoLinear:
    """Modelo de Regressão Linear Simples."""
    def __init__(self):
        self.coef_angular = None
        self.coef_linear = None

    def fit(self, x: list, y: list, method="ols", taxa_aprendizado=0.01, epocas=5000):
        """
        Ajusta o modelo calibrando os coeficientes.
        Métodos:
        - 'ols': Mínimos Quadrados Ordinários (analítico).
        - 'gd' ou 'gradient_descent': Gradiente Descendente (iterativo).
        """
        if len(x) != len(y):
            raise ValueError("Os vetores x e y devem ter o mesmo tamanho.")
            
        if method == "ols":
            self.coef_angular, self.coef_linear = minimos_quadrados(x, y)
        elif method in ("gd", "gradient_descent"):
            coef_l = 0.0
            coef_a = 0.0
            for _ in range(epocas):
                ajustes = gradient_descent(x, y, taxa_aprendizado, coef_l, coef_a)
                coef_l = ajustes["novo_coef_linear"]
                coef_a = ajustes["novo_coef_angular"]
            self.coef_linear = coef_l
            self.coef_angular = coef_a
        else:
            raise ValueError("Método desconhecido. Escolha entre 'ols' ou 'gd' (gradient_descent).")
            
        return self

    def predict(self, x: list) -> list:
        """Gera previsões para novos dados de entrada."""
        if self.coef_angular is None or self.coef_linear is None:
            raise RuntimeError("O modelo precisa ser treinado com .fit() antes de prever.")
            
        return [self.coef_angular * xi + self.coef_linear for xi in x]

"""REGRESSÃO LINEAR SIMPLES"""

def minimos_quadrados(x: list, y: list):
    """
    Ajusta o modelo de Regressão Linear Simples.
    Fórmula do Coeficiente Angular: Covariância(X, Y) / Variância(X)
    Fórmula do Coeficiente Linear: média(Y) - Coeficiente Angular * média(X)
    """
    # Escalar
    x_media = alg.media(x)

    # Escalar
    y_media = alg.media(y)

    # Vetor
    x_desvios = alg.desvio(x)

    # Vetor
    y_desvios = alg.desvio(y)

    # Escalar
    covariancia_xy = sum(alg.produto_vetor(x_desvios, y_desvios))

    # Escalar
    variancia_x = sum(alg.vetor_expoente(x_desvios, 2))

    # Escalar
    coef_angular = covariancia_xy / variancia_x

    # Escalar
    coef_linear = y_media - (coef_angular * x_media)
    
    return coef_angular, coef_linear

def realizar_previsao(x, coef_angular, coef_linear):

    y_pred = [(coef_angular * xi) + coef_linear for xi in x]

    return y_pred

def calcular_erro(y, y_pred):
        
    y_erro = [yi - y_pred_i for yi, y_pred_i in zip(y, y_pred) ]

    return y_erro

def rmse(y, y_pred):

    n_y = len(y)

    y_media = sum(y)/n_y
        
    y_erro = [yi - y_pred_i for yi, y_pred_i in zip(y, y_pred)]

    y_erro_quadrado = [y_erro_i ** 2 for y_erro_i in y_erro]

    soma_erros_quadrados = sum(y_erro_quadrado)

    mse = soma_erros_quadrados/n_y

    rmse = mse ** (1/2)

    erro_percentual_medio = (rmse/y_media)*100

    valores_encontrados = {
        "RMSE" : rmse,
        "MSE" : mse,
        "Erro Percentual Médio" : rmse
    }

    return valores_encontrados

def gradient_descent(x, y, taxa_de_aprendizado, coef_linear=0.0, coef_angular=0.0):
    tamanho_vetor = len(x)

    # Previsão: y = ax + b
    y_pred = [coef_angular * xi + coef_linear for xi in x]

    y_erro = calcular_erro(y, y_pred)

    soma_erro = sum(y_erro)

    prod_erro_xi = [y_erro_i * xi for y_erro_i, xi in zip(y_erro, x)]

    soma_prod_erro_xi = sum(prod_erro_xi)

    gradiente_coef_angular = (-2 / tamanho_vetor) * soma_prod_erro_xi

    gradiente_coef_linear = (-2 / tamanho_vetor) * soma_erro

    novo_coef_linear = coef_linear - (taxa_de_aprendizado * gradiente_coef_linear)

    novo_coef_angular = coef_angular - (taxa_de_aprendizado * gradiente_coef_angular)

    valores = {
        "antigo_coef_linear": coef_linear,
        "novo_coef_linear": novo_coef_linear,
        "antigo_coef_angular": coef_angular,
        "novo_coef_angular": novo_coef_angular
    }

    return valores


"""REGRESSÃO LINEAR MÚLTIPLA (POLINOMIAL)"""

def criar_matriz(*args):
    matriz = [vetor for vetor in args]
    return matriz

def dimensionar_matriz(matriz):

    i_tamanho = range(len(matriz))

    j_tamanho = range(len(matriz[0]))

    return i_tamanho, j_tamanho

def transpor_matriz(matriz):

    matriz_transposta=[]

    i_tamanho, j_tamanho = dimensionar_matriz(matriz)
    
    try:

        for j in j_tamanho: 
                
            for i in i_tamanho:

                while len(matriz_transposta) < j_tamanho[-1]+1:
                    matriz_transposta.append([])

                matriz_transposta[j].insert(i,matriz[i][j])
                
        return matriz_transposta

    except IndexError:
        print(f"\nDeu erro!\nCoordenada da falha: {i,j}")
        print(f"Resultado até aqui:\n {matriz_transposta}")

    
    except KeyboardInterrupt:
        print(f"TimeOut!\nCoordenada da falha: {i,j}")
        print(f"Tamanho:\n {len(matriz_transposta[j])}")
        print(f"Resultado até aqui:\n {dimensionar_matriz(matriz_transposta)}")

def multiplicar_matrizes(matriz1, matriz2):

    if len(matriz1[0]) != len(matriz2):
        raise ValueError("O número de colunas da matriz deve ser igual ao tamanho do vetor.")

    matriz_produto = []

    i_m1_range, j_m1_range = dimensionar_matriz(matriz1)
    i_m2_range, j_m2_range = dimensionar_matriz(matriz2)

    try:
        for im1 in i_m1_range:

            for j_m2 in j_m2_range:

                produtos = []

                for j_m1, im2 in zip(j_m1_range, i_m2_range):

                    while len(matriz_produto) < j_m2_range[-1]+1:
                        matriz_produto.append([])

                    
                    fator1 = matriz1[im1][j_m1]
                    fator2 = matriz2[im2][j_m2]


                    produtos.append(fator1 * fator2)

                matriz_produto[j_m2].insert(im2,sum(produtos))

    except IndexError:
        print(f"\nDeu erro!\nCoordenada da falha: {im1,j_m1}")
        print(f"Resultado até aqui:\n {matriz_produto}")

    except KeyboardInterrupt:
        print(f"TimeOut!\nCoordenada da falha: {im1, j_m1}")
        print(f"Tamanho:\n {len(matriz_produto[j_m1])}")
        print(f"Resultado até aqui:\n {dimensionar_matriz(matriz_produto)}")
        # print(f"Resultado até aqui:\n {matriz_transposta}")

    print(f"Matriz produto: {matriz_produto}")

    return matriz_produto

def multiplicar_matriz_vetor(matriz, vetor):
    # O sinal de != (diferente) libera a passagem quando os tamanhos são iguais
    if len(matriz[0]) != len(vetor):
        return "O número de colunas da matriz1 deve ser igual ao número de linhas da matriz2!"

    resultado = []
    for linha in matriz:
        soma = sum(a * b for a, b in zip(linha, vetor))
        resultado.append(soma)
        
    return resultado

def diagonal_matriz(matriz):

    diagonal = [[],[]]

    tamanho_i, tamanho_j = dimensionar_matriz(matriz)
    
    for coordenada_diagonal in range(min(tamanho_i[-1], tamanho_j[-1])+1):
        
        diagonal[0].append(coordenada_diagonal)
        diagonal[1].append(coordenada_diagonal)       

    return diagonal 

def transformar_em_triangular(matriz):
    # n aqui representa o número de variáveis (colunas)
    n_colunas = len(matriz) 
    
    # Criamos a cópia mantendo sua estrutura de colunas
    A = [[float(x) for x in coluna] for coluna in matriz]
    
    # n_linhas é o número de observações em cada coluna
    n_linhas = len(A[0])

    # O escalonamento deve ser feito sobre as observações (linhas)
    for j in range(n_colunas):
        # O pivô está na coluna j, linha j
        pivo = A[j][j]
        
        if pivo == 0:
            continue

        for i in range(j + 1, n_linhas):
            # O fator busca zerar o elemento na linha i usando a linha j
            # Na sua estrutura: A[coluna][linha]
            fator = A[j][i] / pivo
            
            # Atualiza todas as colunas para aquela linha i
            for k in range(j, n_colunas):
                A[k][i] -= fator * A[k][j]
                
    return A

def multiplicar_diagonal(matriz_diagonal):

    coordenadas_diagonal = diagonal_matriz(matriz_diagonal)

    soma = [1]

    for coluna, linha in zip(coordenadas_diagonal[0],coordenadas_diagonal[1]):
        
        soma[0] = (soma[0] * matriz_diagonal[coluna][linha])

    return soma[0]

def determinante_2x2(matriz_menor):
    """Calcula o determinante de uma matriz 2x2: (a*d) - (b*c)"""
    a = matriz_menor[0][0]
    b = matriz_menor[0][1]
    c = matriz_menor[1][0]
    d = matriz_menor[1][1]
    return (a * d) - (b * c)

def matriz_cofatores(matriz):
    """Recorta a matriz 3x3, calcula os menores e aplica a regra de sinal"""
    n = len(matriz)
    cofatores = []
    
    for i in range(n):
        linha_cofatores = []
        
        for j in range(n):
            # 1. O Recorte (O "esconde-esconde")
            submatriz = []
            for linha_idx in range(n):
                # Se a linha atual não for a linha que queremos esconder (i)
                if linha_idx != i:
                    # Usamos slicing para pegar tudo antes da coluna 'j' e tudo depois da coluna 'j'
                    nova_linha = matriz[linha_idx][:j] + matriz[linha_idx][j+1:]
                    submatriz.append(nova_linha)
            
            # 2. Calcular o determinante da 2x2 que sobrou
            menor = determinante_2x2(submatriz)
            
            # 3. Regra do Tabuleiro de Xadrez (Sinal)
            # Se (i + j) for par, (-1) elevado a par é positivo (1)
            # Se (i + j) for ímpar, (-1) elevado a ímpar é negativo (-1)
            sinal = (-1) ** (i + j)
            cofator = sinal * menor
            
            # Guarda o resultado na nova linha
            linha_cofatores.append(cofator)
            
        # Guarda a linha completa na matriz de cofatores
        cofatores.append(linha_cofatores)
        
    return cofatores

def matriz_adjunta(matriz):
    """A matriz adjunta é simplesmente a transposta da matriz de cofatores"""
    cofatores = matriz_cofatores(matriz)
    
    # Reaproveitando a função que você já tem no functions.py!
    adjunta = transpor_matriz(cofatores) 
    
    return adjunta

def primeiro_fator(determinante, matriz_adjunta):
    det_inverso = (1/determinante)

    primeiro_fator = [[det_inverso * elemento for elemento in linha] for linha in matriz_adjunta]

    return primeiro_fator

