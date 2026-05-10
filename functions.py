"""REGRESSÃO LINEAR SIMPLES"""
def media(vetor):
    return sum(vetor)/len(vetor)

def minimos_quadrados(x: list,y: list):
    x_media = media(x)
    y_media = media(y)

    x_diferenca_media = [(xi - x_media) for xi in x]
    
    y_diferenca_media = [(yi - y_media) for yi in y]

    x_dif_quadrado = [(xi ** 2) for xi in x_diferenca_media]

    soma_x_dif_quadrado = sum(x_dif_quadrado)
    
    produto_dif_xy = [x_dif_i * y_dif_i for x_dif_i, y_dif_i in zip(x_diferenca_media, y_diferenca_media)]

    soma_prod_dif_xy = sum(produto_dif_xy)

    coef_angular = (soma_prod_dif_xy/soma_x_dif_quadrado) # Calcular o coeficiente angular

    coef_linear = y_media - (coef_angular * x_media) # Calcular o intercepto

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
        
    y_erro = [yi - y_pred_i for yi, y_pred_i in zip(y, y_pred) ]

    y_erro_quadrado = [y_erro_i ** 2 for y_erro_i in y_erro ]

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

def gradient_descent(x, y, taxa_de_aprendizado, coef_linear = 0, coef_angular = 0):

    tamanho_vetor = len(x)

    coef_linear = [coef_linear] * tamanho_vetor

    coef_angular = [coef_angular] * tamanho_vetor

    y_pred = [coef_angular * xi + coef_linear for coef_angular, xi, coef_linear in zip(coef_angular, x, coef_linear)]

    y_erro = calcular_erro(y, y_pred)

    soma_erro = sum(y_erro)

    prod_erro_xi = [y_erro_i * xi for y_erro_i, xi in zip(y_erro, x)]

    soma_prod_erro_xi = sum(prod_erro_xi)

    gradiente_coef_angular = (-2 / tamanho_vetor) * soma_prod_erro_xi

    gradiente_coef_linear = (-2 / tamanho_vetor) * soma_erro

    novo_coef_linear = coef_linear[0] - (taxa_de_aprendizado * gradiente_coef_linear)

    novo_coef_angular = coef_angular[0] - (taxa_de_aprendizado * gradiente_coef_angular)

    valores = {
        "antigo_coef_linear": coef_linear[0],
        "novo_coef_linear": novo_coef_linear,
        "antigo_coef_angular": coef_angular[0],
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

        for j in j_tamanho: # range(0,2)
                
            for i in i_tamanho: # range(0,5)

                while len(matriz_transposta) < j_tamanho[-1]+1:
                    matriz_transposta.append([])
                
                # while len(matriz_transposta[j]) < j_tamanho[-1]-4:
                #     matriz_transposta[j].append(None)

                matriz_transposta[j].insert(i,matriz[i][j])
                
        return matriz_transposta

    except IndexError:
        print(f"\nDeu erro!\nCoordenada da falha: {i,j}")
        print(f"Resultado até aqui:\n {matriz_transposta}")

    
    except KeyboardInterrupt:
        print(f"TimeOut!\nCoordenada da falha: {i,j}")
        print(f"Tamanho:\n {len(matriz_transposta[j])}")
        print(f"Resultado até aqui:\n {dimensionar_matriz(matriz_transposta)}")
        # print(f"Resultado até aqui:\n {matriz_transposta}")



    # for i in i_tamanho:
    #     print(matriz[j_tamanho[-1]][i])

    # matriz_transposta = [matriz[j][i] for matriz[i][j] in matriz]

def multiplicar_matrizes(matriz1, matriz2):

    if len(matriz1) == len(matriz2):
        return "O número de colunas da matriz1 deve igual ao número de linhas da matriz2!"

    matriz_produto = []

    i_m1_range, j_m1_range = dimensionar_matriz(matriz1)
    i_m2_range, j_m2_range = dimensionar_matriz(matriz2)

    # print(i_m1_range)
    # print(j_m1_range)
    # print(i_m2_range)
    # print(j_m2_range)

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

                    # print(f"{fator1}x{fator2}={fator1 * fator2}")

                    # matriz_produto[im1].insert(matriz1[im1][j_m1] * matriz2[j_m2][im2],matriz1[im1][j_m2])
                    # print()
                
                # print(produtos)

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
