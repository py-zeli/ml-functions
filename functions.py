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

def gradient_descent(x, y, taxa_de_aprendizado):

    tamanho_vetor = len(x)

    coef_linear = [0] * tamanho_vetor

    coef_angular = [0] * tamanho_vetor

    y_pred = [coef_angular * xi + coef_linear for coef_angular, xi, coef_linear in zip(coef_angular, x, coef_linear)]

    y_erro = calcular_erro(y, y_pred)

    soma_erro = sum(y_erro)

    prod_erro_xi = [y_erro_i * xi for y_erro_i, xi in zip(y_erro, x)]

    soma_prod_erro_xi = sum(prod_erro_xi)

    gradiente_coef_linear = (-2 / tamanho_vetor) * soma_prod_erro_xi

    gradiente_coef_angular = (-2 / tamanho_vetor) * soma_erro

    novo_coef_linear = coef_linear[0] - (taxa_de_aprendizado * gradiente_coef_linear)

    novo_coef_angular = coef_angular[0] - (taxa_de_aprendizado * gradiente_coef_angular)

    valores = {
        "antigo_coef_linear": coef_linear[0],
        "antigo_coef_angular": coef_angular[0],
        "novo_coef_linear": novo_coef_linear,
        "novo_coef_angular": novo_coef_angular
    }

    return valores