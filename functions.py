def media(vetor):
    return sum(vetor)/len(vetor)

def minimos_quadrados(x: list,y: list):
    x_media = media(x)
    y_media = media(y)

    x_diferenca_media = [(xi - x_media) for xi in x]
    
    y_diferenca_media = [(yi - y_media) for yi in y]

    x_dif_quadrado = [(xi ** 2) for xi in x_diferenca_media]

    soma_x_dif_quadrado = sum(x_dif_quadrado)   
    
    y_dif_quadrado = [(yi ** 2) for yi in y_diferenca_media]
    
    produto_dif_xy = [x_dif_i * y_dif_i for x_dif_i, y_dif_i in zip(x_diferenca_media, y_diferenca_media)]

    soma_prod_dif_xy = sum(produto_dif_xy)

    coef_angular = (soma_prod_dif_xy/soma_x_dif_quadrado) # Calcular o coeficiente angular

    coef_linear = y_media - (coef_angular * x_media) # Calcular o intercepto

    return coef_angular, coef_linear

def realizar_previsao(x, coef_linear, coef_angular):

    y_pred = [(coef_linear * xi) + coef_angular for xi in x]

    return y_pred

def raiz_soma_erros_quadrados(y, y_pred):

    n_y = len(y)
        
    y_erro = [yi - y_pred_i for yi, y_pred_i in zip(y, y_pred) ]

    y_erro_quadrado = [y_erro_i ** 2 for y_erro_i in y_erro ]

    soma_erros_quadrados = sum(y_erro_quadrado)

    mse = soma_erros_quadrados/n_y

    rmse = mse ** (1/2)

    return rmse, mse