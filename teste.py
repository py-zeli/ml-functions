import functions

x = [2, 3, 5, 7, 8]
y = [5, 7, 8, 10, 12]

# print(functions.media(x))

coef_linear, coef_angular = functions.minimos_quadrados(x,y)

y_pred = functions.realizar_previsao(x, coef_linear, coef_angular)

print(functions.raiz_soma_erros_quadrados(y, y_pred))

# print(functions.soma_erros_quadrados(y, y_pred))

