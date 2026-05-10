import functions_test

x = [2, 3, 5, 7, 8]
y = [5, 7, 8, 10, 12]

coef_angular, coef_linear = functions_test.minimos_quadrados(x,y)

y_pred = functions_test.realizar_previsao(x, coef_angular, coef_linear)

print(f"Coeficiente Linear: {coef_linear}\nCoeficiente Angular: {coef_angular}\n")


print(functions_test.rmse(y, y_pred))

# print(functions.gradient_descent(x, y, 0.01))

# for i in range(5):

#     if 'novo_coef_linear' in locals():
#         valores_calibrados = functions.gradient_descent(x, y, 0.01, novo_coef_linear, novo_coef_angular)
    
#     else:
#         valores_calibrados = functions.gradient_descent(x, y, 0.01)
    

#     print(f"Resultados da {i+1} iteração:\n{valores_calibrados}\n")
#     # # valores_calibrados = print(functions.gradient_descent(x, y, 0.01))
#     novo_coef_linear = valores_calibrados['novo_coef_linear']
#     novo_coef_angular = valores_calibrados['novo_coef_angular']