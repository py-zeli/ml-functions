import functions

intercepto = [1, 1, 1, 1, 1]
horas = [2, 3, 5, 7, 8]
qi = [100, 90, 105, 110, 115]

matriz = functions.criar_matriz(intercepto, horas, qi)

print(f" \nMatriz original: \n{matriz}\n")
# print(functions.dimensionar_matriz(matriz))

matriz_transposta = functions.transpor_matriz(matriz)
print(f" \nMatriz Transposta: \n{matriz_transposta}\n")

# print(matriz_transposta)
functions.multiplicar_matrizes(matriz, matriz_transposta)