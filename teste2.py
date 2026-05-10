import functions

intercepto = [1, 1, 1, 1, 1]
horas = [2, 3, 5, 7, 8]
qi = [100, 90, 105, 110, 115]

notas = [5, 7, 8, 10, 12]

matriz = functions.criar_matriz(intercepto, horas, qi)

matriz_transposta = functions.transpor_matriz(matriz)
# print(f" \nMatriz Transposta: \n{matriz_transposta}\n")

# functions.multiplicar_matrizes(matriz, matriz_transposta)

print(functions.multiplicar_matriz_vetor(matriz_transposta, notas))