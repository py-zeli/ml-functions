import functions

intercepto = [1, 1, 1, 1, 1]
horas = [2, 3, 5, 7, 8]
qi = [100, 90, 105, 110, 115]

notas = [5, 7, 8, 10, 12]

matriz = functions.criar_matriz(intercepto, horas, qi)

matriz_transposta = functions.transpor_matriz(matriz)

# print(f" \nMatriz Transposta: \n{matriz_transposta}\n")


produto_matrizes = functions.multiplicar_matrizes(matriz, matriz_transposta)

matriz_diagonal = functions.transformar_em_triangular(produto_matrizes)

determinante = functions.multiplicar_diagonal(matriz_diagonal)

print(determinante)

matriz_adjunta = functions.matriz_adjunta(produto_matrizes)


primeiro_fator = functions.primeiro_fator(determinante, matriz_adjunta)
# Trocamos 'matriz_transposta' por 'matriz'
produto_matriz_vetor = functions.multiplicar_matriz_vetor(matriz, notas)

print(primeiro_fator)

print(produto_matriz_vetor)

print(functions.multiplicar_matriz_vetor(primeiro_fator, produto_matriz_vetor))