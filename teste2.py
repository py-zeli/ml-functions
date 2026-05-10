import functions

horas = [2, 3, 5, 7, 8]
qi = [100, 90, 105, 110, 115]
ano = [5, 6, 7, 8, 9]
idade = [10, 12, 14, 16, 18]

matriz = functions.criar_matriz(horas, qi, ano, idade)

print(matriz)
# print(functions.dimensionar_matriz(matriz))

print(functions.transpor_matriz(matriz))