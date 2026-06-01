# -*- coding: utf-8 -*-
"""
Script de Teste: Regressão Linear Múltipla (teste2.py)
"""
import algebra as alg
import machine_learning as ml

# Vetores de dados
intercepto = [1, 1, 1, 1, 1]
horas = [2, 3, 5, 7, 8]
qi = [100, 90, 105, 110, 115]
notas = [5, 7, 8, 10, 12]

print("=== TESTE DE REGRESSÃO LINEAR MÚLTIPLA ===")

# 1. Criação da matriz de design transposta (representada por colunas)
X_transposta = alg.criar_matriz(intercepto, horas, qi)
X_normal = alg.transpor_matriz(X_transposta)

print("\n1. Estrutura Matricial:")
print(f"  Colunas de X^T (3 colunas, tamanho 5): {X_transposta}")
print(f"  Colunas de X   (5 colunas, tamanho 3): {X_normal}")

# 2. Covariância XtX
XtX = alg.multiplicar_matrizes(X_normal, X_transposta)
print(f"\n2. Matriz de Covariância XtX (X^T * X):\n  {XtX}")

# 3. Determinante de XtX
triangular = alg.transformar_em_triangular(XtX)
determinante = alg.multiplicar_diagonal(triangular)
print(f"\n3. Determinante de XtX: {determinante:.6f}")

# 4. Inversa de XtX
inversa = alg.inverter_matriz(XtX)
print(f"\n4. Matriz Inversa (XtX)^-1:\n  {inversa}")

# 5. Ajuste do Modelo de Regressão Múltipla
modelo = ml.RegressaoLinearMultipla().fit(X_transposta, notas)
print(f"\n5. Coeficientes Ajustados (Beta):\n  {modelo.coeficientes}")

# 6. Previsões e Métricas
previsoes = modelo.predict(X_transposta)
metricas = ml.rmse(notas, previsoes)
print(f"\n6. Previsões: {previsoes}")
print("   Métricas de Ajuste:")
for k, v in metricas.items():
    print(f"     - {k}: {v:.6f}")
