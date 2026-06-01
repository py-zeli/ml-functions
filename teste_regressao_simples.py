# -*- coding: utf-8 -*-
"""
Script de Teste: Regressão Linear Simples (teste.py)
"""
import machine_learning as ml
import estatistica as est

# Dados de Teste
x = [2, 3, 5, 7, 8]
y = [5, 7, 8, 10, 12]

print("=== TESTE DE REGRESSÃO LINEAR SIMPLES ===")
print("Dados de Entrada:")
print("x:", x)
print("y:", y)
print(f"Média de x: {est.media(x):.4f}")
print(f"Variância de x: {est.variancia(x):.4f}")
print(f"Covariância entre x e y: {est.covariancia(x, y):.4f}")

# 1. Ajuste Analítico via Mínimos Quadrados (MQO)
modelo_mqo = ml.RegressaoLinear().fit(x, y, metodo="mqo")
print("\n1. Método Analítico (MQO):")
print(f"  Coeficiente Linear (b): {modelo_mqo.coef_linear:.6f}")
print(f"  Coeficiente Angular (a): {modelo_mqo.coef_angular:.6f}")

y_pred_mqo = modelo_mqo.predict(x)
metricas_mqo = ml.rmse(y, y_pred_mqo)
print("  Métricas de Erro:")
for k, v in metricas_mqo.items():
    print(f"    - {k}: {v:.6f}")

# 2. Ajuste Numérico via Gradiente Descendente (GD)
modelo_gd = ml.RegressaoLinear().fit(x, y, metodo="gd", taxa_aprendizado=0.01, epocas=3000)
print("\n2. Método Numérico (Gradiente Descendente):")
print(f"  Coeficiente Linear (b): {modelo_gd.coef_linear:.6f}")
print(f"  Coeficiente Angular (a): {modelo_gd.coef_angular:.6f}")

y_pred_gd = modelo_gd.predict(x)
metricas_gd = ml.rmse(y, y_pred_gd)
print("  Métricas de Erro:")
for k, v in metricas_gd.items():
    print(f"    - {k}: {v:.6f}")
