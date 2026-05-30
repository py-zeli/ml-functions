def media(vetor):

    return float(sum(vetor)/len(vetor))

def vetor_expoente(vetor, n):

    return [i ** n for i in vetor]

def produto_vetor(vetor_a, vetor_b):

    return [vetor_a_i * vetor_b_i for vetor_a_i, vetor_b_i in zip(vetor_a, vetor_b)]

def variancia(vetor: list[float]) -> float:
    """Calcula a variância amostral de um conjunto de dados."""
    m = media(vetor) 
    list_variancia = [(i - m)**2 for i in vetor]
    return sum(list_variancia) / len(vetor)


def desvio(vetor, m=None):
    if m is None:
        m = media(vetor)
    # Retorna um novo vetor contendo o resíduo de cada escalar do vetor original
    return [(i - m) for i in vetor]

def desvio_padrao(vetor):

    # Retorna outro escalar
    return variancia(vetor) ** 0.5


def coeficiente_pearson(vetor_x, vetor_y):

    n = len(vetor_x)

    soma_x = sum(vetor_x)

    soma_y = sum(vetor_y)
    
    soma_x2 = sum(xi ** 2 for xi in vetor_x)
    
    soma_y2 = sum(yi ** 2 for yi in vetor_y)
    
    soma_xy = sum(i * j for i, j in zip(vetor_x, vetor_y))

    numerador = (n * soma_xy) - (soma_x * soma_y)

    denominador = ((n * soma_x2 - soma_x ** 2) * (n * soma_y2 - soma_y ** 2)) ** (1/2)

    return numerador / denominador


def coeficiente_angular(vetor_x, vetor_y):

    return (coeficiente_pearson(vetor_x, vetor_y) * desvio_padrao(vetor_y)) / desvio_padrao(vetor_x)

# Dados do teste
horas_estudo = [2, 4, 6]
notas_prova  = [5, 7, 9]

print(coeficiente_angular(horas_estudo, notas_prova))


def sigmoide(z):

    EULER = 2.718281828459045
    
    return 1 / (1 + (EULER ** (-z)))

def predizer_probabilidade(x, peso, bias):
    """Calcula o logit (z) e retorna a probabilidade (y_predito)"""
    z = (x * peso) + bias
    
    return sigmoide(z)