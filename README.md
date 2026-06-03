# 🧠 ml-functions (Machine Learning & Álgebra do Zero)

Bem-vindo ao **ml-functions**! Este repositório é uma suíte educacional completa de algoritmos de **Machine Learning (Modelagem Preditiva)**, **Álgebra Linear Matricial** e **Estatística/Probabilidade**, implementados **100% do zero** em Python, sem o uso de bibliotecas de alto nível (como *Scikit-Learn* ou *SciPy*).

O objetivo principal deste projeto é mergulhar na matemática profunda que rege as ciências de dados, expondo a beleza geométrica e analítica de cada cálculo e distribuição.

---

## 📂 Organização Didática do Projeto

Para tornar a experiência prática e de aprendizado extremamente fluida e eficiente ("Slim & Play"), o repositório está organizado em uma trilha sequencial e modular na raiz:

```text
ml-functions/
├── .gitignore
├── README.md                          # Este guia didático e matemático
│
│   # 🛠️ Camada de Bibliotecas de Produção (Módulos Locais)
├── algebra.py                         # Operações vetoriais, sigmoide estável e matrizes por colunas
├── estatistica.py                     # Médias, desvios, covariância amostral e correlações
├── probabilidades.py                  # Binomial, Poisson e Normal (erf) do zero
├── machine_learning.py                # Regressão Simples (OLS, GD), Múltipla (Normal) e Logística
│
│   # 📓 Trilha de Aprendizado (Notebooks Jupyter)
├── Notebooks/
│   ├── 01_explicacao_algoritmos.ipynb     # Guia didático das libs locais e plots das distribuições
│   ├── 02_regressao_linear_simples.ipynb  # Regressão Simples (OLS e GD com plots)
│   ├── 03_regressao_linear_multipla.ipynb # Regressão Múltipla (Matrizes e Equação Normal)
│   └── 04_regressao_logistica.ipynb       # Aplicação prática da Regressão Logística e métricas do zero
│
│   # 🧪 Scripts Rápidos de Validação e Teste
└── Testes/
    ├── teste_regressao_simples.py         # Valida o ajuste analítico (MQO) vs. numérico (GD)
    └── teste_regressao_multipla.py        # Valida a inversão de matriz e regressão múltipla
```

---

## 🧮 A Matemática por Trás das Camadas

### 1. Camada de Álgebra Linear (`algebra.py`)
Centraliza as operações primitivas sobre vetores e matrizes na convenção de **lista de colunas** (ideal para processamento vetorial).
* **Ativação Sigmoide Estável:**
  $$\sigma(z) = \frac{1}{1 + e^{-\max(0, -z)}} \cdot \text{ou} \cdot \frac{e^{\min(0, z)}}{1 + e^{\min(0, z)}}$$
* **Transposição de Matriz:** Permutação limpa usando desembrulho de coleções em Python (`zip(*matriz)`).
* **Inversão de Matrizes (via Adjunta e Cofatores):**
  $$A^{-1} = \frac{1}{\det(A)} \text{adj}(A)$$

### 2. Camada Estatística (`estatistica.py`)
Métricas fundamentais com correções amostrais rigorosas para evitar viés estatístico.
* **Variância Amostral (com Correção de Bessel):**
  $$s^2 = \frac{\sum (x_i - \bar{x})^2}{n - 1}$$
* **Covariância Amostral:**
  $$\text{Cov}(X, Y) = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{n - 1}$$
* **Coeficiente de Correlação de Pearson ($r$):**
  $$r = \frac{n \sum x_i y_i - \sum x_i \sum y_i}{\sqrt{[n \sum x_i^2 - (\sum x_i)^2][n \sum y_i^2 - (\sum y_i)^2]}}$$

### 3. Camada de Probabilidades (`probabilidades.py`)
Implementa distribuições de probabilidade clássicas integrando recursos matemáticos estáveis.
* **Distribuição Binomial (PMF & CDF):**
  $$P(X = k) = \binom{n}{k} p^k (1 - p)^{n - k}$$
* **Distribuição de Poisson (PMF & CDF):**
  $$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$
* **Distribuição Normal (CDF via Função Erro Analítica `erf`):**
  $$P(X \le x) = 0.5 \cdot \left[ 1 + \text{erf}\left( \frac{x - \mu}{\sigma\sqrt{2}} \right) \right]$$

### 4. Camada de Modelos de ML (`machine_learning.py`)
Implementa os algoritmos clássicos de regressão e calibração de pesos de forma interpretável.
* **Mínimos Quadrados Ordinários (OLS):**
  $$\beta_1 = \frac{\text{Cov}(X, Y)}{\text{Var}(X)} \quad \text{e} \quad \beta_0 = \bar{y} - \beta_1\bar{x}$$
* **Equação Normal Matricial para Regressão Múltipla:**
  $$\beta = (X^T X)^{-1} X^T y$$
* **Gradiente Descendente (Regressão Simples):**
  $$b \leftarrow b - \alpha \left(-\frac{2}{n} \sum (y_i - y_{\text{pred}})\right)$$
  $$a \leftarrow a - \alpha \left(-\frac{2}{n} \sum (y_i - y_{\text{pred}})x_i\right)$$

---

## 🚀 Como Executar e Validar

### Requisitos
Basta ter o Python instalado na máquina (versão 3.8+ recomendada) e a biblioteca gráfica `matplotlib` e `seaborn` para visualizações nos notebooks (opcional).

### Executando os Testes via Terminal
Valide o funcionamento dos modelos com comandos simples:

```bash
# Executa o teste de Regressão Simples (MQO vs. Gradiente Descendente)
python teste_regressao_simples.py

# Executa o teste de Regressão Múltipla (Inversão de Matriz e Equação Normal)
python teste_regressao_multipla.py
```

### Explorando a Trilha de Aprendizagem
Abra o Jupyter Notebook ou o VS Code e percorra os arquivos numerados:
1. `01_explicacao_algoritmos.ipynb`: Conheça didaticamente os conceitos e veja os plots de densidade das distribuições.
2. `02_regressao_linear_simples.ipynb`: Veja o ajuste analítico por MQO, o Gradiente Descendente e sua convergência com anotações de progresso e distância ao ideal.
3. `03_regressao_linear_multipla.ipynb`: Explore a matemática complexa de matrizes (cofator, adjunta, determinante) e a inversão do zero pela Equação Normal.