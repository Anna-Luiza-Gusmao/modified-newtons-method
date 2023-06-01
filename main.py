import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sympify, diff, lambdify


def user_function(vetor_da_variaveis):
    global funcao
    # Cria uma expressão simbólica da função a partir da string fornecida pelo usuário
    new_function = sympify(funcao)
    new_function = new_function.subs(x, vetor_da_variaveis[0]).subs(y, vetor_da_variaveis[1])
    print("Funçao", new_function)
    return new_function


def newton_modificado(v0, epsilon, user_function):
    global funcao
    x, y = symbols('x y')
    xk = v0

    max_iter = 1000
    num_iters = []
    x_vals = []
    y_vals = []

    # Calcula o gradiente da função
    def gradient(x_val, y_val):
        grad_x = diff(funcao, x).subs(x, x_val).subs(y, y_val)
        grad_y = diff(funcao, y).subs(x, x_val).subs(y, y_val)
        return np.array([grad_x, grad_y], dtype=np.float64)

    # Calcule as derivadas parciais de segunda ordem
    d2f_dx2 = diff(diff(funcao, x), x)
    d2f_dy2 = diff(diff(funcao, y), y)
    d2f_dxdy = diff(diff(funcao, x), y)

    # Converta as expressões simbólicas em funções numéricas
    d2f_dx2_func = lambdify((x, y), d2f_dx2)
    d2f_dy2_func = lambdify((x, y), d2f_dy2)
    d2f_dxdy_func = lambdify((x, y), d2f_dxdy)

    # Calcula os valores numéricos das derivadas parciais de segunda ordem e a matriz hessiana
    def calculate_hessian(x_val, y_val):
        d2f_dx2_val = d2f_dx2_func(x_val, y_val)
        d2f_dy2_val = d2f_dy2_func(x_val, y_val)
        d2f_dxdy_val = d2f_dxdy_func(x_val, y_val)
        hessian = np.array([[d2f_dx2_val, d2f_dxdy_val], [d2f_dxdy_val, d2f_dy2_val]], dtype=np.float64)
        return hessian

    for i in range(max_iter):
        x_vals.append(xk[0])
        y_vals.append(xk[1])
        max_values = np.max(xk, axis=0)
        min_values = np.min(xk, axis=0)
        delta = max_values - min_values

        g = gradient(xk[0], xk[1])

        # Critério de Parada do Gradiente
        if np.linalg.norm(g) < epsilon:
            print("Critério de parada do gradiente atingido")
            break

        # Calcule a inversa da Hessiana
        hessiana_inversa = np.linalg.inv(calculate_hessian(xk[0], xk[1]))

        dk = -np.dot(hessiana_inversa, g)

        alpha = 1
        while user_function(xk + alpha * dk) > user_function(xk) + 0.1 * alpha * np.dot(g, dk):
            alpha *= 0.5
        ft_alpha = alpha * dk

        x_new = xk + ft_alpha
        xk = x_new

        # Critério de Parada das Variáveis
        if i >= 5:
            sigma = np.max(xk[i - 5:]) - np.min(xk[i - 5:])
            if sigma < 0.001 * delta:
                print("Critério de parada para as variáveis atingido")
                break

        num_iters.append(i + 1)

    return xk, user_function(xk), len(num_iters)


# Define os símbolos das variáveis independentes
x, y = symbols('x y')

# Entrada das informações do usuário
funcao = input("Insira a função que deseja otimizar: ")

# Solicita os valores das variáveis x e y
x0 = float(input("Insira o x inicial: "))
y0 = float(input("Insira o y inicial: "))

# A variável inicial recebe os valores do usuário
v0 = np.array([x0, y0])

epsilon = 1e-6

# Chama o método de Newton Modificado
x_opt, f_opt, num_iter = newton_modificado(v0, epsilon, user_function)

print(f"Solução ótima encontrada: x = {x_opt}, f(x) = {f_opt}, número de interações = {num_iter}")
