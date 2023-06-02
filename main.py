import numpy as np
from sympy import symbols, sympify, diff, lambdify
import math
import graphic as graphics_solution
import matplotlib.pyplot as plt

def calculate_function(vetor_da_variaveis):
    global funcao
    # Cria uma expressão simbólica da função a partir da string fornecida pelo usuário
    new_function = sympify(funcao)
    new_function = new_function.subs(x, vetor_da_variaveis[0]).subs(y, vetor_da_variaveis[1])
    return new_function


def newton_modificado(v0, user_function):
    global funcao
    x, y = symbols('x y')
    xk = v0

    max_iter = 1000
    num_iters = []
    function_values = []
    x_vals = []
    y_vals = []
    modulo_do_vetor = []

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
        modulo_do_vetor.append(math.sqrt(xk[0] + xk[1]))

        max_values = np.max(modulo_do_vetor, axis=0)
        min_values = np.min(modulo_do_vetor, axis=0)
        delta = max_values - min_values

        g = gradient(xk[0], xk[1])

        # Cálculo da inversa da Hessiana
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
            sigma = np.max(modulo_do_vetor[i - 5:]) - np.min(modulo_do_vetor[i - 5:])
            if sigma < 0.001 * delta:
                print("Critério de parada para as variáveis atingido")
                break

        num_iters.append(i + 1)
        function_values.append(user_function(xk))

    # Plot Curva de Convergência
    graphics_solution.curve_convergence(num_iters, function_values)

    return xk, user_function(xk), len(num_iters)


# Define os símbolos das variáveis independentes
x, y = symbols('x y')

# Entrada das informações do usuário
funcao = input("Insira a função que deseja otimizar: ")

# Solicita os valores das variáveis x e y
x0 = float(input("Insira o x inicial: "))
y0 = float(input("Insira o y inicial: "))

# A variável inicial recebe os valores do usuário
variavel_inicial = np.array([x0, y0])

# Chama o método de Newton Modificado
x_opt, f_opt, num_iter = newton_modificado(variavel_inicial, calculate_function)

print(f"Solução ótima encontrada: x = {x_opt}, f(x) = {f_opt}, número de interações = {num_iter}")
plt.show()
