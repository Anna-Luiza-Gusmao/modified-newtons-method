import matplotlib.pyplot as plt
import numpy as np
import operator
from sympy import symbols, sympify


def convergence_curve(interacoes, funcao):
    plt.figure(num='Convergência')
    plt.plot(interacoes, funcao, linestyle='-', color='blue', marker='o')
    plt.xlabel('Interações')
    plt.ylabel('f(x,y)')
    plt.title('Curva de Convergência')
    plt.grid(True, linestyle=':', linewidth=0.7, alpha=0.5)


def function_aux(funcao, symbol, x, y):
    expr = sympify(funcao)
    l = operator.length_hint(x)
    f = np.zeros((l, l), dtype=np.float64)

    for i in range(0, l):
        for j in range(0, l):
            f[i][j] = expr.subs([(symbol[0], x[i][j]), (symbol[1], y[i][j])])

    return f


def function_graph(funcao):
    symbol = symbols('x y')

    plt.figure(num='Função Objetivo em 3D', figsize=(8, 6), dpi=80)
    ax = plt.axes(projection='3d')

    x = np.linspace(-6, 6, 100)
    y = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(x, y)
    Z = function_aux(funcao, symbol, X, Y)

    ax.set_xlabel('x', fontsize=10, color='gray')
    ax.set_ylabel('y', fontsize=10, color='gray')
    ax.set_zlabel('z', fontsize=10, color='gray')

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    ax.set_title('Função em 3D', fontsize=8)


def plot_path(x, y):
    for i in range(1, len(x)):
        plt.annotate('', xy=(x[i], y[i]), xytext=(x[i - 1], y[i - 1]),
                     arrowprops=dict(facecolor='yellow', width=0.5, headwidth=1.5))


def contour_lines(funcao):
    symbol = symbols('x y')

    plt.figure(num='Curvas de Nível')

    x = np.linspace(-6, 6, 100)
    y = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(x, y)

    Z = function_aux(funcao, symbol, X, Y)

    levels = np.linspace(-100, 100, 200)  # Intervalos personalizados para os níveis de contorno

    plt.contour(X, Y, Z, levels=levels, cmap='viridis')
    plt.colorbar()

    plt.xlabel('x')
    plt.ylabel('y')


def contour_lines_with_steps(funcao, arrayX, arrayY, ponto_otimo):
    symbol = symbols('x y')

    plt.figure(num='Curvas de Nível com Deslocamento')

    x = np.linspace(-6, 6, 100)
    y = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(x, y)

    Z = function_aux(funcao, symbol, X, Y)

    levels = np.linspace(-100, 100, 200)  # Intervalos personalizados para os níveis de contorno

    plt.contour(X, Y, Z, levels=levels, cmap='viridis')
    plot_path(arrayX, arrayY)

    plt.scatter(ponto_otimo[0], ponto_otimo[1], c='red', marker='*', s=100)

    # Cálculo dos limites adequados
    lim_x_min = min(arrayX)
    lim_x_max = max(arrayX)
    lim_y_min = min(arrayY)
    lim_y_max = max(arrayY)

    margin = 1.0  # Margem adicional para os limites (opcional)
    lim_x_min -= margin
    lim_x_max += margin
    lim_y_min -= margin
    lim_y_max += margin

    # Definição dos limites dos eixos x e y
    plt.xlim(lim_x_min, lim_x_max)
    plt.ylim(lim_y_min, lim_y_max)

    plt.colorbar()

    plt.title(f"Ponto Ótimo: {ponto_otimo}", fontsize=10)
    plt.xlabel('x')
    plt.ylabel('y')


def doable_region(funcao, restricao, ponto_otimo):
    symbol = symbols('x y')

    plt.figure(num='Curvas de Nível com Restrições')

    x = np.linspace(-6, 6, 100)
    y = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(x, y)

    Z = function_aux(funcao, symbol, X, Y)

    levels = np.linspace(-100, 100, 200)  # Intervalos personalizados para os níveis de contorno

    plt.contour(X, Y, Z, levels=levels, cmap='viridis')
    plt.scatter(ponto_otimo[0], ponto_otimo[1], c='red', marker='.', s=20)

    plt.contour(X, Y, restricao(X, Y), [0], colors='r')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)


def function_graph_constraint(restricao):
    plt.figure(num='Função da Restrição em 3D', figsize=(8, 6), dpi=80)
    ax = plt.axes(projection='3d')

    x = np.linspace(-6, 6, 100)
    y = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(x, y)
    Z = restricao(X, Y)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='magma', edgecolor='none')

    ax.set_title('Restrição em 3D', fontsize=8)


def objetive_function_with_constraint(funcao, restricoes):
    symbol = symbols('x y')

    plt.figure(num='Função Objetivo + Função de Restrição em 3D', figsize=(8, 6), dpi=80)
    ax = plt.axes(projection='3d')

    x = np.linspace(-6, 6, 100)
    y = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(x, y)

    ax.set_xlabel('x', fontsize=10, color='gray')
    ax.set_ylabel('y', fontsize=10, color='gray')
    ax.set_zlabel('z', fontsize=10, color='gray')

    if len(restricoes) != 0:
        for restricao in restricoes:
            constraint_values = restricao(X, Y)
            Z = constraint_values + function_aux(funcao, symbol, X, Y)
            ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='plasma', edgecolor='none')

    ax.set_title('Funções em 3D', fontsize=8)