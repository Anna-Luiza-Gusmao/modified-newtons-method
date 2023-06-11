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

    plt.figure(num='Função Objetivo em 3D', figsize=(24, 16), dpi=130)
    ax = plt.subplot2grid((7, 7), (0, 0), rowspan=6, colspan=6, projection='3d')
    plt.style.use('Solarize_Light2')

    x = np.linspace(-6, 6, 100)
    y = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(x, y)
    Z = function_aux(funcao, symbol, X, Y)

    ax.plot_wireframe(X, Y, Z, color='green')

    ax.set_xlabel('x', fontsize=10, color='gray')
    ax.set_ylabel('y', fontsize=10, color='gray')
    ax.set_zlabel('z', fontsize=10, color='gray')

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('Função em 3D', fontsize=8)


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