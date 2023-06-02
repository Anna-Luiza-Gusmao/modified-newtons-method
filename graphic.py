import matplotlib.pyplot as plt


def curve_convergence(interacoes, funcao):
    plt.figure(num='Convergência')
    plt.plot(interacoes, funcao, linestyle='-', color='blue', marker='o')
    plt.xlabel('Interações')
    plt.ylabel('f(x,y)')
    plt.title('Curva de Convergência')
    plt.grid(True, linestyle=':', linewidth=0.7, alpha=0.5)
