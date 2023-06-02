import matplotlib.pyplot as plt


def curve_convergence(interacoes, funcao):
    plt.plot(interacoes, funcao)
    plt.xlabel('Interações')
    plt.ylabel('Função')
    plt.title('Curva de Convergência')
    plt.grid(True)
    plt.show()
