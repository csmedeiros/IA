from rbm import RBM
import numpy as np

rbm = RBM(num_visible=6, num_hidden=2)

base = np.array([[1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 0, 1]])

filmes = ["A Bruxa", "Invocacao do Mal", "O Chamado", "Se Beber Nao Case", "Gente Grande", "American Pie"]

rbm.train(base, max_epochs=5000)
rbm.weights

usuario1 = np.array([[1, 1, 0, 1, 0, 0]])
usuario2 = np.array([[0, 0, 0, 1, 1, 0]])

camada_escondida = np.array([[1, 0]])
recomendacao = rbm.run_hidden(camada_escondida)

rbm.run_visible(usuario1)

for i in range(len(usuario1[0])):
    if usuario1[0, i] == 0 and recomendacao[0, i] == 1:
        print(filmes[i])