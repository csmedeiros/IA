from rbm import RBM
import numpy as np

rbm = RBM(num_visible=6, num_hidden=3)

data = np.array([[0,1,1,1,0,1],
                 [1,1,0,1,1,1],
                 [0,1,0,1,0,1],
                 [0,1,1,1,0,1], 
                 [1,1,0,1,0,1],
                 [1,1,0,1,1,1]])

filmes = ["Freddy x Jason", "O Ultimato Bourne", "Star Trek", "Exterminador Do Futuro", "Norbit", "Star Wars"]

rbm.train(data, max_epochs=5000)

usuario = np.array([[0,1,0,1,0,0]])

camada_escondida = rbm.run_visible(usuario)

recomendacao = rbm.run_hidden(camada_escondida)

for i in range(len(usuario[0])):
    if usuario[0, i]==0 and recomendacao[0, i]==1:
        print(filmes[i])