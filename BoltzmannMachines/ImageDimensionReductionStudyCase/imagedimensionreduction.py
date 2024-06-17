import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

base = datasets.load_digits()
x = np.asarray(base.data, 'float32')
y = base.target

scaler = MinMaxScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

rbm = BernoulliRBM(random_state=0)
rbm.n_iter = 25
rbm.n_components = 50
naive_rbm = GaussianNB()
rbmClassifier = Pipeline(steps = [('rbm', rbm), ('naive', naive_rbm)])
rbmClassifier.fit(x_train, y_train)

plt.figure(figsize=(20, 20))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10, 10, i+1)
    plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r)
    plt.xticks(())
    plt.yticks(())
plt.show()

preds = rbmClassifier.predict(x_test)
acc = metrics.accuracy_score(y_test, preds)

naive = GaussianNB()
naive.fit(x_train, y_train)
naivePreds = naive.predict(x_test)
naiveAcc = metrics.accuracy_score(y_test, naivePreds)