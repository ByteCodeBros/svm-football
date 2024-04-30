import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
import pandas as pd

# Carregar o conjunto de dados de treinamento
dataset = np.loadtxt('jogadores.csv', delimiter=",", skiprows=1)

X_train = dataset[:, [5, 6]]  # Pegando Chute e Força
y_train = dataset[:, [9]]  # Classe

# Criar o modelo SVM
C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=1, gamma=0)

# # Criar objeto de validação cruzada com k-folds (n_splits=6)
# kf = KFold(n_splits=5, shuffle=False, random_state=None)
#
# for i, (train_index, test_index) in enumerate(kf.split(X_train)):
#     print(f"Fold {i}:")
#     print(f"{test_index}")
#
# # Realizar a validação cruzada e obter as previsões
# y_train_pred = cross_val_predict(svc, X_train, y_train.ravel(), cv=kf)
# print(y_train_pred)

# Treinar o modelo SVM com todo o conjunto de treinamento
svc.fit(X_train, y_train.ravel())

# Criar um novo conjunto de dados para validação
validation_data = np.loadtxt('validacao.csv', delimiter=",", skiprows=1)
X_validation = validation_data[:, [5, 6]]  # Pegando Chute e Força

# Prever as classes para o conjunto de dados de validação
y_validation_pred = svc.predict(X_validation)

# Adicionar as classes previstas ao conjunto de dados de validação
validation_data_with_pred = np.c_[X_validation, y_validation_pred]

# Criar um DataFrame Pandas para facilitar a manipulação e salvar em um novo CSV
columns = ["Chute", "Forca", "Classe_pred"]
df_validation_with_pred = pd.DataFrame(validation_data_with_pred, columns=columns)
df_validation_with_pred.to_csv('validation_with_predictions.csv', index=False, sep=';')

# Plotar
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Scatter plot para os dados de treinamento
scatter_train = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train.ravel(), cmap=plt.cm.Paired, label='Treinamento')

# Scatter plot para os dados de validação preditos
scatter_validation_pred = ax.scatter(X_validation[:, 0], X_validation[:, 1], c=y_validation_pred, cmap=plt.cm.Paired,
                                     marker='x', s=100, label='Validação (Predito)')

# Plotar o hiperplano
ax.autoscale(enable=False)
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Criar grid para avaliação do modelo
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                     np.linspace(ylim[0], ylim[1], 50))
Z = svc.decision_function(np.c_[xx.ravel(), yy.ravel()])

# Plotar hiperplano e margens
Z = Z.reshape(xx.shape)
ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

# Definir rótulos e título
ax.set_xlabel('Chute do jogador')
ax.set_ylabel('Força do jogador')
ax.set_title('SVC with linear kernel')
ax.legend()

plt.savefig("image.jpg", dpi=500)
plt.show()
