# %% [markdown]
# # Teste com apenas labelEncoder nos dados

# %%
dataset = pd.read_csv('data.csv', sep=',')

# Separa a classe dos dados

dataset.info()

classes = dataset['target']
dataset.drop('target', axis=1, inplace=True)

# Pre-processamento de dados

def remove_features(lista_features):
    for i in lista_features:
        dataset.drop(i, axis=1, inplace=True)
    return 0

# Remove features
remove_features(['id','song_title'])

# %%
# Label Encoder

from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

inteiros = enc.fit_transform(dataset['artist'])

# %%
# Visualizando valores únicos.
set(inteiros)

# Cria uma nova coluna chamada 'artist_inteiros'
dataset['artist_inteiros'] = inteiros

dataset.head()

# %%
remove_features(['artist'])

dataset.head(10)

# %%
dataset.columns

# %%
len(dataset.columns)

# %% [markdown]
# recriando os pipelines apenas com label encoder

# %%
# Importe as bibliotecas de Pipelines e Pré-processadores
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# %%
# Função que retorna a acurácia após fazer um validação cruzada (cross validation)
def Acuracia(clf,X,y):
    resultados = cross_val_predict(clf, X, y, cv=10)
    return metrics.accuracy_score(y,resultados)

# %%
Acuracia(clf,dataset,classes)

# %%
pip_1 = Pipeline([
    ('scaler',StandardScaler()),
    ('clf', svm.SVC())
])

# %%
# Criando vários Pipelines
pip_2 = Pipeline([
    ('min_max_scaler', MinMaxScaler()),
    ('clf', svm.SVC())
])

pip_3 = Pipeline([
    ('scaler',StandardScaler()),
    ('clf', svm.SVC(kernel='rbf'))
])

pip_4 = Pipeline([
    ('scaler',StandardScaler()),
    ('clf', svm.SVC(kernel='poly'))
])

pip_5 = Pipeline([
    ('scaler',StandardScaler()),
    ('clf', svm.SVC(kernel='linear'))
])

# %%
# Teste com apenas LabelEncoder na coluna 'artist' usando o pipeline 'pip_1'
Acuracia(pip_1,dataset,classes)

# %%
# # Teste com apenas LabelEncoder na coluna 'artist' usando o pipeline 'pip_1'
Acuracia(pip_2,dataset,classes)

# %% [markdown]
# # Testando o Desempenho dos Kernels

# %%
# Testando o Kernel RBF
Acuracia(pip_3,dataset,classes)

# %%
# Teste de kernel poly
Acuracia(pip_4,dataset,classes)

# %%
# Teste de Kernel linear
Acuracia(pip_5,dataset,classes)

# %% [markdown]
# ## Teste de Overfitting

# %%
# Utiliza a função train_test_split para separar conjunto de treino e teste em 80/20
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, classes, test_size=0.2, random_state=123)

# %%
# Scala os dados de treino e teste.
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

scaler2 = StandardScaler().fit(X_test)
X_test = scaler2.transform(X_test)

# %%
# Treina o algoritmo
clf.fit(X_train, y_train)

# %%
# Resultados de predição.
y_pred  = clf.predict(X_test)

# %%
# Imprime a acurácia.
metrics.accuracy_score(y_test,y_pred)

# %%
# Testando a classificação com o próprio teste

# %%
# Resultados de predição
y_pred  = clf.predict(X_train)

# %%
# Imprime a Acurácia.
metrics.accuracy_score(y_train,y_pred)

# %% [markdown]
# 
# # Tunning

# %%
# Importa o utilitário GridSearchCV
from sklearn.model_selection import GridSearchCV

# %%
# Lista de Valores de C
lista_C = [0.001, 0.01, 0.1, 1, 10, 100]

# Lista de Valores de gamma
lista_gamma = [0.001, 0.01, 0.1, 1, 10, 100]

# %%
# Define um dicionário que recebe as listas de parâmetros e valores.
parametros_grid = dict(clf__C=lista_C, clf__gamma=lista_gamma)

# %%
parametros_grid

# %%
# Objeto Grid recebe parâmetros de Pipeline, e configurações de cross validation
grid = GridSearchCV(pip_3, parametros_grid, cv=10, scoring='accuracy')

# %%
# Aplica o gridsearch passando os dados de treino e classes.
grid.fit(dataset,classes)

# %%
grid

# %% [markdown]
# # Resultados de Grid

# %%
# Imprime os scores por combinações
#grid.grid_scores_
grid.cv_results_

# %%
# Imprime os melhores parâmetros
grid.best_params_

# %%
grid.best_score_


