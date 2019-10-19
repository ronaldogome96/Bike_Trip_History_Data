
#Importa as bibliotecas necessarias
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , accuracy_score
from sklearn.svm import SVC

#Le o data frame
base = pd.read_csv('201909-capitalbikeshare-tripdata.csv')

#Divide o data frame em atributos e classe
atributos = base.iloc[:, 0:8].values
classe = base.iloc[:, 8].values

#Muda as variaveis nominais para variaveis numericas
labelencoder_atributos = LabelEncoder()
atributos[:, 1] = labelencoder_atributos.fit_transform(atributos[:, 1])
atributos[:, 2] = labelencoder_atributos.fit_transform(atributos[:, 2])
atributos[:, 4] = labelencoder_atributos.fit_transform(atributos[:, 4])
atributos[:, 6] = labelencoder_atributos.fit_transform(atributos[:, 6])
atributos[:, 7] = labelencoder_atributos.fit_transform(atributos[:, 7])
labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

#Aplica o escalonamento
scaler = StandardScaler()
atributos = scaler.fit_transform(atributos)

#Divide os dados para treinamento e teste
atributos_treinamento, atributos_teste, classe_treinamento, classe_teste = train_test_split(atributos, classe, test_size=0.20, random_state=0)

#Regressao logistica
classificador = SVC(kernel = 'linear', random_state=1)
classificador.fit(atributos_treinamento, classe_treinamento)
previsoes = classificador.predict(atributos_teste)

#Mostra a precisao
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)
#PRECISAO ??????
