import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
np.random.seed(2)
#il modello viene inizializzato in modo sempre diverso per cui setto il seed
pd.read_csv("BankNote_Authentication.csv")

df = pd.read_csv("BankNote_Authentication.csv")

X, y = df.values[:, :-1], df.values[:, -1]
#print(max(X[:,0]),min(X[:,0]))
X_train, X_test, y_train, y_test=train_test_split(X,y)
#print(X_train)
modelGB = GradientBoostingClassifier()
modelSvm = svm.SVC()
modelNB = GaussianNB()
modelRF = RandomForestClassifier()
modelTree = DecisionTreeClassifier()

modelNB.fit(X_train,y_train)
modelGB.fit(X_train,y_train)
modelSvm.fit(X_train,y_train)
modelTree.fit(X_train,y_train)
modelRF.fit(X_train,y_train)

p_test_GB=modelGB.predict(X_test)
p_test_tree=modelTree.predict(X_test)
p_test_svm=modelSvm.predict(X_test)
p_test_NB=modelNB.predict(X_test)
p_test_RF=modelRF.predict(X_test)

accuracy_test_RF=accuracy_score(y_test,p_test_RF)
precision_test_RF=precision_score(y_test,p_test_RF)
recall_test_RF=recall_score(y_test,p_test_RF)
f1_test_RF=f1_score(y_test,p_test_RF)

accuracy_test_NB=accuracy_score(y_test,p_test_NB)
precision_test_NB=precision_score(y_test,p_test_NB)
recall_test_NB=recall_score(y_test,p_test_NB)
f1_test_NB=f1_score(y_test,p_test_NB)

accuracy_test_GB=accuracy_score(y_test,p_test_GB)
precision_test_GB=precision_score(y_test,p_test_GB)
recall_test_GB=recall_score(y_test,p_test_GB)
f1_test_GB=f1_score(y_test,p_test_GB)

accuracy_test_tree=accuracy_score(y_test,p_test_tree)
precision_test_tree=precision_score(y_test,p_test_tree)
recall_test_tree=recall_score(y_test,p_test_tree)
f1_test_tree=f1_score(y_test,p_test_tree)

accuracy_test_svm=accuracy_score(y_test,p_test_svm)
precision_test_svm=precision_score(y_test,p_test_svm)
recall_test_svm=recall_score(y_test,p_test_svm)
f1_test_svm=f1_score(y_test,p_test_svm)


metriche = ['Accuracy', 'Precision', 'Recall', 'F1-score']
valori = [accuracy_test_svm, precision_test_svm, recall_test_svm,  f1_test_svm]

fig, ax = plt.subplots()
ax.bar(metriche, valori)
ax.set_ylim([0, 1])
ax.set_title('Metriche di valutazione del modello')
ax.set_xlabel('Metriche')
ax.set_ylabel('Valori')
plt.show()


print( f'Decision Tree: accuracy:  {accuracy_test_tree} recall: {recall_test_tree} precision: {precision_test_tree} f1-score: {f1_test_tree}')
print( f'Gradient Boosting: accuracy:  {accuracy_test_GB} recall: {recall_test_GB} precision: {precision_test_GB} f1-score: {f1_test_GB}')
print( f'SVM: accuracy:  {accuracy_test_svm} recall: {recall_test_svm} precision: {precision_test_svm} f1-score: {f1_test_svm}')
print( f'Random Forest: accuracy:  {accuracy_test_RF} recall: {recall_test_RF} precision: {precision_test_RF} f1-score: {f1_test_RF}')
print( f'Naive Bayes Tree: accuracy:  {accuracy_test_NB} recall: {recall_test_NB} precision: {precision_test_NB} f1-score: {f1_test_NB}')

print("Usando il 25% per il test si ha: ")
X_train, X_test, y_train, y_test=train_test_split(X,y,train_size=0.25)
modelGB = GradientBoostingClassifier()
modelSvm = svm.SVC()
modelNB = GaussianNB()
modelRF = RandomForestClassifier()
modelTree = DecisionTreeClassifier()

modelNB.fit(X_train,y_train)
modelGB.fit(X_train,y_train)
modelSvm.fit(X_train,y_train)
modelTree.fit(X_train,y_train)
modelRF.fit(X_train,y_train)

p_test_GB=modelGB.predict(X_test)
p_test_tree=modelTree.predict(X_test)
p_test_svm=modelSvm.predict(X_test)
p_test_NB=modelNB.predict(X_test)
p_test_RF=modelRF.predict(X_test)

accuracy_test_RF=accuracy_score(y_test,p_test_RF)
precision_test_RF=precision_score(y_test,p_test_RF)
recall_test_RF=recall_score(y_test,p_test_RF)
f1_test_RF=f1_score(y_test,p_test_RF)

accuracy_test_NB=accuracy_score(y_test,p_test_NB)
precision_test_NB=precision_score(y_test,p_test_NB)
recall_test_NB=recall_score(y_test,p_test_NB)
f1_test_NB=f1_score(y_test,p_test_NB)

accuracy_test_GB=accuracy_score(y_test,p_test_GB)
precision_test_GB=precision_score(y_test,p_test_GB)
recall_test_GB=recall_score(y_test,p_test_GB)
f1_test_GB=f1_score(y_test,p_test_GB)

accuracy_test_tree=accuracy_score(y_test,p_test_tree)
precision_test_tree=precision_score(y_test,p_test_tree)
recall_test_tree=recall_score(y_test,p_test_tree)
f1_test_tree=f1_score(y_test,p_test_tree)

accuracy_test_svm=accuracy_score(y_test,p_test_svm)
precision_test_svm=precision_score(y_test,p_test_svm)
recall_test_svm=recall_score(y_test,p_test_svm)
f1_test_svm=f1_score(y_test,p_test_svm)

print( f'Decision Tree: accuracy:  {accuracy_test_tree} recall: {recall_test_tree} precision: {precision_test_tree} f1-score: {f1_test_tree}')
print( f'Gradient Boosting: accuracy:  {accuracy_test_GB} recall: {recall_test_GB} precision: {precision_test_GB} f1-score: {f1_test_GB}')
print( f'SVM: accuracy:  {accuracy_test_svm} recall: {recall_test_svm} precision: {precision_test_svm} f1-score: {f1_test_svm}')
print( f'Random Forest: accuracy:  {accuracy_test_RF} recall: {recall_test_RF} precision: {precision_test_RF} f1-score: {f1_test_RF}')
print( f'Naive Bayes Tree: accuracy:  {accuracy_test_NB} recall: {recall_test_NB} precision: {precision_test_NB} f1-score: {f1_test_NB}')

print("Usando il 50% dei dati per il test si ha: ")
X_train, X_test, y_train, y_test=train_test_split(X,y,train_size=0.5)
modelGB = GradientBoostingClassifier()
modelSvm = svm.SVC()
modelNB = GaussianNB()
modelRF = RandomForestClassifier()
modelTree = DecisionTreeClassifier()

modelNB.fit(X_train,y_train)
modelGB.fit(X_train,y_train)
modelSvm.fit(X_train,y_train)
modelTree.fit(X_train,y_train)
modelRF.fit(X_train,y_train)

p_test_GB=modelGB.predict(X_test)
p_test_tree=modelTree.predict(X_test)
p_test_svm=modelSvm.predict(X_test)
p_test_NB=modelNB.predict(X_test)
p_test_RF=modelRF.predict(X_test)

accuracy_test_RF=accuracy_score(y_test,p_test_RF)
precision_test_RF=precision_score(y_test,p_test_RF)
recall_test_RF=recall_score(y_test,p_test_RF)
f1_test_RF=f1_score(y_test,p_test_RF)

accuracy_test_NB=accuracy_score(y_test,p_test_NB)
precision_test_NB=precision_score(y_test,p_test_NB)
recall_test_NB=recall_score(y_test,p_test_NB)
f1_test_NB=f1_score(y_test,p_test_NB)

accuracy_test_GB=accuracy_score(y_test,p_test_GB)
precision_test_GB=precision_score(y_test,p_test_GB)
recall_test_GB=recall_score(y_test,p_test_GB)
f1_test_GB=f1_score(y_test,p_test_GB)

accuracy_test_tree=accuracy_score(y_test,p_test_tree)
precision_test_tree=precision_score(y_test,p_test_tree)
recall_test_tree=recall_score(y_test,p_test_tree)
f1_test_tree=f1_score(y_test,p_test_tree)

accuracy_test_svm=accuracy_score(y_test,p_test_svm)
precision_test_svm=precision_score(y_test,p_test_svm)
recall_test_svm=recall_score(y_test,p_test_svm)
f1_test_svm=f1_score(y_test,p_test_svm)

print( f'Decision Tree: accuracy:  {accuracy_test_tree} recall: {recall_test_tree} precision: {precision_test_tree} f1-score: {f1_test_tree}')
print( f'Gradient Boosting: accuracy:  {accuracy_test_GB} recall: {recall_test_GB} precision: {precision_test_GB} f1-score: {f1_test_GB}')
print( f'SVM: accuracy:  {accuracy_test_svm} recall: {recall_test_svm} precision: {precision_test_svm} f1-score: {f1_test_svm}')
print( f'Random Forest: accuracy:  {accuracy_test_RF} recall: {recall_test_RF} precision: {precision_test_RF} f1-score: {f1_test_RF}')
print( f'Naive Bayes Tree: accuracy:  {accuracy_test_NB} recall: {recall_test_NB} precision: {precision_test_NB} f1-score: {f1_test_NB}')

print("Usando il 75% dei dati per il test si ha: ")
X_train, X_test, y_train, y_test=train_test_split(X,y,train_size=0.75)
modelGB = GradientBoostingClassifier()
modelSvm = svm.SVC()
modelNB = GaussianNB()
modelRF = RandomForestClassifier()
modelTree = DecisionTreeClassifier()

modelNB.fit(X_train,y_train)
modelGB.fit(X_train,y_train)
modelSvm.fit(X_train,y_train)
modelTree.fit(X_train,y_train)
modelRF.fit(X_train,y_train)

p_test_GB=modelGB.predict(X_test)
p_test_tree=modelTree.predict(X_test)
p_test_svm=modelSvm.predict(X_test)
p_test_NB=modelNB.predict(X_test)
p_test_RF=modelRF.predict(X_test)

accuracy_test_RF=accuracy_score(y_test,p_test_RF)
precision_test_RF=precision_score(y_test,p_test_RF)
recall_test_RF=recall_score(y_test,p_test_RF)
f1_test_RF=f1_score(y_test,p_test_RF)

accuracy_test_NB=accuracy_score(y_test,p_test_NB)
precision_test_NB=precision_score(y_test,p_test_NB)
recall_test_NB=recall_score(y_test,p_test_NB)
f1_test_NB=f1_score(y_test,p_test_NB)

accuracy_test_GB=accuracy_score(y_test,p_test_GB)
precision_test_GB=precision_score(y_test,p_test_GB)
recall_test_GB=recall_score(y_test,p_test_GB)
f1_test_GB=f1_score(y_test,p_test_GB)

accuracy_test_tree=accuracy_score(y_test,p_test_tree)
precision_test_tree=precision_score(y_test,p_test_tree)
recall_test_tree=recall_score(y_test,p_test_tree)
f1_test_tree=f1_score(y_test,p_test_tree)

accuracy_test_svm=accuracy_score(y_test,p_test_svm)
precision_test_svm=precision_score(y_test,p_test_svm)
recall_test_svm=recall_score(y_test,p_test_svm)
f1_test_svm=f1_score(y_test,p_test_svm)

print( f'Decision Tree: accuracy:  {accuracy_test_tree} recall: {recall_test_tree} precision: {precision_test_tree} f1-score: {f1_test_tree}')
print( f'Gradient Boosting: accuracy:  {accuracy_test_GB} recall: {recall_test_GB} precision: {precision_test_GB} f1-score: {f1_test_GB}')
print( f'SVM: accuracy:  {accuracy_test_svm} recall: {recall_test_svm} precision: {precision_test_svm} f1-score: {f1_test_svm}')
print( f'Random Forest: accuracy:  {accuracy_test_RF} recall: {recall_test_RF} precision: {precision_test_RF} f1-score: {f1_test_RF}')
print( f'Naive Bayes Tree: accuracy:  {accuracy_test_NB} recall: {recall_test_NB} precision: {precision_test_NB} f1-score: {f1_test_NB}')


#andando ad aumentare la dimensione dei dati usati per il test l'accuratezza del modello diminuisce
print("Vediamo ora l'impatto della normalizzazione MinMax sull'accuratezza del modello, usando il 25% dei dati per il test")
scaler = MinMaxScaler()
X_scala = scaler.fit_transform(X)
X_train, X_test, y_train, y_test=train_test_split(X_scala,y,train_size=0.25)
modelTree = DecisionTreeClassifier()
modelTree.fit(X_train,y_train)
p_test_tree=modelTree.predict(X_test)
accuracy_test_tree=accuracy_score(y_test,p_test_tree)

print("Usando il 25% per il test ed avendo normalizzato con MinMax si ha: ")
print( f'Accuratezza test:  {accuracy_test_tree}')