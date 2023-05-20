import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler
np.random.seed(2)

def outlier(feature,n):
    # Trasformazione dei dati aggiungendo un valore costante per renderli tutti positivi
    shifted_data = feature - np.min(feature) + 1
    #print(shifted_data)
    # Calcolo del punteggio Z
    z_scores = stats.zscore(shifted_data)
    #print(z_scores)
    # Identificazione degli outlier
    threshold = np.mean(np.abs(feature))
    #print(threshold)
    outliers = np.where(np.abs(z_scores) > threshold)[0]
    # Stampa degli outlier
    print("Nella feature n°",n," sono presenti ",len(outliers)," valori anomali")

#il modello viene inizializzato in modo sempre diverso per cui setto il seed
pd.read_csv("BankNote_Authentication.csv")

df = pd.read_csv("BankNote_Authentication.csv")

X, y = df.values[:, :-1], df.values[:, -1]
#print(X)
feature1=X[:,0]
feature2=X[:,1]
feature3=X[:,2]
feature4=X[:,3]
#isolation_forest = IsolationForest()
#isolation_forest.fit(X)
#anomaly_scores = isolation_forest.decision_function(X)
#print(anomaly_scores)
"""
outlier(feature1,1)
outlier(feature2,2)
outlier(feature3,3)
outlier(feature4,4)

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.25)

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

print("Matrice di confusione Decision Tree")
tn, fp, fn, tp=confusion_matrix(y_test,p_test_tree).ravel()
#print(confusion_matrix(y_test,p_test_tree))
print(f'Falsi positivi: {fp}')
print(f'Veri positivi: {tp}')
print(f'Falsi negativi: {fn}')
print(f'Veri positivi: {tn}')

print("Matrice di confusione SVM")
tn, fp, fn, tp=confusion_matrix(y_test,p_test_svm).ravel()
#print(confusion_matrix(y_test,p_test_tree))
print(f'Falsi positivi: {fp}')
print(f'Veri positivi: {tp}')
print(f'Falsi negativi: {fn}')
print(f'Veri positivi: {tn}')

plt.hist(feature1, bins=30)
plt.xlabel('Valore')
plt.ylabel('Frequenza')
plt.title('Distribuzione della feature 1')
#plt.show()

plt.hist(feature2, bins=30)
plt.xlabel('Valore')
plt.ylabel('Frequenza')
plt.title('Distribuzione della feature 2')
#plt.show()

plt.hist(feature3, bins=30)
plt.xlabel('Valore')
plt.ylabel('Frequenza')
plt.title('Distribuzione della feature 3')
#plt.show()

plt.hist(feature4, bins=30)
plt.xlabel('Valore')
plt.ylabel('Frequenza')
plt.title('Distribuzione della feature 4')
#plt.show()

plt.tight_layout()
#plt.show()

#verifichiamo ora che esse siano indipendenti controllando la matrice di correlazione
print("Matrice di correlazione")
correlation_matrix = np.corrcoef(X, rowvar=False)
print(correlation_matrix)

tn, fp, fn, tp=confusion_matrix(y_test,p_test_NB).ravel()
#print(confusion_matrix(y_test,p_test_tree))
print("Matrice di confusione Naive Bayes")
print(f'Falsi positivi: {fp}')
print(f'Veri positivi: {tp}')
print(f'Falsi negativi: {fn}')
print(f'Veri positivi: {tn}')

print("Matrice di confusione Random Forest")
tn, fp, fn, tp=confusion_matrix(y_test,p_test_RF).ravel()
#print(confusion_matrix(y_test,p_test_tree))
print(f'Falsi positivi: {fp}')
print(f'Veri positivi: {tp}')
print(f'Falsi negativi: {fn}')
print(f'Veri positivi: {tn}')

print("Matrice di confusione Gradient Boosting")
tn, fp, fn, tp=confusion_matrix(y_test,p_test_GB).ravel()
#print(confusion_matrix(y_test,p_test_tree))
print(f'Falsi positivi: {fp}')
print(f'Veri positivi: {tp}')
print(f'Falsi negativi: {fn}')
print(f'Veri positivi: {tn}')

metriche = ['Accuracy', 'Precision', 'Recall', 'F1-score']
valori = [accuracy_test_svm, precision_test_svm, recall_test_svm,  f1_test_svm]

fig, ax = plt.subplots()
ax.bar(metriche, valori)
ax.set_ylim([0, 1])
ax.set_title('Metriche di valutazione del modello')
ax.set_xlabel('Metriche')
ax.set_ylabel('Valori')
#plt.show()

data = {'Algoritmo': ['Decision Tree', 'SVM', 'Naive Bayes', 'Random Forest', 'Gradient Boosting'],
        'Accuracy': [accuracy_test_tree, accuracy_test_svm, accuracy_test_NB, accuracy_test_RF,accuracy_test_GB],
        'Recall':[recall_test_tree,recall_test_svm,recall_test_NB,recall_test_RF,recall_test_GB],
        'Precision':[precision_test_tree,precision_test_svm,precision_test_NB,precision_test_RF,precision_test_GB],
        'F1-score': [f1_test_tree,f1_test_svm,f1_test_NB,f1_test_RF,f1_test_GB]}

# creiamo un oggetto DataFrame di Pandas con i dati
tabella = pd.DataFrame(data)

# stampiamo la tabella
print(tabella)
"""
#andando ad aumentare la dimensione dei dati usati per il test l'accuratezza del modello diminuisce
print("Vediamo ora l'impatto della normalizzazione MinMax sull'accuratezza del modello, usando il 25% dei dati per il test")
scaler = MinMaxScaler()
X_scala = scaler.fit_transform(X)
X_train, X_test, y_train, y_test=train_test_split(X_scala,y,test_size=0.25)

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

data = {'Algoritmo': ['Decision Tree', 'SVM', 'Naive Bayes', 'Random Forest', 'Gradient Boosting'],
        'Accuracy': [accuracy_test_tree, accuracy_test_svm, accuracy_test_NB, accuracy_test_RF,accuracy_test_GB],
        'Recall':[recall_test_tree,recall_test_svm,recall_test_NB,recall_test_RF,recall_test_GB],
        'Precision':[precision_test_tree,precision_test_svm,precision_test_NB,precision_test_RF,precision_test_GB],
        'F1-score': [f1_test_tree,f1_test_svm,f1_test_NB,f1_test_RF,f1_test_GB]}

# creiamo un oggetto DataFrame di Pandas con i dati
tabella = pd.DataFrame(data)

# stampiamo la tabella
print(tabella)

print("Matrice di confusione Decision Tree")
tn, fp, fn, tp=confusion_matrix(y_test,p_test_tree).ravel()
#print(confusion_matrix(y_test,p_test_tree))
print(f'Falsi positivi: {fp}')
print(f'Veri positivi: {tp}')
print(f'Falsi negativi: {fn}')
print(f'Veri positivi: {tn}')

print("Matrice di confusione SVM")
tn, fp, fn, tp=confusion_matrix(y_test,p_test_svm).ravel()
#print(confusion_matrix(y_test,p_test_tree))
print(f'Falsi positivi: {fp}')
print(f'Veri positivi: {tp}')
print(f'Falsi negativi: {fn}')
print(f'Veri positivi: {tn}')

tn, fp, fn, tp=confusion_matrix(y_test,p_test_NB).ravel()
#print(confusion_matrix(y_test,p_test_tree))
print("Matrice di confusione Naive Bayes")
print(f'Falsi positivi: {fp}')
print(f'Veri positivi: {tp}')
print(f'Falsi negativi: {fn}')
print(f'Veri positivi: {tn}')

print("Matrice di confusione Random Forest")
tn, fp, fn, tp=confusion_matrix(y_test,p_test_RF).ravel()
#print(confusion_matrix(y_test,p_test_tree))
print(f'Falsi positivi: {fp}')
print(f'Veri positivi: {tp}')
print(f'Falsi negativi: {fn}')
print(f'Veri positivi: {tn}')

print("Matrice di confusione Gradient Boosting")
tn, fp, fn, tp=confusion_matrix(y_test,p_test_GB).ravel()
#print(confusion_matrix(y_test,p_test_tree))
print(f'Falsi positivi: {fp}')
print(f'Veri positivi: {tp}')
print(f'Falsi negativi: {fn}')
print(f'Veri positivi: {tn}')


print("Vediamo ora l'impatto della normalizzazione StandardScaler sull'accuratezza del modello, usando il 25% dei dati per il test")
scaler = StandardScaler()
X_scala = scaler.fit_transform(X)
#print(X)
X_train, X_test, y_train, y_test=train_test_split(X_scala,y,test_size=0.25)
#print(X_scala)
#isolation_forest = IsolationForest()
#isolation_forest.fit(X_scala)
#anomaly_scores = isolation_forest.decision_function(X_scala)
#print(anomaly_scores)

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

data = {'Algoritmo': ['Decision Tree', 'SVM', 'Naive Bayes', 'Random Forest', 'Gradient Boosting'],
        'Accuracy': [accuracy_test_tree, accuracy_test_svm, accuracy_test_NB, accuracy_test_RF,accuracy_test_GB],
        'Recall':[recall_test_tree,recall_test_svm,recall_test_NB,recall_test_RF,recall_test_GB],
        'Precision':[precision_test_tree,precision_test_svm,precision_test_NB,precision_test_RF,precision_test_GB],
        'F1-score': [f1_test_tree,f1_test_svm,f1_test_NB,f1_test_RF,f1_test_GB]}

# creiamo un oggetto DataFrame di Pandas con i dati
tabella = pd.DataFrame(data)

# stampiamo la tabella
print(tabella)

print("Matrice di confusione Decision Tree")
tn, fp, fn, tp=confusion_matrix(y_test,p_test_tree).ravel()
#print(confusion_matrix(y_test,p_test_tree))
print(f'Falsi positivi: {fp}')
print(f'Veri positivi: {tp}')
print(f'Falsi negativi: {fn}')
print(f'Veri positivi: {tn}')

print("Matrice di confusione SVM")
tn, fp, fn, tp=confusion_matrix(y_test,p_test_svm).ravel()
#print(confusion_matrix(y_test,p_test_tree))
print(f'Falsi positivi: {fp}')
print(f'Veri positivi: {tp}')
print(f'Falsi negativi: {fn}')
print(f'Veri positivi: {tn}')

tn, fp, fn, tp=confusion_matrix(y_test,p_test_NB).ravel()
#print(confusion_matrix(y_test,p_test_tree))
print("Matrice di confusione Naive Bayes")
print(f'Falsi positivi: {fp}')
print(f'Veri positivi: {tp}')
print(f'Falsi negativi: {fn}')
print(f'Veri positivi: {tn}')

print("Matrice di confusione Random Forest")
tn, fp, fn, tp=confusion_matrix(y_test,p_test_RF).ravel()
#print(confusion_matrix(y_test,p_test_tree))
print(f'Falsi positivi: {fp}')
print(f'Veri positivi: {tp}')
print(f'Falsi negativi: {fn}')
print(f'Veri positivi: {tn}')

print("Matrice di confusione Gradient Boosting")
tn, fp, fn, tp=confusion_matrix(y_test,p_test_GB).ravel()
#print(confusion_matrix(y_test,p_test_tree))
print(f'Falsi positivi: {fp}')
print(f'Veri positivi: {tp}')
print(f'Falsi negativi: {fn}')
print(f'Veri positivi: {tn}')

print("Matrice di correlazione")
correlation_matrix = np.corrcoef(X_scala, rowvar=False)
print(correlation_matrix)

feature1=X_scala[:,0]
feature2=X_scala[:,1]
feature3=X_scala[:,2]
feature4=X_scala[:,3]

outlier(feature1,1)
outlier(feature2,2)
outlier(feature3,3)
outlier(feature4,4)

plt.hist(feature1, bins=30)
plt.xlabel('Valore')
plt.ylabel('Frequenza')
plt.title('Distribuzione della feature 1')
plt.show()

plt.hist(feature2, bins=30)
plt.xlabel('Valore')
plt.ylabel('Frequenza')
plt.title('Distribuzione della feature 2')
plt.show()

plt.hist(feature3, bins=30)
plt.xlabel('Valore')
plt.ylabel('Frequenza')
plt.title('Distribuzione della feature 3')
plt.show()

plt.hist(feature4, bins=30)
plt.xlabel('Valore')
plt.ylabel('Frequenza')
plt.title('Distribuzione della feature 4')
plt.show()
