import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

#linear SVM function
def lin_csf(features, label):
    lin_svm = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1, loss="hinge"))
        ])
    lin_svm.fit(features,label)
    sonuc=lin_svm.predict(x_test)
    return sonuc

#nonlinear SVM(Kernnel="RBF") function 
def nonlin_rbf(features, label):
    nonlin_svm = Pipeline([
        ("scaler", StandardScaler()),
      ("svm_clf", SVC(kernel="rbf", gamma=64, C=1))
        ])
    nonlin_svm.fit(features,label)
    sonuc=nonlin_svm.predict(x_test)
    return sonuc

#Nonlinear SVM(Kernnel="Polynomial") function
def nonlin_poly(features, label):
    lin_svm = Pipeline([
        ("scaler", StandardScaler()),
       ("svm_clf", SVC(kernel="poly", degree=3, coef0=0.4, C=1))
        ])
    lin_svm.fit(features,label)
    sonuc=lin_svm.predict(x_test)
    return sonuc

#verisetini okuyup etiketleri  "versicolor", "versicolor degil" olarak degistirdik.
dataset = pd.read_csv('iris.csv')
X= dataset.iloc[:, [0,1,2,3]]
Y= (dataset.iloc[:, 4]=="versicolor").astype(np.float64)
#%80 egitim, %20 test olarak parcaladik. Veriseti sirali oldugu icin random_state'e 42 degerini verdik.
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42)

#fonksiyonu cagirarak egitim, test islemlerini gerceklestirdik. bir degiskene atadik.
linsvm_sonuc=lin_csf(X,Y)
print("LINEAR SVC")
print("###########")
print("Confusion_matrix:")
conf_matrix=confusion_matrix(y_test, linsvm_sonuc)
print(conf_matrix)
print("--------------")
#f1_score, accuracy, precision, recall degerlerini hesapladik ve yazdirdik.
acc = accuracy_score(y_test,linsvm_sonuc)
recall=recall_score(y_test,linsvm_sonuc)
precision=precision_score(y_test,linsvm_sonuc)
f_measure=f1_score(y_test,linsvm_sonuc)
print( "Accuracy değeri:",acc)
print("Recall değeri:",recall)
print("Precision değeri:",precision)
print("F1 Score değeri",f_measure)
print("###########\n\n")

nonlin_rbf_sonuc=nonlin_rbf(X,Y)
print("NONLINEAR SVC(RBF)")
print("###########")
print("Confusion_matrix:")
conf_matrix=confusion_matrix(y_test, nonlin_rbf_sonuc)
print(conf_matrix)
print("--------------")

acc = accuracy_score(y_test,nonlin_rbf_sonuc)
recall=recall_score(y_test,nonlin_rbf_sonuc)
precision=precision_score(y_test,nonlin_rbf_sonuc)
f_measure=f1_score(y_test,nonlin_rbf_sonuc)
print( "Accuracy değeri:",acc)
print("Recall değeri:",recall)
print("Precision değeri:",precision)
print("F1 Score değeri",f_measure)
print("###########\n\n")

#nonliner(kernel=poly) results:
nonlin_poly_sonuc=nonlin_poly(X,Y)
print("NONLINEAR SVC(POLY)")
print("###########")
print("Confusion_matrix:")
conf_matrix=confusion_matrix(y_test, nonlin_poly_sonuc)
print(conf_matrix)
print("--------------")

acc = accuracy_score(y_test,nonlin_poly_sonuc)
recall=recall_score(y_test,nonlin_poly_sonuc)
precision=precision_score(y_test,nonlin_poly_sonuc)
f_measure=f1_score(y_test,nonlin_poly_sonuc)
print( "Accuracy değeri:",acc)
print("Recall değeri:",recall)
print("Precision değeri:",precision)
print("F1 Score değeri",f_measure)
print("###########\n\n")
