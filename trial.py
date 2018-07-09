
"""
Created on Sat Mar 17 12:04:37 2018

@author: ROHIT
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
dataset = pd.read_csv('spam.csv', encoding='cp437')
dataset.loc[dataset["Status"]=='ham','Status']=0
dataset.loc[dataset["Status"]=='spam','Status']=1

X=dataset.iloc[:,1]
y=dataset["Status"]

accuracies=[]
false_positive_indices=[]
false_negative_indices=[]
yplt_true=[]
yplt_pred=[]
yplt=[]
xplt=[]
xplttmp=[]
yplttmp=[]
"""
type 1
cv=CountVectorizer(min_df=1,stop_words='english',max_features=3000)
X_cv=cv.fit_transform(X).toarray()

X_train,X_test,y_train,y_test=train_test_split(X_cv,y,test_size=0.2)
"""
def plot_false_negatives():
    yplt_pred.clear()
    yplt_true.clear()
    yplt.clear()
    xplt.clear()
    false_negative_indices.clear()

    for i in range(0,len(y_pred)):
        if(y_pred[i]==0 and y_true[i]==1):
            false_negative_indices.append(i)
            yplt_true.append(y_true[i])
            yplt_pred.append(y_pred[i])
    
    print(false_negative_indices)
    
    
    #to generate graph of false positives and false negatives
    
    for i in range (0,1115):
        if (i%50==0 and (i not in false_negative_indices)):
            xplt.append(i)
            yplt.append(y_true[i])
    yplt.extend(yplt_true)
    xplt.extend(false_negative_indices)
    
    xplttmp=xplt[:]
    yplttmp=yplt[:]
    xplt.sort()
    for i in range(0,len(xplt)):
        for j in range(0,len(xplt)):
            if(xplt[i]==xplttmp[j]):
                yplt[i]=yplttmp[j]
    
    
    plt.plot(xplt,yplt,color='blue')
    
    xplt.clear()
    yplt.clear()
    for i in range (0,1115):
        if (i%50==0 and (i not in false_negative_indices)):
            xplt.append(i)
            yplt.append(y_pred[i])
    yplt.extend(yplt_pred)
    xplt.extend(false_negative_indices)
    
    xplttmp=xplt[:]
    yplttmp=yplt[:]
    xplt.sort()
    for i in range(0,len(xplt)):
        for j in range(0,len(xplt)):
            if(xplt[i]==xplttmp[j]):
                yplt[i]=yplttmp[j]
    
    
    plt.scatter(xplt,yplt,color='red')
    plt.xlabel("Message number")
    plt.ylabel("Spam or Ham")
    plt.show()
    
    
def plot_false_positives():
    yplt_pred.clear()
    yplt_true.clear()
    yplt.clear()
    xplt.clear()
    false_positive_indices.clear()

    for i in range(0,len(y_pred)):
        if(y_pred[i]==1 and y_true[i]==0):
            false_positive_indices.append(i)
            yplt_true.append(y_true[i])
            yplt_pred.append(y_pred[i])
    
    print(false_positive_indices)
    
    
    #to generate graph of false positives and false negatives
    
    for i in range (0,1115):
        if (i%50==0 and (i not in false_positive_indices)):
            xplt.append(i)
            yplt.append(y_true[i])
    yplt.extend(yplt_true)
    xplt.extend(false_positive_indices)
    
    xplttmp=xplt[:]
    yplttmp=yplt[:]
    xplt.sort()
    for i in range(0,len(xplt)):
        for j in range(0,len(xplt)):
            if(xplt[i]==xplttmp[j]):
                yplt[i]=yplttmp[j]
    
    
    plt.plot(xplt,yplt,color='blue')
    
    xplt.clear()
    yplt.clear()
    for i in range (0,1115):
        if (i%50==0 and (i not in false_positive_indices)):
            xplt.append(i)
            yplt.append(y_pred[i])
    yplt.extend(yplt_pred)
    xplt.extend(false_positive_indices)
    
    xplttmp=xplt[:]
    yplttmp=yplt[:]
    xplt.sort()
    for i in range(0,len(xplt)):
        for j in range(0,len(xplt)):
            if(xplt[i]==xplttmp[j]):
                yplt[i]=yplttmp[j]
    
    
    plt.scatter(xplt,yplt,color='red')
    plt.xlabel("Message number")
    plt.ylabel("Spam or Ham")
    plt.show()


cv=TfidfVectorizer(min_df=1,stop_words='english',max_features=3000)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

X_cv_train=cv.fit_transform(X_train)

X_cv_test=cv.transform(X_test)
y_train=y_train.astype('int')

#Multinomial Naive Bayes----------------------------------------------------------------
print('Multinomial Naive Bayes')
mnb=MultinomialNB()
mnb.fit(X_cv_train,y_train)
y_true=np.array(y_test)
y_true=y_true.astype('int')
pred=mnb.predict(X_cv_test)

#Making predicted vector and test vector as integer
y_pred=np.array(pred)
y_pred=y_pred.astype('int')

#Making confusion matrix
from sklearn.metrics import confusion_matrix
print('Confusion Matrix')
cm = confusion_matrix(y_true,y_pred)
print(cm)

#Accuracy score evaluation
from sklearn.metrics import accuracy_score
print('Accuracy of model is :- '+str(accuracy_score(y_true,y_pred)))

#appending for final histogram
accuracies.append(accuracy_score(y_true,y_pred))

print("\nFalse positives")
plot_false_positives()
print("False negatives")
plot_false_negatives()



#Random Forest Classifier---------------------------------------------------------
print('\nRandom Forest Classification')
from sklearn.ensemble import RandomForestClassifier
classifier1=RandomForestClassifier(n_estimators=15,criterion='entropy')
classifier1.fit(X_cv_train,y_train)
pred=classifier1.predict(X_cv_test)

#Making predicted vecotr and test vector as integer
y_pred=np.array(pred)
y_pred=y_pred.astype('int')
y_true=np.array(y_test)
y_true=y_true.astype('int')

#Making confusion matrix
from sklearn.metrics import confusion_matrix
print('Confusion Matrix')
cm = confusion_matrix(y_true,y_pred)
print(cm)

#Accuracy score evaluation
from sklearn.metrics import accuracy_score
print('Accuracy of model is :- '+str(accuracy_score(y_true,y_pred)))

#appending for final histogram
accuracies.append(accuracy_score(y_true,y_pred))

print("\nFalse positives")
plot_false_positives()
print("False negatives")
plot_false_negatives()




#Logistic Regression------------------------------------------------------------
print('\nLogistic Regression')
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_cv_train, y_train)
pred=classifier.predict(X_cv_test)

#Making predicted vecotr and test vector as integer
y_pred=np.array(pred)
y_pred=y_pred.astype('int')
y_true=np.array(y_test)
y_true=y_true.astype('int')

#Making confusion matrix
from sklearn.metrics import confusion_matrix
print('Confusion Matrix')
cm = confusion_matrix(y_true,y_pred)
print(cm)

#Accuracy score evaluation
from sklearn.metrics import accuracy_score
print('Accuracy of model is :- '+str(accuracy_score(y_true,y_pred)))

#appending for final histogram
accuracies.append(accuracy_score(y_true,y_pred))

print("\nFalse positives")
plot_false_positives()
print("False negatives")
plot_false_negatives()

#K-Nearest Neighbour------------------------------------------------------------
print('\nK-Nearest Neighbour')
knnclass=KNeighborsClassifier(n_neighbors=5, metric='minkowski',p=2)
knnclass.fit(X_cv_train,y_train)
pred=knnclass.predict(X_cv_test)

#Making predicted vecotr and test vector as integer
y_pred=np.array(pred)
y_pred=y_pred.astype('int')
y_true=np.array(y_test)
y_true=y_true.astype('int')

#Making confusion matrix
from sklearn.metrics import confusion_matrix
print('Confusion Matrix')
cm = confusion_matrix(y_true,y_pred)
print(cm)

#Accuracy score evaluation
from sklearn.metrics import accuracy_score
print('Accuracy of model is :- '+str(accuracy_score(y_true,y_pred)))

#appending for final histogram
accuracies.append(accuracy_score(y_true,y_pred))

print("\nFalse positives")
plot_false_positives()
print("False negatives")
plot_false_negatives()



"to test my own test case use pd.Series('string of text') in .predict method of the respcetive classifier"
#Scalable Vector Classifier------------------------------------------------------
print('\nSVC')
from sklearn.svm import SVC
svc=SVC(kernel='linear',random_state=0)
svc.fit(X_cv_train,y_train)
pred=svc.predict(X_cv_test)

#Making predicted vecotr and test vector as integer
y_pred=np.array(pred)
y_pred=y_pred.astype('int')
y_true=np.array(y_test)
y_true=y_true.astype('int')

#Making confusion matrix
from sklearn.metrics import confusion_matrix
print('Confusion Matrix')
cm = confusion_matrix(y_true,y_pred)
print(cm)

#Accuracy score evaluation
from sklearn.metrics import accuracy_score
print('Accuracy of model is :- '+str(accuracy_score(y_true,y_pred)))

#appending for final histogram
#accuracies.append(accuracy_score(y_true,y_pred))

print("\nFalse positives")
plot_false_positives()
print("False negatives")
plot_false_negatives()


#final histogram
classifier_types =['MNB','RFC','LRG','KNN']
tp = [1,2,3,4]
for i in range (0,4):
    accuracies[i]=accuracies[i]*100
    
print('+ - - + - - - - +')
for i in range (0,4):
    print('| '+classifier_types[i]+' | '+str(accuracies[i].round(4))+' |')
    print('+ - - + - - - - +')

plt.bar(tp,accuracies,color='red')
plt.yticks(np.arange(0,110,10))
plt.xticks(tp,classifier_types)
plt.xlabel('Classifier')
plt.ylabel('% Accuracy')
plt.title('Accuracy Comparision')
plt.legend()
plt.show()

