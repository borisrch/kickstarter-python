import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

df = pd.read_csv('./WA_Fn-UseC_-HR-Employee-Attrition.csv')
le = preprocessing.LabelEncoder()
df['Attrition'] = le.fit_transform(df['Attrition'])
df = df.select_dtypes(exclude=['object'])

target = df['Attrition']
train = df.drop('Attrition',axis = 1)
train.shape

X_train, X_test, y_train, y_test = train_test_split(
    train, target, test_size=0.33, random_state=42)

def train_test_error(y_train,y_test):
    train_error = ((y_train==Y_train).sum())/len(y_train)*100
    test_error = ((y_test==Y_test).sum())/len(Y_test)*100
    print('{}'.format(train_error) + " is the train accuracy")
    print('{}'.format(test_error) + " is the test accuracy")

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)
train_predict = log_reg.predict(X_train)
test_predict = log_reg.predict(X_test)
print('\n ---Logistic Regression---')
print('\n Accuracy:', metrics.accuracy_score(y_test, test_predict))
print('\n Precision:', metrics.precision_score(y_test, test_predict))
print('\n Recall:', metrics.recall_score(y_test, test_predict))
print(metrics.confusion_matrix(y_test, test_predict))

# y_prob = log_reg.predict(train)
# y_pred = np.where(y_prob > 0.5, 1, 0)
# train_test_error(train_predict , test_predict)

gnb = GaussianNB()
gnb.fit(X_train,y_train)
train_predict = gnb.predict(X_train)
test_predict = gnb.predict(X_test)
print('\n ---Naive Bayes---')
print('\n Accuracy:', metrics.accuracy_score(y_test, test_predict))
print('\n Precision:', metrics.precision_score(y_test, test_predict))
print('\n Recall:', metrics.recall_score(y_test, test_predict))
print(metrics.confusion_matrix(y_test,test_predict))


dec = tree.DecisionTreeClassifier()
dec.fit(X_train,y_train)
train_predict = dec.predict(X_train)
test_predict = dec.predict(X_test)
print('\n ---Decision Tree---')
print('\n Accuracy:', metrics.accuracy_score(y_test, test_predict))
print('\n Precision:', metrics.precision_score(y_test, test_predict))
print('\n Recall:', metrics.recall_score(y_test, test_predict))
print(metrics.confusion_matrix(y_test,test_predict))