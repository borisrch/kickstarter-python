'''
Ada Boost and 10-fold cross validation method calls have been commented
out (takes a long time to compute). Uncomment the relevant method calls to use them. 
'''
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib as mpl

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from sklearn import metrics
from matplotlib.ticker import PercentFormatter
import numpy as np
import seaborn as sns

# Model libraries:
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


df_2016 = pd.read_csv('ks-projects-201612.csv', encoding="ISO-8859-1", low_memory=False)
df_2018 = pd.read_csv('ks-projects-201801.csv', encoding="ISO-8859-1", low_memory=False)

# Display each dataset
print(df_2016.head())
print(df_2018.head())

# get coloumns and data types
df_2016.info()
df_2018.info()

df_2018['deadline'] = pd.to_datetime(df_2018['deadline'])
print(df_2018['deadline'].sort_values().head())

# get dimensions of each dataset
print(df_2018.shape)
print(df_2016.shape)

df_2016.columns = [s.replace(' ', '') for s in df_2016.columns]
print(df_2016[df_2016['deadline'] != 'USD']['deadline'].sort_values().head())

# The above code shows us to use only dataset 2018 because it is essentially a more substantial dataset
df = df_2018
# get null values and the percentage of them
print(df.isnull().sum())
print(df.isnull().sum() / df.shape[0])

# Find counts of unique subcategories
print(df.category.value_counts())

# Find counts of unique main categories
print(df.main_category.value_counts())

# Find how many unique values are in main category and subcategory and determine which one keeps dimensionality low
print(len(df.category.unique()))
print(len(df.main_category.unique()))

# Convert date to common datetime format
df.launched = pd.to_datetime(df.launched)

# Investigate countries
print(df.country.value_counts())

# Clean up the country codes - converting 'N,0"' to 'NO'
df.country = df.country.replace(to_replace='N,0"', value='NO')

# Define success variable. Success is defined by a project raising at least as much as their goal.
df['success'] = (df.usd_goal_real <= df.usd_pledged_real) * 1
print(df.success.describe())

# Define duration variable to see if the timeline of a project influences the chances at success.
df['duration'] = (df.deadline - df.launched).astype('timedelta64[h]')

# Perform one hot encode on categorical variables, dropping unneeded variables
df_encoded = pd.get_dummies(df.drop(labels=['name', 'launched', 'deadline', 'state',
                                            'category', 'currency', 'usd pledged', 'pledged',
                                            'ID', 'goal'], axis=1),
                            columns=['main_category', 'country'])
print(df_encoded)
# Add a variable that shows average pledge for each project
df_encoded['average_backing'] = (df_encoded['usd_pledged_real'] / (df_encoded['backers'] + 1))
print(df_encoded['average_backing'])

# Data Analysis Part 1

# This helper function plots a particular attribute in the dataset.
def bar_plot(data, title):
    ax = data.plot(kind = 'bar')
    plt.title(title)
    ax.yaxis.set_major_formatter(PercentFormatter())
    plt.show();

# Total projects by country
bar_plot((df.country.value_counts()/df.shape[0]*100), "Total Kickstarter Projects by Country")

# Successful projects by country
bar_plot((df[df.usd_pledged_real>=df.usd_goal_real].country.value_counts()/
          df[df.usd_pledged_real>=df.usd_goal_real].shape[0]*100), 
        "Successful Kickstarter Projects by Country")

# Project categories by country
bar_plot((df.main_category.value_counts()/df.shape[0]*100), 
             "Kickstarter Projects by Category")

# Successful project categories by country
bar_plot((df[df.usd_pledged_real>=df.usd_goal_real].main_category.value_counts()/
              df[df.usd_pledged_real>=df.usd_goal_real].shape[0]*100), 
             "Successful Kickstarter Projects by Category")

# Histogram of Successful Project Duration
plt.hist(df_encoded[df_encoded.success==1].duration, bins=20)
plt.title('Successful Project Duration')
plt.xlabel('# of hours')
plt.ylabel('# of projects')
plt.show();

# mean, std, quartiles of Successful Projects (Duration)  
df_encoded[df_encoded.success==1].duration.describe()

# mean, std, quartiles of Failed Projects (Duration)  
df_encoded[df_encoded.success==0].duration.describe()

# mean, std, quartiles of Successful Projects (Goal)
df_encoded[df_encoded.success==1].usd_goal_real.describe()

# mean, std, quartiles of Failed Projects (Goal)
df_encoded[df_encoded.success==0].usd_goal_real.describe()

# mean, std, quartiles of Successful Projects (Backing)
df_encoded[df_encoded.success==1].average_backing.describe()

# mean, std, quartiles of Failed Projects (Backing)
df_encoded[df_encoded.success==0].average_backing.describe()

# bar plot of Kickstarter Projects by Success
bar_plot((df_encoded.success.value_counts()/df.shape[0]*100),
             "Kickstarter Projects by Success")

def Corr_plot(attr, s):
    corr = df_encoded[attr].corr()
    fig, ax = plt.subplots(figsize = s)
    sns.heatmap(corr, 
                xticklabels = corr.columns, 
                yticklabels = corr.columns, 
                ax=ax, 
                linewidths = 0.01,
                cmap='Blues');
                
Corr_plot(['success','duration', 'usd_goal_real', 
                        'usd_pledged_real', 'backers', 'average_backing'], (5,5))

# This graph visualizes the Pearson product-moment correlation coefficient and while there
# is no direct correlation between success and any of the attributes, there is a correlation between
# Backers and usd_pledged_real
    
    
# Function to implement RandomForestClassifier given train/test set as args
def RFC_model(randomState, X_train, X_test, y_train, y_test):
    '''
    INPUT:
            randomState - the random state parameter for random forest classifier
            X_train - training set split for the independent variables
            X_test - testing set split for independent variables
            y_train - training set split for the dependent variables
            y_test - testing set split for the dependent variables
    OUTPUT: prints accuracy of the random forest classifier
    '''
    rand_forest = RandomForestClassifier()
    rand_forest.fit(X_train, y_train)
    forest_test_predictions = rand_forest.predict(X_test)
    print('\n --Random Forests Classifier--')
    print(accuracy_score(y_test, forest_test_predictions))
    
    # Evaluation for Random Forests
    print('\n Confusion Matrix')
    cnf_matrix = metrics.confusion_matrix(y_test, forest_test_predictions)
    print(cnf_matrix)
    print('\n Accuracy:', metrics.accuracy_score(y_test, forest_test_predictions))
    print('\n Precision:', metrics.precision_score(y_test, forest_test_predictions))
    print('\n Recall:', metrics.recall_score(y_test, forest_test_predictions))
    y_predict_proba = rand_forest.predict_proba(X_test)[::, 1]
    fpr, tpr, _  = metrics.roc_curve(y_test, y_predict_proba)
    auc = metrics.roc_auc_score(y_test, y_predict_proba)
    mpl.style.use('seaborn')
    plt.legend(loc = 4, fontsize = 16)
    plt.plot(fpr, tpr, label="AUC: " + str(auc))    
    plt.title('Random Forest Classifier ROC Curve')
    print('\n ROC Curve')
    plt.show()
    print('\n AUC:', auc)

X_train, X_test, y_train, y_test = train_test_split(df_encoded.drop(['success'], axis='columns').values,
                                                    df_encoded.success, 
                                                    test_size=0.2)

# 0.99441% accuracy, seems incorrect or dataset is biased.
RFC_model(42, X_train, X_test, y_train, y_test)
LR_model(X_train, X_test, y_train, y_test)


# Create training set without usd_pledged and backers.
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
        df_encoded.drop(['success', 'usd_pledged_real', 'backers'], axis='columns').values,
        df_encoded.success,
        test_size = 0.2)


# Function to implement Logistic Regression model given train/test set as args
def LR_model(X_train, X_test, y_train, y_test):
    lr = LogisticRegression(penalty='l2', solver='liblinear')
    fit = lr.fit(X_train, y_train)
    predict = lr.predict(X_test)
    print('\n --Logistic Regression--')
    print('\n Accuracy by success')
    print(pd.crosstab(predict, y_test))
    print('\n Percentage accuracy')
    print(lr.score(X_test, y_test))
    
    # Evaluation for Logistic Regression
    print('\n Confusion Matrix')
    cnf_matrix = metrics.confusion_matrix(y_test, predict)
    print(cnf_matrix)
    print('\n Accuracy:', metrics.accuracy_score(y_test, predict))
    print('\n Precision:', metrics.precision_score(y_test, predict))
    print('\n Recall:', metrics.recall_score(y_test, predict))
    y_predict_proba = lr.predict_proba(X_test)[::, 1]
    fpr, tpr, _  = metrics.roc_curve(y_test, y_predict_proba)
    auc = metrics.roc_auc_score(y_test, y_predict_proba)
    plt.plot(fpr, tpr, label="AUC: " + str(auc))
    mpl.style.use('seaborn')
    plt.legend(loc = 4, fontsize = 16)
    plt.title('Logistic Regression ROC Curve')
    print('\n ROC Curve')
    plt.show()
    print('\n AUC:', auc)


def KNN_model(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=2, n_jobs=-1)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    predict = knn.predict(X_test)
    print('\n --KNN--')
    print('\n Percentage accuracy')
    print(acc)
    
    # Evaluation for KNN
    print('\n Confusion Matrix')
    cnf_matrix = metrics.confusion_matrix(y_test, predict)
    print(cnf_matrix)
    print('\n Accuracy:', metrics.accuracy_score(y_test, predict))
    print('\n Precision:', metrics.precision_score(y_test, predict))
    print('\n Recall:', metrics.recall_score(y_test, predict))
    y_predict_proba = knn.predict_proba(X_test)[::, 1]
    fpr, tpr, _  = metrics.roc_curve(y_test, y_predict_proba)
    auc = metrics.roc_auc_score(y_test, y_predict_proba)
    plt.plot(fpr, tpr, label="AUC: " + str(auc))
    mpl.style.use('seaborn')
    plt.legend(loc = 4, fontsize = 16)
    plt.title('K Nearest Neighbours ROC Curve')
    print('\n ROC Curve')
    plt.show()
    print('\n AUC:', auc)
    

def ADA_model(X_train, X_test, y_train, y_test):
    tree = DecisionTreeClassifier(max_depth=1)
    bdt = AdaBoostClassifier(tree, algorithm='SAMME', n_estimators = 200)
    bdt.fit(X_train, y_train)
    acc = bdt.score(X_test, y_test)
    predict = bdt.predict(X_test)
    
    print('\n --Ada Boost Classifier--')
    print('\n Percentage accuracy')
    print(acc)
    
    # Evaluation for Ada Boost Classifier
    print('\n Confusion Matrix')
    cnf_matrix = metrics.confusion_matrix(y_test, predict)
    print(cnf_matrix)
    print('\n Accuracy:', metrics.accuracy_score(y_test, predict))
    print('\n Precision:', metrics.precision_score(y_test, predict))
    print('\n Recall:', metrics.recall_score(y_test, predict))
    y_predict_proba = bdt.predict_proba(X_test)[::, 1]
    fpr, tpr, _  = metrics.roc_curve(y_test, y_predict_proba)
    auc = metrics.roc_auc_score(y_test, y_predict_proba)
    plt.plot(fpr, tpr, label="AUC: " + str(auc))
    mpl.style.use('seaborn')
    plt.legend(loc = 4, fontsize = 16)
    plt.title('Ada Boost Classifier ROC Curve')
    print('\n ROC Curve')
    plt.show()
    print('\n AUC:', auc)

def Tree_model(X_train, X_test, y_train, y_test):
    tree = DecisionTreeClassifier(max_depth=1)
    tree.fit(X_train, y_train)
    acc = tree.score(X_test, y_test)
    predict = tree.predict(X_test)
    print('\n --Decision Tree--')
    print('\n Percentage accuracy')
    print(acc)
    
    # Evaluation for Decision Tree
    print('\n Confusion Matrix')
    cnf_matrix = metrics.confusion_matrix(y_test, predict)
    print(cnf_matrix)
    print('\n Accuracy:', metrics.accuracy_score(y_test, predict))
    print('\n Precision:', metrics.precision_score(y_test, predict))
    print('\n Recall:', metrics.recall_score(y_test, predict))
    y_predict_proba = tree.predict_proba(X_test)[::, 1]
    fpr, tpr, _  = metrics.roc_curve(y_test, y_predict_proba)
    auc = metrics.roc_auc_score(y_test, y_predict_proba)
    plt.plot(fpr, tpr, label="AUC: " + str(auc))
    mpl.style.use('seaborn')
    plt.legend(loc = 4, fontsize = 16)
    plt.title('Decision Tree ROC Curve')
    print('\n ROC Curve')
    plt.show()
    print('\n AUC:', auc)
    

# Run these functions individually after the init run.

# --- Random Forest Classifier ---
    # Accuracy: 0.7935774365204072
    # Precision: 0.7304790791724997
    # Recall: 0.6834139775166442
    # AUC: 0.8724931200626465
RFC_model(42, X_train_2, X_test_2, y_train_2, y_test_2)

# --- Logistic Regression Classifier ---
    # Accuracy: 0.7153552612467484
    # Precision: 0.7457725464190982
    # Recall: 0.3273183686833776
    # AUC: 0.8156880920433939
LR_model(X_train_2, X_test_2, y_train_2, y_test_2)

# --- Decision Tree Classifier ---
    # Accuracy: 0.6919836795056317
    # Precision: 0.5436260500562917
    # Recall: 0.9178272471396718
    # AUC: 0.7410473262323132
Tree_model(X_train_2, X_test_2, y_train_2, y_test_2)

# --- KNN Classifier ---
    # Accuracy: 0.7417374196189244
    # Precision: 0.7244874844263224
    # Recall: 0.4654200167351839
    # AUC: 0.7924082658794459
KNN_model(X_train_2, X_test_2, y_train_2, y_test_2)

# Note: takes a while to compute
# --- Ada Boost Classifier ---
    # Accuracy: 0.8062799572181215
    # Precision: 0.7238993710691823
    # Recall: 0.7537381307527194
    # AUC: 0.8779823184619542
ADA_model(X_train_2, X_test_2, y_train_2, y_test_2)
    
# --- Cross Validation ---

from sklearn.model_selection import cross_val_score

# Utility function to calculate 95% CI of a given array
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    print('Margin of Error: ', h)
    print('Error Rate: ', 1-m)
    return m, m-h, m+h


# --- Cross validation: Random Forests ---
    # Margin of Error:  0.0012592184196155202
    # Error Rate:  0.2053050012730635
    # RFC cross validation mean accuracy at 95% CI (0.7946949987269365, 0.793435780307321, 0.795954217146552)
rfc_scores = cross_val_score(
        RandomForestClassifier(), 
        df_encoded.drop(['success', 'usd_pledged_real', 'backers'], axis='columns').values,
        df_encoded.success,
        cv=10)
print('RFC cross validation mean accuracy at 95% CI', mean_confidence_interval(rfc_scores))


# --- Cross validation: Logistic Regression ---
    # Margin of Error: 0.0012322849608193214
    # Eror Rate: 0.2053050012730635
    # Mean accuracy: 0.7182651449323484, CI: 0.7170328599715291, 0.7194974298931677
lr_scores = cross_val_score(
        LogisticRegression(penalty='l2', solver='liblinear'), 
        df_encoded.drop(['success', 'usd_pledged_real', 'backers'], axis='columns').values,
        df_encoded.success,
        cv=10)
print('LR cross validation mean accuracy at 95% CI', mean_confidence_interval(lr_scores))

# --- Cross validation: KNN ---
    # Margin of Error: 0.0010318501379904907
    # Eror Rate: 0.2568946944906566
    # Mean accuracy: 0.7431053055093434, CI: 0.7420734553713529, 0.7441371556473338
knn_scores = cross_val_score(
        KNeighborsClassifier(n_neighbors=2, n_jobs=-1), 
        df_encoded.drop(['success', 'usd_pledged_real', 'backers'], axis='columns').values,
        df_encoded.success,
        cv=10)
print('KNN cross validation mean accuracy at 95% CI', mean_confidence_interval(knn_scores))


# --- Cross validation: Decision Tree ---
    # Margin of Error: 0.0010781728062298009
    # Eror Rate: 0.3087141281178931
    # Mean accuracy: 0.6912858718821069, CI: 0.6902076990758771, 0.6923640446883367
tree_scores = cross_val_score(
        DecisionTreeClassifier(max_depth=1), 
        df_encoded.drop(['success', 'usd_pledged_real', 'backers'], axis='columns').values,
        df_encoded.success,
        cv=10)
print('Decision Tree cross validation mean accuracy at 95% CI', mean_confidence_interval(tree_scores))


# --- Cross validation: Ada Boost ---
    # Margin of Error: 0.0010041684129616206
    # Eror Rate: 0.19436645574745715
    # Mean accuracy: 0.8056335442525429, CI: 0.8046293758395813, 0.8066377126655044
    # [0.80613727, 0.80455278, 0.80703005, 0.8042043 , 0.80623779,
    #  0.80497016, 0.80576243, 0.80467966, 0.8042043 , 0.80855671]
ada_scores = cross_val_score(
        AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm='SAMME', n_estimators = 200), 
        df_encoded.drop(['success', 'usd_pledged_real', 'backers'], axis='columns').values,
        df_encoded.success,
        cv=10)
print('Ada Boost cross validation mean accuracy at 95% CI', mean_confidence_interval(ada_scores))

print('T-Test for AdaBoost and RFC')
tt = scipy.stats.ttest_ind(ada_scores, rfc_scores)

# Ttest_indResult(statistic=14.62702465172062, pvalue=1.9629611278610977e-11)
print(tt)

