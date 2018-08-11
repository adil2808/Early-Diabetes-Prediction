import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.utils import resample
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cross_validation import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import scikitplot as skplt
from sklearn.metrics import mean_squared_error


plt.style.use('ggplot')


df = pd.read_csv('diabetes.csv')
df=df.drop('SEQN',axis=1)
df=df.drop('DID341',axis=1)
df=df.drop('DID350',axis=1)
df=df.drop('DIQ350U',axis=1)
df=df.drop('DIQ080',axis=1)
df.head()

df.head()
df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
gg= df.apply(np.int64)

df['DIQ172'] = df['DIQ172'].apply(lambda x:1 if x==1 else 2)


X=gg.iloc[:,:19]
y=gg.iloc[:,19]

print("Processing......")

X_train, X_test, y_train, y_test = train_test_split(df.drop('DIQ172',axis=1), df['DIQ172'], test_size=0.2,random_state=123)


lr = LogisticRegression( random_state=123)
#print(lr.get_params().keys())
cv = ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.2, random_state=123)
tuned_parameters = {'C': [0.1, 0.5]}

print(tuned_parameters)
logistic = GridSearchCV(lr,tuned_parameters, cv=cv, scoring = 'accuracy',verbose = 3, 
                               n_jobs=1)

logistic.fit(X_train,y_train)

#print(logistic.decision_function(X_train))
y_pred=logistic.predict(X_test)

probs = logistic.decision_function(X_test)
preds = probs
print preds
fpr, tpr, threshold = metrics.roc_curve(y_test, preds, pos_label=2)
roc_auc = metrics.auc(fpr, tpr)



print( "Mean Error rate between actual and predicted values\n%s" %mean_squared_error(y_test, y_pred))
 
print("\n");
print("Best parameters set found on training set:")
print(logistic.best_params_)
print("\n");

print("Grid scores on training set:")
means = logistic.cv_results_['mean_test_score']
stds = logistic.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, logistic.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

print("\n");
 
truePos = X_test[((y_pred == 1) & (y_test == y_pred))]
falsePos = X_test[((y_pred == 1) & (y_test != y_pred))]
trueNeg = X_test[((y_pred == 2) & (y_test == y_pred))]
falseNeg = X_test[((y_pred == 2) & (y_test != y_pred))]

TP = truePos.shape[0]
FP = falsePos.shape[0]
TN = trueNeg.shape[0]
FN = falseNeg.shape[0]

print ('True Positive: %s' % TP)
print ('False Positive: %s' % FP)
print ('True Negative: %s' % TN)
print ('False Negative: %s' % FN)

print("\n");



print 'recall: %0.3f' % recall_score(y_test, y_pred, average='macro')  
print 'precision: %0.3f' % precision_score(y_test, y_pred, average='macro') 
print 'accuracy score: %0.3f' % accuracy_score(y_test, y_pred)
print 'F1 score:', f1_score(y_test, y_pred,average='macro')
#confusion_matrix = confusion_matrix(y_test, y_pred)
#print("Confusion matrix:\n%s" % confusion_matrix)


plt.title('ROC Curve - Logistic Regression')
plt.plot(fpr, tpr, color='darkred',
         lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

import pickle
with open('logistic.pkl', 'wb') as fid:
    pickle.dump(logistic, fid,2) 

#fig = plt.figure()
#ax1 = fig.add_subplot(111)

#ax1.scatter(df['DIQ230']==1,df['DIQ010']==1, s=100, alpha=0.7, c='b', marker="o", label='Diabetic')
#ax1.scatter(df['DIQ230']==5,df['DIQ010']==2, s=100, alpha=0.7, c='r', marker="+", label='Non-Diabetic')

#plt.legend(loc='upper left');

#plt.show()

