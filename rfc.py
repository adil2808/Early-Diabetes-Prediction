import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.utils import resample
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import mean_squared_error
import pickle
import os

plt.style.use('ggplot')

df = pd.read_csv('diabetes.csv')
df=df.drop('SEQN',axis=1)
df=df.drop('DID341',axis=1)
df=df.drop('DID350',axis=1)
df=df.drop('DIQ350U',axis=1)
df=df.drop('DIQ080',axis=1)

df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
gg= df.apply(np.int64)



df['DIQ172'] = df['DIQ172'].apply(lambda x:1 if x==1 else 2)

X=gg.iloc[:,:19]
y=gg.iloc[:,19]

n_estimator = 10

print("Processing..........");
X_train, X_test, y_train, y_test = train_test_split(df.drop('DIQ172',axis=1), df['DIQ172'], test_size=0.2,random_state=123)

print X_test.shape
crfc = RandomForestClassifier(max_depth=3, n_estimators=n_estimator, random_state=123) 

param_grid = { 
    'n_estimators': [100,200],
    'max_depth': [3]
}

print (param_grid)

rfc = GridSearchCV(estimator=crfc, param_grid=param_grid, cv= 10, scoring='accuracy',
                               verbose = 3, 
                               n_jobs=1)

rfc.fit(X_train,y_train)


y_pred=rfc.predict(X_test)

probs = rfc.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds, pos_label=2)
roc_auc = metrics.auc(fpr, tpr)

for i in range(1,38):
   if y_pred[i] == 1  :
    print ('Chance of diabetes in future with a probability:\n%s' %max(probs[i]))
   else:
       print 'No chance of diabetes in future\n%s' %min(probs[i]);


print( "Mean Error rate between actual and predicted values\n%s" %mean_squared_error(y_test, y_pred))
 
print("\n");
    
print("Best parameters set found on training set:")
print(rfc.best_params_)
print("\n");

print("Grid scores on development set:")
means = rfc.cv_results_['mean_test_score']
stds = rfc.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, rfc.cv_results_['params']):
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

plt.title('ROC Curve - Random Forest')
plt.plot(fpr, tpr, color='darkred',
         lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n%s" % confusion_matrix)

with open('rfc.pkl', 'wb') as fid:
    pickle.dump(rfc, fid,2) 

#with open('rfc.pkl', 'wb') as fid:
 #   pickle.dump(rfc, fid,2) 

#cat = df.drop('DIQ160',axis=1)
#index_dict = dict(zip(cat.columns,range(cat.shape[1])))

#with open('cat', 'wb') as fid:
#    pickle.dump(index_dict, fid,2)  