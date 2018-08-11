import pandas as pd
import numpy  as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.utils import resample
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn import neighbors
plt.style.use('ggplot')

df = pd.read_csv('diabetes.csv')

df=df.drop('SEQN',axis=1)
df=df.drop('DID341',axis=1)
df=df.drop('DID350',axis=1)
df=df.drop('DIQ350U',axis=1)
df=df.drop('DIQ080',axis=1)

df.head()
df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
gg= df.apply(np.int64)


X=gg.iloc[:,:19]
y=gg.iloc[:,19]

X_train, X_test, y_train, y_test = train_test_split(df.drop('DIQ172',axis=1), df['DIQ172'], test_size=0.2,random_state=123)

gb_grid_params = {'n_neighbors': [5,6],
              
              }
#print(gb_grid_params)


knn = KNeighborsClassifier(n_neighbors=5)
knearest = GridSearchCV(knn, gb_grid_params,cv=10,scoring='accuracy',verbose = 3, n_jobs=1);



knearest.fit(X_train,y_train)


#print(knearest.predict_proba(X_train))
y_pred=knearest.predict(X_test)

probs = knearest.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds, pos_label=2)
roc_auc = metrics.auc(fpr, tpr)

for i in range(1,100):
   if y_pred[i] == 1  :
    print ('Chance of diabetes in future with a probability:\n%s' %max(probs[i]))
   else:
       print 'No chance of diabetes in future\n%s' %min(probs[i]);


print( "Mean Error rate between actual and predicted values\n%s" %mean_squared_error(y_test, y_pred))
 
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

plt.title('ROC Curve - K-NearestNeighbor')
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
with open('knearest.pkl', 'wb') as fid:
    pickle.dump(knearest, fid,2) 
    