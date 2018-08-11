import pandas as pd
import numpy  as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

plt.style.use('ggplot')

df = pd.read_csv('diabetes.csv')
df=df.drop('SEQN',axis=1)
df=df.drop('DID341',axis=1)
df=df.drop('DID350',axis=1)
df=df.drop('DIQ350U',axis=1)
df=df.drop('DIQ080',axis=1)

df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
gg= df.apply(np.int64)



X=gg.iloc[:,:18]
y=gg.iloc[:,18]
print("Processing.......")

import pickle

pkl_file = open('gbc.pkl', 'rb')
gbc = pickle.load(pkl_file)

pkl_file = open('rfc.pkl', 'rb')
rfc = pickle.load(pkl_file)

pkl_file = open('knearest.pkl', 'rb')
knearest = pickle.load(pkl_file)

pkl_file = open('logistic.pkl', 'rb')
logistic = pickle.load(pkl_file)

X_train, X_test, y_train, y_test = train_test_split(df.drop('DIQ172',axis=1), df['DIQ172'], test_size=0.2,random_state=123)


eclf1 = VotingClassifier(estimators=[('lr', logistic), ('rfc', rfc), ('knn', knearest), ('gbc', gbc)], voting='soft')
eclf1 = eclf1.fit(X_train, y_train)
y_pred1=eclf1.predict(X_test)


eprobs = eclf1.predict_proba(X_test)
epreds = eprobs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, epreds, pos_label=2)
roc_auc = metrics.auc(fpr, tpr)

for i in range(1,4):
   if y_pred1[i] == 1 :
    print ('Chance of diabetes in future with a probability:\n%s' %max(eprobs[i]))
   else:
       print 'No chance of diabetes in future\n%s' %min(eprobs[i]);


print( "Mean Error rate between actual and predicted values\n%s" %mean_squared_error(y_test, y_pred1))
 
print("\n");

truePos = X_test[((y_pred1 == 1) & (y_test == y_pred1))]
falsePos = X_test[((y_pred1 == 1) & (y_test != y_pred1))]
trueNeg = X_test[((y_pred1 == 2) & (y_test == y_pred1))]
falseNeg = X_test[((y_pred1 == 2) & (y_test != y_pred1))]

TP = truePos.shape[0]
FP = falsePos.shape[0]
TN = trueNeg.shape[0]
FN = falseNeg.shape[0]

print ('True Positive: %s' % TP)
print ('False Positive: %s' % FP)
print ('True Negative: %s' % TN)
print ('False Negative: %s' % FN)

print("\n");

    

print 'recall: %0.3f' % recall_score(y_test, y_pred1, average='macro')  
print 'precision: %0.3f' % precision_score(y_test, y_pred1, average='macro') 
print 'accuracy score: %0.3f' % accuracy_score(y_test, y_pred1)
print 'F1 score:', f1_score(y_test, y_pred1,average='macro')

plt.title('ROC Curve - Ensemble')
plt.plot(fpr, tpr, color='darkred',
         lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

probs1 = logistic.predict_proba(X_test)
preds1 = probs1[:,1]
lr_fpr, lr_tpr, threshold = metrics.roc_curve(y_test, preds1, pos_label=2)
lr_roc_auc = metrics.auc(lr_fpr, lr_tpr)

probs2 = rfc.predict_proba(X_test)
preds2 = probs2[:,1]
rfc_fpr, rfc_tpr, threshold = metrics.roc_curve(y_test, preds2, pos_label=2)
rfc_roc_auc = metrics.auc(rfc_fpr, rfc_tpr)

probs2 = knearest.predict_proba(X_test)
preds2 = probs2[:,1]
knearest_fpr, knearest_tpr, threshold = metrics.roc_curve(y_test, preds2, pos_label=2)
knearest_roc_auc = metrics.auc(knearest_fpr, knearest_tpr)


probs4 = gbc.predict_proba(X_test)
preds4 = probs4[:,1]
gbc_fpr, gbc_tpr, threshold = metrics.roc_curve(y_test, preds4, pos_label=2)
gbc_roc_auc = metrics.auc(gbc_fpr, gbc_tpr)

plt.figure()
plt.plot(lr_fpr, lr_tpr, color='aqua',
         lw=2, label='Logistic Regression (area = %0.2f)' % lr_roc_auc)
plt.plot(rfc_fpr, rfc_tpr, color='darkgreen',
         lw=2, label='Random Forest (area = %0.2f)' % rfc_roc_auc)
plt.plot(rfc_fpr, rfc_tpr, color='darkorange',
         lw=2, label='KNearest Neighbor (area = %0.2f)' % knearest_roc_auc)
plt.plot(gbc_fpr, gbc_tpr, color='darkblue',
         lw=2, label='Gradient Boosting(area = %0.2f)' % gbc_roc_auc)
plt.plot(gbc_fpr, gbc_tpr, color='darkred',
         lw=2, label='Ensemble(AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(' Comparative ROC Curve - All Models')
plt.legend(loc="lower right")
plt.show()

import pickle
with open('eclf1.pkl', 'wb') as fid:
    pickle.dump(eclf1, fid,2) 
    




#confusion_matrix = confusion_matrix(y_test, y_pred)
#print("Confusion matrix:\n%s" % confusion_matrix)

#lr_probas = nclf1.fit(X_train, y_train).predict_proba(X_test)
#rfc_probas = nclf2.fit(X_train, y_train).predict_proba(X_test)

#probas_list = [lr_probas, rfc_probas]
#clf_names = ['Logistic Regression','Random Forest']
#skplt.metrics.plot_calibration_curve(y_test,probas_list,clf_names)
#plt.show()
