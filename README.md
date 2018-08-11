# Early Diabetes Prediction Using Ensemble Learning

Predicting the onset of diabetes at early stage is a complex problem as it is a lifestyle disease and requires intervention of 
self-assessment prevention at every step. To reduce uncertainty, an ensemble learning (collection of machine learning models) 
are used for prediction of diabetes risk in future.

Two objectives of this prediction system are:
1) Whether the person have a risk of diabetes in future or not?
2) What is the risk probability associated with it ?

The Dataset used is NHANES (National Health and Nutrition Examination Survey), which is questionnaire of 10,176 patients. 
More info of this dataset can be obtained at (https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/DIQ_H.htm)

Libraries used are: Scikit-Learn, Pandas, Numpy

Once model trained, pickle the model for ensemble learning. This is useful as it requires no re-training
