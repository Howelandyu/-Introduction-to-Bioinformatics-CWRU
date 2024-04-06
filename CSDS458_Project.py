#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from google.colab import files
# from google.colab import drive 
# drive.mount("/drive")
# uploaded = files.upload()


# In[58]:


import numpy as np 
import pandas as pd

from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report
import scipy.sparse as sp
from sklearn.model_selection  import train_test_split
from obj.parser import Parser;


# training_variantsDF = pd.read_csv("training_variants")
# training_variantsDF.head(5)

# training_textDF = pd.read_csv("training_text", sep="\|\|", engine='python',header=None, skiprows=1, names=["ID","Text"])
# training_textDF.head(5)

# training_mergeDF = training_variantsDF.merge(training_textDF,left_on="ID",right_on="ID")

# training_mergeDF.head(5)


# testing_variantsDF = pd.read_csv("test_variants")
# testing_variantsDF.head(5)

# testing_textDF = pd.read_csv("test_text", sep="\|\|", engine='python',header=None, skiprows=1, names=["ID","Text"])
# testing_textDF.head(5)

# testing_mergeDF = testing_variantsDF.merge(testing_textDF,left_on="ID",right_on="ID")
# testing_mergeDF.head(5)


# training_mergeDF['Text'] = training_mergeDF.apply(lambda row: row['Gene'] if pd.isnull(row['Text']) else row['Text'],
#     axis=1
# )

# y=training_mergeDF.Class
# X=training_mergeDF[["Text","Variation",]]


# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# text_vec= CountVectorizer(analyzer='word', stop_words ='english')
# variantion_vec= CountVectorizer(analyzer='word', stop_words ='english')

# text_vec.fit(X_train["Text"])
# text_vec.fit(X_test["Text"])

# variantion_vec.fit(X_train["Variation"])
# variantion_vec.fit(X_test["Variation"])

# variation_tranform_train=variantion_vec.transform(X_train["Variation"])
# variation_tranform_test=variantion_vec.transform(X_test["Variation"])


# text_transform_train= text_vec.transform(X_train["Text"])
# text_transform_test=text_vec.transform(X_test["Text"])


# x_train_final = sp.hstack((variation_tranform_train,text_transform_train))
# x_test_final = sp.hstack((variation_tranform_test,text_transform_test))



# print(x_train_final)
# print(x_test_final)


# svc_model=svm.LinearSVC(C=1.0,dual=False, max_iter=1000)
# svc_model.fit(x_train_final,y_train)


# tfidftransformer = TfidfTransformer()
# tfidf_text_train=tfidftransformer.fit_transform(vect_text.fit_transform(X_train["Text"]))
# tfidf_variation_train=tfidftransformer.fit_transform(vect_variation.fit_transform(X_train["Variation"]))

# tfidf_text_test=tfidftransformer.fit_transform(vect_text.fit_transform(X_test["Text"]))
# tfidf_variation_test=tfidftransformer.fit_transform(vect_variation.fit_transform(X_test["Variation"]))

# x_train_final = sp.hstack((tfidf_text_train,tfidf_variation_train))
# x_test_final = sp.hstack((tfidf_text_test,tfidf_variation_test))

# svc_model=SVC()

p: Parser=Parser("training_text","training_variants")
p.exec()
X_train, y_train, validate=p.build(0.2)
ValiX, ValiY = validate
# print(X_train.shape)
# print("---------------------------")
# print(y_train.shape)
# print("---------------------------")
# print(validate)
# print(ValiX)
# print("---------------------------")
# print(ValiY)
# print("---------------------------")

# svc_model=Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', svm.LinearSVC())])
# svc_model=svm.LinearSVC()
svc_model=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
                max_iter=-1, probability=False, random_state=None, shrinking=True,
                tol=0.001, verbose=False)
svc_model.fit(X_train,y_train)
yHat: np.ndarray = svc_model.predict(ValiX)
# print("model here")
accuracy = svc_model.accuracy_score(ValiY,yHat)
print(accuracy)
# y_pred_class_df=pd.DataFrame(y_pred_class)
# print(y_pred_class_df)


# testing_mergeDF['predicted_class'] = y_pred_class_df

# onehot = pd.get_dummies(testing_mergeDF['predicted_class'])
# testing_mergeDF = testing_mergeDF.join(onehot)


# testing_mergeDF.rename(columns={'1.0':'1','2.0':'2','3.0':'3','4.0':'4','5.0':'5','6.0':'6','7.0':'7','8.0':'8','9.0':'9'},inplace = True)
# testing_mergeDF.head(5)



# submission_df = testing_mergeDF[["ID",1,2,3,4,5,6,7,8,9]]
# submission_df.columns = ['ID', 'class1','class2','class3','class4','class5','class6','class7','class8','class9']



# submission_df.to_csv('submission.csv', index=False)


# In[ ]:




