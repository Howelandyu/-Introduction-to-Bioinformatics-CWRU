

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
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA



p: Parser=Parser("training_text","training_variants")
p.exec()
X_train, y_train, validate=p.build(0.2)
ValiX, ValiY = validate

# print(X_train.shape)

# print(y_train.shape)

# print(y_train)


# print(ValiX.shape)

# print(ValiY.shape)


# pca=PCA(n_components=2)
# pca=pca.fit(X_train)
# X_dr=pca.fit_transform(X_train)
# # print(X_dr.shape)

# # print(X_dr)


# plt.figure()
# plt.scatter(X_dr[:,0],y_train,alpha=1,c='red')
# plt.scatter(X_dr[:,1],y_train,alpha=1,c='blue')
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()

# svc_rbf = SVC()
# parameters_rbf = [
#      {'C': [0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000],
#       'gamma': [0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000],
#       'kernel': ['rbf']}
#  ]
# grid_rbf = GridSearchCV(svc_rbf, parameters_rbf, cv=5, n_jobs=-1)
# grid_rbf.fit(X_train, y_train)
# print(grid_rbf.best_params_)
# print(grid_rbf.best_score_)
# grid_rbf.fit(ValiX, ValiY)
# print(grid_rbf.best_params_)
# print(grid_rbf.best_score_)

# svc_linear=SVC()
# parameter_linear=[{'C': [0.00001,0.00005, 0.000009],'kernel': ['linear']}  ]
# grid_linear = GridSearchCV(svc_linear, parameter_linear, cv=4, n_jobs=-1)
# grid_linear.fit(X_train, y_train)
# print(grid_linear.best_params_)
# print(grid_linear.best_score_)

# svc_poly=SVC()
# parameter_poly=[{'C': [0.00001,0.00005, 0.000009],'kernel': ['poly'],
#                  'degree':[2,3],
#                  'gamma':[0.0001,0.001,0.01,0.1,1,10,100,100],
#                  'coef0':[0.0001,0.001,0.01,0.1,1,10,100,100]
#                 }  ]
# grid_poly = GridSearchCV(svc_poly, parameter_poly, cv=4, n_jobs=-1)
# grid_poly.fit(X_train, y_train)
# print(grid_poly.best_params_)
# print(grid_poly.best_score_)



# svc_sig=SVC()
# parameter_sig=[{'C': [0.03,0.05,0.07],'kernel': ['sigmoid'],
#                  'gamma':[0.002,0.003,0.004],
#                  'coef0':[-0.3,-0.1,-0.03]
#                 }  ]
# grid_sig = GridSearchCV(svc_sig, parameter_sig, cv=4, n_jobs=-1,scoring='accuracy')
# grid_sig.fit(X_train, y_train)
# print(grid_sig.best_params_)
# print(grid_sig.best_score_)


# model=SVC(C=10, gamma=0.01,kernel='rbf', class_weight='balanced')
# model.fit(X_train, y_train)
# pred_result=model.predict(ValiX)
# # print(pred_result)

# print(model.score(X_train,y_train))
# print(model.score(ValiX,ValiY))


model=SVC(C=0.00001, gamma=10000,coef0=5.5,kernel='sigmoid')
model.fit(X_train, y_train)
pred_result=model.predict(ValiX)
# print(pred_result)
print(model.score(X_train,y_train))
print(model.score(ValiX,ValiY))

# pca=PCA(n_components=2)
# # pca=pca.fit(X_train)
# VX_dr=pca.fit_transform(ValiX)
# print(VX_dr.shape)



# plt.figure()
# plt.scatter(VX_dr[:,0],pred_result,alpha=1,c='red')
# plt.scatter(VX_dr[:,1],pred_result,alpha=1,c='blue')
# plt.legend()
# plt.show()

