
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.linear_model import SGDRegressor
# from sklearn.linear_model import BayesianRidge, LinearRegression
# from scipy import stats
# np.random.seed(58)
# d_input = 8
# np.random.seed(58)
# # data = pd.read_csv('/home/mzhu/madesi/mzhu_code/windmill14.csv')
# # data2 = pd.read_csv('/home/mzhu/madesi/mzhu_code/windmill14_ab.csv')
# # data = pd.DataFrame(data).dropna()
# # data2 = pd.DataFrame(data2).dropna()
# # dmean, dstd = data.mean(), data.std()
# # data = (data - dmean) / dstd
# # dmean2, dstd2 = data2.mean(), data2.std()
# # data2 = (data2 - dmean2) / dstd2
# # train = data.sample(frac=0.8, random_state=58)
# # test = data.drop(train.index)
# # train2 = data2.sample(frac=0.8, random_state=58)
# # test2 = data2.drop(train2.index)
# # x, y = train.iloc[:, :d_input].values, train.iloc[:, d_input:].values
# # y_d = y.shape[1]
# # x1, y1 = test.iloc[:, :d_input].values, test.iloc[:, d_input:].values
# # x2, y2 = test2.iloc[:, :d_input].values, test2.iloc[:, d_input:].values
# # print(x.shape)
# #
# # RMSE=[]
# # RMSE_AB=[]
# # for k in range(y_d):
# #     all_rmse = []
# #     all_rmse_abnormal = []
# #     # # sgdregressor on full data
# #     y_loop = y[:,k]
# #     # clf = SGDRegressor()
# #     # clf.fit(x, y_loop)
# #     # clf = LinearRegression()
# #     # bayesian ridge regression
# #     # Fit the Bayesian Ridge Regression and an OLS for comparison
# #     clf = BayesianRidge(compute_score=True)
# #     clf.fit(x, y_loop)
# #     y_pred = clf.predict(x1)
# #     y_pred2 = clf.predict(x2)
# #     mu_t = y1[:,k]
# #     mu_t2 = y2[:, k]
# #     sqe = (y_pred - mu_t) ** 2
# #     sqe2 = (y_pred2 - mu_t2) ** 2
# #     all_rmse.append(np.sqrt(sqe))
# #     all_rmse_abnormal.append(np.sqrt(sqe2))
# #     RMSE.append(all_rmse)
# #     RMSE_AB.append(all_rmse_abnormal)
# # RMSE = np.array(RMSE).reshape((6,1000))
# # RMSE_AB = np.array(RMSE_AB).reshape((6,1000))
# # RMSE = np.mean(RMSE,axis=0)
# # RMSE_AB = np.mean(RMSE_AB,axis=0)
# # print(RMSE_AB.shape)
# # np.savetxt('brr.csv', RMSE, delimiter=',')
# # np.savetxt('brr_ab.csv', RMSE_AB, delimiter=',')
# # brr
#
# from sklearn.metrics import roc_curve, auc
# from sklearn.model_selection import train_test_split
# import pylab as plt
# import numpy as np
#
# #
# brr = np.genfromtxt('/home/mzhu/madesi/mzhu_code/brr.csv', delimiter=',')
# brr_ab = np.genfromtxt('/home/mzhu/madesi/mzhu_code/brr_ab.csv',delimiter=',')
# sgdr = np.genfromtxt('/home/mzhu/madesi/mzhu_code/sgdr.csv',delimiter=',')
# sgdr_ab = np.genfromtxt('/home/mzhu/madesi/mzhu_code/sgdr_ab.csv',delimiter=',')
# singlegp = np.genfromtxt('/home/mzhu/madesi/mzhu_code/singlegp.csv',delimiter=',')
# singlegp_ab = np.genfromtxt('/home/mzhu/madesi/mzhu_code/singlegp_ab.csv',delimiter=',')
# lr = np.genfromtxt('/home/mzhu/madesi/mzhu_code/lr.csv', delimiter=',')
# lr_ab = np.genfromtxt('/home/mzhu/madesi/mzhu_code/lr_ab.csv',delimiter=',')
# spgpn = np.genfromtxt('/home/mzhu/madesi/mzhu_code/spgpn.csv', delimiter=',')
# spgpn_ab = np.genfromtxt('/home/mzhu/madesi/mzhu_code/spgpn_ab.csv',delimiter=',')
# y_test1 = np.zeros(1000)
# y_test2 = np.ones(1000)
# y_test_ = np.concatenate((y_test2,y_test1))
# y_test = np.concatenate((y_test1,y_test2))
# brr_all = np.concatenate((brr,brr_ab))
# sgdr_all = np.concatenate((sgdr,sgdr_ab))
# singlegp_all = np.concatenate((singlegp,singlegp_ab))
# spgpn_all = np.concatenate((spgpn,spgpn_ab))
# lr_all = np.concatenate((lr,lr_ab))
# # fpr1, tpr1, thresholds1  =  roc_curve(y_test, brr_all)
# # roc_auc1 = auc(fpr1,tpr1)
# fpr2, tpr2, thresholds2  =  roc_curve(y_test, sgdr_all)
# roc_auc2 = auc(fpr2,tpr2)
# fpr3, tpr3, thresholds3  =  roc_curve(y_test,singlegp_all)
# roc_auc3 = auc(fpr3,tpr3)
# fpr4, tpr4, thresholds4  =  roc_curve(y_test, lr_all)
# roc_auc4 = auc(fpr4,tpr4)
# fpr5, tpr5, thresholds5  =  roc_curve(y_test, spgpn_all)
# roc_auc5 = auc(fpr5,tpr5)
# plt.figure(figsize=(6,6))
# plt.title('rmse ROC')
# # plt.plot(fpr1, tpr1, 'r-', label = 'bayesian_ridge_regression Val AUC = %0.3f' % roc_auc1)
# plt.plot(fpr2, tpr2, 'b-', label = 'SGD_regression Val AUC = %0.3f' % roc_auc2)
# plt.plot(fpr3, tpr3, 'g-', label = 'GP Val AUC = %0.3f' % roc_auc3)
# plt.plot(fpr4, tpr4, 'y-', label = 'linear_regression Val AUC= %0.3f' % roc_auc4)
# plt.plot(fpr5, tpr5, 'r-', label = 'SPGPN Val AUC= %0.3f' % roc_auc5)
# # plt.plot(fpr1, tpr1, 'b-', label = 'original SPNGP Val AUC = %0.3f' % roc_auc1)
# # plt.plot(fpr2, tpr2, 'g-', label = 'improved GPSPN Val AUC = %0.3f' % roc_auc2)
# # plt.plot(fpr1, tpr1,'ro-',fpr2, tpr2,'r^-',fpr3, tpr3)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()
# plt.savefig('roc.pdf')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import arff

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# d_input = 32
# # dataarff = arff.loadarff('/home/mzhu/madesi/mzhu_code/scm20d.arff')
# # data = pd.DataFrame(dataarff[0])
# # data = pd.read_csv('/home/mzhu/madesi/mzhu_code/windmill14.csv')
# # data = pd.read_csv('/home/mzhu/madesi/mzhu_code/windmill14_ab.csv')
# data = pd.read_csv('/home/mzhu/madesi/mzhu_code/WECs_DataSet/Adelaide_Data.csv')
# data = pd.DataFrame(data).dropna()
# train = data.sample(frac=0.8, random_state=58)
# test = data.drop(train.index)
# # data_ab = pd.DataFrame(data_ab).dropna()
# # df = pd.read_csv('/home/mzhu/madesi/mzhu_code/VAR.csv',header=None)
# # df = pd.DataFrame(df).dropna()  # miss = data.isnull().sum()/len(data)
# # data2 = df.T
# x_, y_ = train.iloc[:, :d_input].values, train.iloc[:, d_input:].values
# y_d = y_.shape[1]
# x1_, y1_ = test.iloc[:, :d_input].values, test.iloc[:, d_input:].values
# print(x_.shape)
# std1, mu1= np.std(x_,axis=0), np.mean(y_,axis=0)
# std2, mu2= np.std(y_,axis=0), np.mean(x_,axis=0)
# x_digits = (x_-mu2)/ std1  # normalized train_x
# x1 = (x1_-mu2)/std1 # test_x
# y_digits= (y_-mu1)/std2# train_y
# y1 = (y1_-mu1)/std2 #test_y
# # Define a pipeline to search for the best combination of PCA truncation
# # and classifier regularization.
# pca = PCA()
# # set the tolerance to a large value to make the example faster
# logistic = LogisticRegression(max_iter=10000, tol=0.1)
# pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

#
# # Parameters of pipelines can be set using ‘__’ separated parameter names:
# param_grid = {
#     'pca__n_components': [5, 15, 30, 45, 64],
#     'logistic__C': np.logspace(-4, 4, 4),
# }
# search = GridSearchCV(pipe, param_grid, n_jobs=-1)
# search.fit(x_digits, y_digits)
# print("Best parameter (CV score=%0.3f):" % search.best_score_)
# print(search.best_params_)

# Plot the PCA spectrum
# pca.fit(x_digits)
#
# fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
# ax0.plot(np.arange(1, pca.n_components_ + 1),
#          pca.explained_variance_ratio_, '+', linewidth=2)
# ax0.set_ylabel('PCA explained variance ratio')
# # ax0.axvline(search.best_estimator_.named_steps['pca'].n_components,
# #             linestyle=':', label='n_components chosen')
# ax0.legend(prop=dict(size=12))
#
# plt.show()
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import pylab as plt
import numpy as np

gpgpn_rmse = np.genfromtxt('/home/mzhu/madesi/mzhu_code/SPGPN_windmill_rmse.csv', delimiter=',')
gpspn_rmse_ab = np.genfromtxt('/home/mzhu/madesi/mzhu_code/SPGPN_windmill_rmse_ab.csv',delimiter=',')
lr_rmse = np.genfromtxt('/home/mzhu/madesi/mzhu_code/lr_windmill_rmse.csv', delimiter=',')
lr_rmse_ab = np.genfromtxt('/home/mzhu/madesi/mzhu_code/lr_windmill_rmse_ab.csv',delimiter=',')
MOGP_rmse = np.genfromtxt('/home/mzhu/madesi/mzhu_code/MOGP_windmill_rmse.csv', delimiter=',')
MOGP_rmse_ab = np.genfromtxt('/home/mzhu/madesi/mzhu_code/MOGP_windmill_rmse_ab.csv',delimiter=',')
GP_rmse = np.genfromtxt('/home/mzhu/madesi/mzhu_code/GP_windmill_rmse.csv', delimiter=',')
GP_rmse_ab = np.genfromtxt('/home/mzhu/madesi/mzhu_code/GP_windmill_rmse_ab.csv',delimiter=',')



gpgpn_nlpd = np.genfromtxt('/home/mzhu/madesi/mzhu_code/SPGPN_windmill_nlpd.csv', delimiter=',')
gpspn_nlpd_ab = np.genfromtxt('/home/mzhu/madesi/mzhu_code/SPGPN_windmill_nlpd_ab.csv',delimiter=',')
lr_nlpd = np.genfromtxt('/home/mzhu/madesi/mzhu_code/lr_windmill_nlpd.csv', delimiter=',')
lr_nlpd_ab = np.genfromtxt('/home/mzhu/madesi/mzhu_code/lr_windmill_nlpd_ab.csv',delimiter=',')
MOGP_nlpd = np.genfromtxt('/home/mzhu/madesi/mzhu_code/MOGP_windmill_nlpd.csv', delimiter=',')
MOGP_nlpd_ab = np.genfromtxt('/home/mzhu/madesi/mzhu_code/MOGP_windmill_nlpd_ab.csv',delimiter=',')
GP_nlpd = np.genfromtxt('/home/mzhu/madesi/mzhu_code/GP_windmill_nlpd.csv', delimiter=',')
GP_nlpd_ab = np.genfromtxt('/home/mzhu/madesi/mzhu_code/GP_windmill_nlpd_ab.csv',delimiter=',')
# random_dataset
# rmse_random_model3 = np.genfromtxt('Users/zmypps/Desktop/GP-SPN/GPSPN CODE/mzhu_code/mzhu_code/all_rmse_random.csv',delimiter=',')
# rmse_random_model2 = np.genfromtxt('Users/zmypps/Desktop/GP-SPN/GPSPN CODE/mzhu_code/mzhu_code/all_rmse_random_model2.csv',delimiter=',')
y_test1 = np.zeros(1000)
y_test2 = np.ones(1000)

y_test = np.concatenate((y_test1,y_test2))
rmse_gpspn = np.concatenate((gpgpn_rmse,gpspn_rmse_ab))
rmse_lr = np.concatenate((lr_rmse,lr_rmse_ab))
rmse_mogp = np.concatenate((MOGP_rmse,MOGP_rmse_ab))
rmse_GP = np.concatenate((GP_rmse,GP_rmse_ab))
nlpd_gpspn = np.concatenate((gpgpn_nlpd,gpspn_nlpd_ab))
nlpd_lr = np.concatenate((lr_nlpd,lr_nlpd_ab))
nlpd_mogp = np.concatenate((MOGP_nlpd,MOGP_nlpd_ab))
nlpd_GP = np.concatenate((GP_nlpd,GP_nlpd_ab))

y_test_ = np.concatenate((y_test2,y_test1))
fpr1, tpr1, thresholds1  =  roc_curve(y_test, rmse_gpspn)
roc_auc1 = auc(fpr1,tpr1)
fpr2, tpr2, thresholds2  =  roc_curve(y_test, rmse_lr)
roc_auc2 = auc(fpr2,tpr2)
fpr3, tpr3, thresholds3  =  roc_curve(y_test, rmse_mogp)
roc_auc3 = auc(fpr3,tpr3)
fpr4, tpr4, thresholds4  =  roc_curve(y_test, rmse_GP)
roc_auc4 = auc(fpr4,tpr4)
plt.figure(figsize=(6,6))
plt.title('RMSE ROC')
plt.plot(fpr1, tpr1, 'r-', label = 'MOMoGP Val AUC = %0.3f' % roc_auc1)
plt.plot(fpr2, tpr2, 'b-', label = 'LR Val AUC = %0.3f' % roc_auc2)
plt.plot(fpr3, tpr3, 'g-', label = 'MOGP Val AUC = %0.3f' % roc_auc3)
plt.plot(fpr4, tpr4, 'y-', label = 'GP Val AUC = %0.3f' % roc_auc4)
# plt.plot(fpr1, tpr1, 'b-', label = 'original SPNGP Val AUC = %0.3f' % roc_auc1)
# plt.plot(fpr2, tpr2, 'g-', label = 'improved GPSPN Val AUC = %0.3f' % roc_auc2)
# plt.plot(fpr1, tpr1,'ro-',fpr2, tpr2,'r^-',fpr3, tpr3)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
# plt.savefig('roc.pdf')