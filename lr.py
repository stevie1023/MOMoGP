import dill
import pandas as pd
import numpy as np
import gc
from scipy.io import arff
import torch
import gpytorch
import random
from sklearn.linear_model import BayesianRidge, LinearRegression

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)

# # load dataset
d_input = 8 # specify number of input dimensions
# dataarff = arff.loadarff('/home/mzhu/madesi/mzhu_code/scm20d.arff')
# data = pd.DataFrame(dataarff[0])
# data_ab = pd.read_csv('/home/mzhu/madesi/mzhu_code/windmill24_ab.csv')
data = pd.read_csv('/home/mzhu/madesi/mzhu_code/windmill14.csv')
# data = pd.read_csv('/home/mzhu/madesi/mzhu_code/WECs_DataSet/Adelaide_Data.csv')
data = pd.DataFrame(data).dropna()
train = data.sample(frac=0.8, random_state=58)
test = data.drop(train.index)
# data_ab = pd.DataFrame(data_ab).dropna()
# df = pd.read_csv('/home/mzhu/madesi/mzhu_code/VAR.csv',header=None)
# df = pd.DataFrame(df).dropna()  # miss = data.isnull().sum()/len(data)
# data2 = df.T
x_, y_ = train.iloc[:, :d_input].values, train.iloc[:, d_input:].values
y_d = y_.shape[1]
x1_, y1_ = test.iloc[:, :d_input].values, test.iloc[:, d_input:].values
print(x_.shape)

##normalization
std1, mu1= np.std(x_,axis=0), np.mean(y_,axis=0)
std2, mu2= np.std(y_,axis=0), np.mean(x_,axis=0)
x = (x_-mu2)/ std1  # normalized train_x
x1 = (x1_-mu2)/std1 # test_x
y = (y_-mu1)/std2# train_y
y1 = (y1_-mu1)/std2 #test_y

## employ PCA on datasets with large input dimensions
# from sklearn.decomposition import PCA
# pca = PCA(n_components=30)
# x = pca.fit_transform(x)
# x1 = pca.fit_transform(x1)

# # linear regression
gpr = LinearRegression().fit(x,y)

## compute the variace of model---from training datt
pm_train = gpr.predict(x[:,:])
var = np.mean((y-pm_train)**2,axis=0)

#test with abnormal data
# data2 = pd.read_csv('/home/mzhu/madesi/mzhu_code/windmill14_ab.csv')
# data2 = pd.DataFrame(data2).dropna()
# train2 = data2.sample(frac=0.8, random_state=58)
# test2 = data2.drop(train2.index)
# x1_, y1_ = test2.iloc[:, :d_input].values, test2.iloc[:, d_input:].values
# x1 = (x1_-mu2)/std1 # test_x
# y1 = (y1_-mu1)/std2 #test_y
mu = gpr.predict(x1[:,:])
all_nlpd = []
all_rmse=np.zeros((y_d,1000))

rmse = 0
mae = 0
for k in range(y.shape[1]):
    mu_s1 = mu[:, k]
    sqe1 = (mu_s1 - y1[:,k]) ** 2
    rmse1 = np.sqrt(sqe1.sum() / len(y1))
    mae1 = np.sqrt(sqe1).sum() / len(y1)
    mae+=mae1
    rmse+=rmse1

    all_rmse[k, :] = np.sqrt(sqe1)

nlpd1 = 0
count=0
for i in range(mu.shape[0]):
    sigma = np.sqrt(np.abs(np.linalg.det(np.diag(var))))
    if sigma == 0:
        count+=1
        continue
    d1 = (y1[i, :] - mu[i, :]).reshape((1, y_d))
    a = 1/(np.power((2*np.pi),y.shape[1]/2)*sigma)
    ni = np.linalg.pinv(np.diag(var))
    b = a * np.exp(np.dot(np.dot(-1 / 2 * d1, ni), d1.T))
    # if b > 0.0000000001:
    nlpd = -np.log(b)
    all_nlpd.append(nlpd[0,0])
    # else:
    #     nlpd = 0

    nlpd1+=nlpd
nlpd2 = nlpd1/len(y1)
print(rmse/y_d)
print(mae/y_d)
print(nlpd2)
all_rmse = np.sum(all_rmse,axis=0)
np.savetxt('lr_windmill_rmse.csv', [all_rmse], delimiter=',')

np.savetxt('lr_windmill_nlpd.csv', all_nlpd, delimiter=',')