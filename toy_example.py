import numpy as np
import pandas as pd
import gc
import dill
import torch
import random

from scipy.io import arff
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)
d_input = 8
# dataarff = arff.loadarff('/home/mzhu/madesi/mzhu_code/scm20d.arff')
# data = pd.DataFrame(dataarff[0])
data = pd.read_csv('/home/mzhu/madesi/mzhu_code/windmill14.csv')
# data = pd.read_csv('/home/mzhu/madesi/mzhu_code/windmill14_ab.csv')
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
std1, mu1= np.std(x_,axis=0), np.mean(y_,axis=0)
std2, mu2= np.std(y_,axis=0), np.mean(x_,axis=0)
x = (x_-mu2)/ std1  # normalized train_x
x1 = (x1_-mu2)/std1 # test_x
y = (y_-mu1)/std2# train_y
y1 = (y1_-mu1)/std2 #test_y
# from sklearn.decomposition import PCA
# pca = PCA(n_components=30)
# x = pca.fit_transform(x)
# x1 = pca.fit_transform(x1)
# # d_input=30
#
# x_version2_train= train.iloc[:1000,[1,2,3,4,5,6,7,18,19,10,11,20,21,14,15,22,23]].values
# y_version2_train = train.iloc[:1000,[8,9,12,13,16,17]].values
# x_version2_test= test.iloc[:250,[1,2,3,4,5,6,7,18,19,10,11,20,21,14,15,22,23]].values
# y_version2_test = test.iloc[:250,[8,9,12,13,16,17]].values
# x_version3_train = train.iloc[:1000,[1,2,3,4,5,8,9,18,19,12,13,20,21,16,17,22,23]].values
# y_version3_train = train.iloc[:1000,[6,7,10,11,14,15]].values
# x_version3_test = test.iloc[:250,[1,2,3,4,5,8,9,18,19,12,13,20,21,16,17,22,23]].values
# y_version3_test = test.iloc[:250,[6,7,10,11,14,15]].values
#
# std_version2x, mu_version2y= np.std(x_version2_train,axis=0), np.mean(y_version2_train,axis=0)
# std_version2y, mu_version2x= np.std(y_version2_train,axis=0), np.mean(x_version2_train,axis=0)
# x_version2_train = (x_version2_train-mu_version2x)/ std_version2x  # normalized train_x
# x_version2_test = (x_version2_test-mu_version2x)/std_version2x # test_x
# y_version2_train = (y_version2_train-mu_version2y)/std_version2y# train_y
# y_version2_test = (y_version2_test-mu_version2y)/std_version2y
#
# std_version3x, mu_version3y= np.std(x_version3_train,axis=0), np.mean(y_version3_train,axis=0)
# std_version3y, mu_version3x= np.std(y_version3_train,axis=0), np.mean(x_version3_train,axis=0)
# x_version3_train = (x_version3_train-mu_version3x)/ std_version3x  # normalized train_x
# x_version3_test = (x_version3_test-mu_version3x)/std_version3x # test_x
# y_version3_train = (y_version3_train-mu_version3y)/std_version3y# train_y
# y_version3_test = (y_version3_test-mu_version3y)/std_version3y
#
# #concatenate three combinations
# x = np.concatenate((x,x_version2_train,x_version3_train),axis=0)
# y = np.concatenate((y,y_version2_train,y_version3_train),axis=0)
# # x1 = np.concatenate((x1,x_version2_test,x_version3_test),axis=0)
# # y1 = np.concatenate((y1,y_version2_test,y_version3_test),axis=0)
# # x1 = x_version2_test
# # y1 = y_version2_test

# x_ = pd.read_csv('/home/mzhu/madesi/datasets/datasets/usflight/x_train.csv')
# x1_ = pd.read_csv('/home/mzhu/madesi/datasets/datasets/usflight/x_test.csv')
# y_ = pd.read_csv('/home/mzhu/madesi/datasets/datasets/usflight/y_train.csv')
# y1_= pd.read_csv('/home/mzhu/madesi/datasets/datasets/usflight/y_test.csv')
#
# print(x_.shape)
# print(x1_.shape)
# mu1,std1 =x_.mean().to_numpy(),x_.std().to_numpy()
# # mu2,std2 = x1.mean(),x1.std()
# mu3,std3 =y_.mean().to_numpy(),y_.std().to_numpy()
# # mu4,std4 = y1.mean(),y1.std()
# x = (x_-mu1)/std1# normalized train_x
# x1 = (x1_-mu1)/std1 # test_x
# y = (y_-mu3)/std3# train_y
# y1 = (y1_-mu3)/std3 #test_y
# x = x.iloc[:,:].values
# x1 = x1.iloc[:,:].values
# y = y.iloc[:,:].values
# y1 = y1.iloc[:,:].values
# y_d = y.shape[1]
from torch.utils.data import TensorDataset, DataLoader
# test_x = torch.from_numpy(x1).float().to("cuda")
# test_y = torch.from_numpy(y1).float().to("cuda")
# # test_x = x1
# test_y = y1
# test_dataset = TensorDataset(test_x, test_y)
# test_loader = DataLoader(test_dataset, batch_size=1048, shuffle=False)
# MU = []
# COV=[]
# for x_batch, y_batch in test_loader:
#
#     with open("/home/mzhu/DSMGP_usflight150.dill", "rb") as dill_file:
#         root = dill.load(dill_file)
#
#     # np.fill_diagonal(cov, 0.01)
#     x_temp = x_batch.cpu().detach().numpy()
#     mu, cov = root.forward(x_temp[:, :], smudge=0,y_d = y_d)
#     MU.append(mu)
#     COV.append(cov)
# # mu2, cov2 = root.forward(x2[:, :], smudge=0,y_d = y_d)
# mu = np.stack(MU,axis=0)
# cov = np.stack(COV,axis=0)
with open("/home/mzhu/madesi/mzhu_code/GPSPNCIT_usflight.dill", "rb") as dill_file:
    root = dill.load(dill_file)

##test with abnormal data
data2 = pd.read_csv('/home/mzhu/madesi/mzhu_code/windmill14_ab.csv')
data2 = pd.DataFrame(data2).dropna()
train2 = data2.sample(frac=0.8, random_state=58)
test2 = data2.drop(train2.index)
x1_, y1_ = test2.iloc[:, :d_input].values, test2.iloc[:, d_input:].values
x1 = (x1_-mu2)/std1 # test_x
y1 = (y1_-mu1)/std2 #test_y
mu, cov = root.forward(x1[:, :], smudge=0,y_d = y_d)
mae = 0
rmse = 0
all_nlpd = []
all_rmse=np.zeros((y_d,1000))

for k in range(y.shape[1]):
    mu_s1 = mu[:, 0, k]
    sqe1 = (mu_s1 - y1[:, k]) ** 2
    rmse1 = np.sqrt(sqe1.sum() / len(y1))
    mae1 = np.sqrt(sqe1).sum() / len(y1)
    mae += mae1
    rmse += rmse1
    all_rmse[k, :] = np.sqrt(sqe1)

nlpd1 = 0
count = 0
for i in range(mu.shape[0]):
    sigma = np.sqrt(np.abs(np.linalg.det(cov[i, :, :])))
    if sigma == 0:
        count += 1
        sigma = 0.01
    # d1 = (test.iloc[i,d_input:].values.reshape((1,1,y_d))-mu[i,:,:]).reshape((1,y_d))
    d1 = (y1[i, :].reshape((1, 1, y_d)) - mu[i, :, :]).reshape((1, y_d))

    a = 1 / (np.power((2 * np.pi), y.shape[1] / 2) * sigma)
    ni = np.linalg.pinv(cov[i, :, :])
    b = a * np.exp(np.dot(np.dot(-1 / 2 * d1, ni), d1.T))
    if b > 0.00000000000001:
        nlpd = -np.log(b)
    else:
        nlpd = 0

    nlpd1 += nlpd
    all_nlpd.append(nlpd)
nlpd2 = nlpd1 / len(y1)

print(rmse/y_d)
print(mae/y_d)
print(nlpd2)
print(count)
# LMM[kkk, :] = logsumexp(outer_LMM, axis=0)
# LMM[kkk, :] = LMM[kkk, :] / len(gps)
all_rmse = np.sum(all_rmse,axis=0)
np.savetxt('SPGPN_windmill_rmse_ab.csv', [all_rmse], delimiter=',')
np.savetxt('SPGPN_windmill_nlpd_ab.csv', all_nlpd, delimiter=',')
# RMSEE.append(rmse / y_d)
# MAEE.append(mae / y_d)
# NLPDD.append(nlpd2)
# scores[kkk, 0] = rmse / y_d
# scores[kkk, 1] = mae / y_d
# scores[kkk, 2] = nlpd2
# # for k in range(y1.shape[1]):
# #     rmse_all=[]
# #     rmse_ab_all=[]
# #     mu_s1 = mu[:, 0, k]
# #     sqe1 = (mu_s1 - y1[:, k]) ** 2
# #     mu_s2 = mu2[:, 0, k]
# #     # sqe2 = (mu_s2 - y2[:, k]) ** 2
# #     rmse1 = np.sqrt(sqe1.sum() / len(y1))
# #     mae1 = np.sqrt(sqe1).sum() / len(y1)
# #     mae += mae1
# #     rmse += rmse1
# #     rmse_all.append(np.sqrt(sqe1))
# #     # rmse_ab_all.append(np.sqrt(sqe2))
# #     # RMSE.append(rmse_all)
# #     # RMSE_AB.append(rmse_ab_all)
# # # RMSE = np.array(RMSE).reshape((6, 1000))
# # # RMSE_AB = np.array(RMSE_AB).reshape((6, 1000))
# # # RMSE = np.mean(RMSE, axis=0)
# # # RMSE_AB = np.mean(RMSE_AB, axis=0)
# # # print(RMSE.shape)
# # nlpd1 = 0
# # count = 0
# # for i in range(mu.shape[0]):
# #     sigma = np.sqrt(np.abs(np.linalg.det(cov[i, :, :])))
# #     if sigma == 0:
# #         count += 1
# #         continue
# #     # d1 = (test.iloc[i,d_input:].values.reshape((1,1,y_d))-mu[i,:,:]).reshape((1,y_d))
# #     d1 = (y1[i, :].reshape((1, 1, y_d)) - mu[i, :, :]).reshape((1, y_d))
# #
# #     a = 1 / (np.power((2 * np.pi), y1.shape[1] / 2) * sigma)
# #     ni = np.linalg.pinv(cov[i, :, :])
# #     b = a * np.exp(np.dot(np.dot(-1 / 2 * d1, ni), d1.T))
# #     if b > 0.00000000000001:
# #         nlpd = -np.log(b)
# #     else:
# #         nlpd = 0
# #
# #     nlpd1 += nlpd
# # nlpd2 = nlpd1 / len(y1)
# # np.savetxt('spgpn.csv', RMSE, delimiter=',')
# # np.savetxt('spgpn_ab.csv', RMSE_AB, delimiter=',')
# # print(f"SPN-GP  RMSE: {rmse / y_d}")
# # print(f"SPN-GP  MAE: {mae / y_d}")
# # print(f"SPN-GP  NLPD: {nlpd2}")
# # print(count)
# #
