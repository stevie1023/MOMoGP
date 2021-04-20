#!/usr/bin/python
# -*- coding: <encoding name> -*-
import gc
import itertools
import numpy as np
import pandas as pd
from gpytorch.likelihoods import GaussianLikelihood
from prod_learnspngp import query, build_bins
from prod_gp import structure, ExactGPModel
import random
from scipy.special import logsumexp
from torch import optim
from torch.optim import *
from gpytorch.mlls import *
import torch
from scipy.io import arff
from sklearn.decomposition import PCA
import gpytorch
import dill

random.seed(23)
np.random.seed(23)
torch.manual_seed(23)
torch.cuda.manual_seed(23)

#specify the number of input dimensions
# d_input = 8
# # dataarff = arff.loadarff('/home/mzhu/madesi/mzhu_code/scm20d.arff')
# # data = pd.DataFrame(dataarff[0])
# data = pd.read_csv('/home/mzhu/madesi/mzhu_code/windmill14.csv')
# # data = pd.read_csv('/home/mzhu/madesi/mzhu_code/windmill14_ab.csv')
# # data = pd.read_csv('/home/mzhu/madesi/mzhu_code/WECs_DataSet/Adelaide_Data.csv')
# data = pd.DataFrame(data).dropna()
# train = data.sample(frac=0.8, random_state=58)
# test = data.drop(train.index)
# # # data_ab = pd.DataFrame(data_ab).dropna()
# # # df = pd.read_csv('/home/mzhu/madesi/mzhu_code/VAR.csv',header=None)
# # # df = pd.DataFrame(df).dropna()  # miss = data.isnull().sum()/len(data)
# # # data2 = df.T
# x_, y_ = train.iloc[:, :d_input].values, train.iloc[:, d_input:].values
# y_d = y_.shape[1]
# x1_, y1_ = test.iloc[:, :d_input].values, test.iloc[:, d_input:].values
#
# # print(x_.shape)
# #
# # #normalization
# # std1, mu1= np.std(x_,axis=0), np.mean(y_,axis=0)
# # std2, mu2= np.std(y_,axis=0), np.mean(x_,axis=0)
# # x = (x_-mu2)/ std1  # normalized train_x
# # x1 = (x1_-mu2)/std1 # test_x
# # y = (y_-mu1)/std2# train_y
# # y1 = (y1_-mu1)/std2 #test_y
#
# # # ##implement PCA on datasets with large input dimensions
# # pca = PCA(n_components=30)
# # x = pca.fit_transform(x)
# # x1 = pca.fit_transform(x1)

#for anomaly detection of windmill
# x_version2_train= train.iloc[::10,[1,2,3,4,5,6,7,18,19,10,11,20,21,14,15,22,23]].values
# y_version2_train = train.iloc[::10,[8,9,12,13,16,17]].values
# x_version2_test= test.iloc[:250,[1,2,3,4,5,6,7,18,19,10,11,20,21,14,15,22,23]].values
# y_version2_test = test.iloc[:250,[8,9,12,13,16,17]].values
# x_version3_train = train.iloc[::10,[1,2,3,4,5,8,9,18,19,12,13,20,21,16,17,22,23]].values
# y_version3_train = train.iloc[::10,[6,7,10,11,14,15]].values
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
# x1 = x_version3_test
# y1 = y_version3_test


x_ = pd.read_csv('/home/mzhu/madesi/datasets/datasets/parkinsons/x_train.csv')
x1_ = pd.read_csv('/home/mzhu/madesi/datasets/datasets/parkinsons/x_test.csv')
y_ = pd.read_csv('/home/mzhu/madesi/datasets/datasets/parkinsons/y_train.csv')
y1_= pd.read_csv('/home/mzhu/madesi/datasets/datasets/parkinsons/y_test.csv')
noise = np. random. normal(0, .1, x_. shape)
# x_ +=noise
print(x_.shape)
print(x1_.shape)
mu1,std1 =x_.mean().to_numpy(),x_.std().to_numpy()
# mu2,std2 = x1.mean(),x1.std()
mu3,std3 =y_.mean().to_numpy(),y_.std().to_numpy()
# mu4,std4 = y1.mean(),y1.std()
x = (x_-mu1)/std1# normalized train_x
x1 = (x1_-mu1)/std1 # test_x
y = (y_-mu3)/std3# train_y
y1 = (y1_-mu3)/std3 #test_y
x = x.iloc[:,:].values
x1 = x1.iloc[:,:].values
y = y.iloc[:,:].values
y1 = y1.iloc[:,:].values
y_d = y.shape[1]
d_input = x.shape[1]

#

MAEE=[]
RMSEE=[]
NLPDD=[]
lr = 0.1
rerun = 5
epoch =200
all_nlpd = []
# all_rmse=np.zeros((y_d,1000))
LMM = np.zeros((rerun,epoch))
scores = np.zeros((rerun,3))
for kkk in range(rerun):
#built the root structure
    opts = {
        'min_samples': 0,
        'X': x,
        'Y': y,
        'qd': 1,
        'max_depth': 100,
        'max_samples': 550,
        'log': True,
        'jump': True,
        'reduce_branching': True
    }
    root_region, gps_ = build_bins(**opts)

    root, gps = structure(root_region,scope = [i for i in range(y.shape[1])], gp_types=['matern1.5_ard'])

# train every leaf separately
    outer_LMM = np.zeros((len(gps),epoch))
    for i, gp in enumerate(gps):
        idx = query(x, gp.mins, gp.maxs)
        gp.x = x[idx]
        y_scope = y[:,gp.scope]
        gp.y = y_scope[idx]
        print(f"Training GP {i + 1}/{len(gps)} ({len(idx)})")
        outer_LMM[i,:]= gp.init(cuda=True,lr = lr, steps=epoch,iter = i)
    root.update()
    filename = 'GPSPNCIT_parkinsons.dill'
    dill.dump(root, open(filename, 'wb'))
    ## global optimization
    # likelihood_scope = [GaussianLikelihood().train() for _ in range(y_d)]
    # tensor_x = torch.from_numpy(np.zeros((500,d_input))).float().to('cuda')
    # tensor_y = torch.from_numpy(np.zeros((500,d_input))).float().to('cuda')
    # model_scope = [ExactGPModel(x = tensor_x,y = tensor_y,likelihood = likelihood_scope[i], type='matern1.5_ard') for i in range(y.shape[1])]
    # l0=[]
    # for m in range(y_d):
    #     l0.extend(list(model_scope[m].parameters()))
    # optimizer_scope = Adam([{'params':l0}], lr=lr)
    #
    # model_scope = [i.to('cuda') for i in model_scope]
    #
    # nlpd = []
    # loss_train = []
    # for i in range(epoch):
    #     optimizer_scope.zero_grad()
    #     for j, gp in enumerate(gps):
    #         if i == 0:
    #             idx = query(x, gp.mins, gp.maxs)
    #             gp.x = x[idx]
    #             y_scope = y[:,gp.scope]
    #             gp.y = y_scope[idx]
    #             gp.n = len(gp.x)
    #         cuda_ = True
    #         temp_device = torch.device("cuda" if cuda_ else "cpu")
    #         if cuda_:
    #             torch.cuda.empty_cache()
    #         x_temp = torch.from_numpy(gp.x).float().to(temp_device)
    #         y_temp = torch.from_numpy(gp.y).float().to(temp_device)
    #         model_scope[gp.scope].set_train_data(inputs=x_temp, targets=y_temp, strict=False)
    #         model_scope[gp.scope].train()
    #         mll = ExactMarginalLogLikelihood(likelihood_scope[gp.scope], model_scope[gp.scope])
    #         output = model_scope[gp.scope](x_temp)  # Output from model
    #
    #         gp.mll_grad = -mll(output, y_temp)
    #         x_temp.detach()
    #         y_temp.detach()
    #         del x_temp
    #         del y_temp
    #
    #         x_temp = y_temp = None
    #         torch.cuda.empty_cache()
    #         gc.collect()
    #
    #     tree_loss_all = root.update_mll()
    #     loss_per_epoch = 0
    #     for loss_scope in range(len(tree_loss_all)):
    #         tree_loss_all[loss_scope].backward()
    #         loss_per_epoch+=tree_loss_all[loss_scope].item()
    #     loss_train.append(tree_loss_all.item())
    #
    #     print(f"\t Step {i + 1}/{epoch}, -mll(loss): {round(loss_per_epoch, 3)}")
    #
    #     tree_loss_all.backward()
    #     optimizer_scope.step()
    #
    # for i, gp in enumerate(gps):
    #     x_temp = torch.from_numpy(gp.x).float().to('cuda')
    #     y_temp = torch.from_numpy(gp.y).float().to('cuda')
    #     # model_mean = ExactGPModel(x=x_temp, y=y_temp, likelihood=likelihood_scope[gp.scope], type='matern2.5_ard')
    #     model_scope[gp.scope].set_train_data(inputs=x_temp, targets=y_temp, strict=False)
    #     gp.model = model_scope[gp.scope]
    #     gp.likelihood = likelihood_scope[gp.scope]
    #     mll = ExactMarginalLogLikelihood(gp.likelihood, gp.model)
    #     output = model_scope[gp.scope](x_temp)
    #     gp.mll = mll(output, y_temp).item()
    #     x_temp.detach()
    #     y_temp.detach()
    #     del x_temp
    #     del y_temp
    #     x_temp = y_temp = None
    #     torch.cuda.empty_cache()
    # root.update()

    # filename = 'GPSPN_windmill24.dill'
    # dill.dump(root, open(filename, 'wb'))

    # filename = 'var200_0.1_spgpn.dill'
    # dill.dump(root, open(filename, 'wb'))

    # with open("/home/mzhu/madesi/mzhu_code/GPSPN_windmill14.dill", "rb") as dill_file:
    #     root = dill.load(dill_file)
#     data2 = pd.read_csv('/home/mzhu/madesi/mzhu_code/windmill24_ab.csv')
#     data2 = pd.DataFrame(data2).dropna()
#     train2 = data2.sample(frac=0.8, random_state=58)
#     test2 = data2.drop(train2.index)
# #  # x1
#     x1_, y1_ = test2.iloc[:, :d_input].values, test2.iloc[:, d_input:].values
#     x1 = (x1_ - mu2) / std1  # test_x
#     y1 = (y1_ - mu1) / std2  # test_y
# # # # x2
# #     x1= test2.iloc[:250,[1,2,3,4,5,6,7,18,19,10,11,20,21,14,15,22,23]].values
# #     y1 = test2.iloc[:250,[8,9,12,13,16,17]].values
# #     x1 = (x1-mu_version2x)/std_version2x
# #     y1 = (y1-mu_version2y)/std_version2y
# # #x3
#     x_version3_test = test2.iloc[:250,[1,2,3,4,5,8,9,18,19,12,13,20,21,16,17,22,23]].values
#     y_version3_test = test2.iloc[:250,[6,7,10,11,14,15]].values
#     x1 = (x_version3_test-mu_version3x)/std_version3x # test_x
#     y1 = (y_version3_test-mu_version3y)/std_version3y

    mu, cov= root.forward(x1[:,:], smudge=0,y_d = y_d)


    all_rmse_temp=[]
    rmse = 0
    mae = 0
    for k in range(y.shape[1]):
        mu_s1 = mu[:,0, k]
        sqe1 = (mu_s1 - y1[:,k]) ** 2
        rmse1 = np.sqrt(sqe1.sum() / len(y1))
        mae1 = np.sqrt(sqe1).sum() / len(y1)
        mae+=mae1
        rmse+=rmse1

        # all_rmse[k,:] =np.sqrt(sqe1)
    nlpd1=0
    count=0
    for i in range(mu.shape[0]):
        sigma = np.sqrt(np.abs(np.linalg.det(cov[i,:,:])))
        if sigma == 0:
            count+=1
            continue

        d1 = (y1[i, :].reshape((1, 1, y_d)) - mu[i, :, :]).reshape((1, y_d))
        a = 1/(np.power((2*np.pi),y.shape[1]/2)*sigma)
        ni =np.linalg.pinv(cov[i, :, :])
        b = a * np.exp(np.dot(np.dot(-1 / 2 * d1, ni), d1.T))
        if b > 0.0000000001:
            nlpd = -np.log(b)
        else:
            nlpd = 0

        nlpd1+=nlpd
        all_nlpd.append(nlpd)
    #
    nlpd2 = nlpd1/len(y1)
    LMM[kkk,:] = logsumexp(outer_LMM,axis=0)
    LMM[kkk, :] = LMM[kkk,:]/len(gps)

    RMSEE.append(rmse / y_d)
    MAEE.append(mae / y_d)
    NLPDD.append(nlpd2)
    scores[kkk,0] = rmse / y_d
    scores[kkk,1] = mae / y_d
    scores[kkk,2] = nlpd2

# #
# all_rmse = np.sum(all_rmse,axis=0)
# np.savetxt('SPGPNCIT_scm20d_rmse.csv', [all_rmse], delimiter=',')
# np.savetxt('SPGPNCIT_scm20d_nlpd.csv', all_nlpd, delimiter=',')


np.savetxt('SPGPNCIT_scores_parkinsons.csv', scores, delimiter=',')
np.savetxt('SPGPNCIT_LMM_parkinsons.csv', LMM, delimiter=',')
print(f"SPN-GP  RMSE: {RMSEE}")
print(f"SPN-GP  MAE1: {MAEE}")
print(f"SPN-GP  NLPD1: {NLPDD}")
print(count)
print(f"SPN-GP  RMSE mean: {np.mean(np.array(RMSEE))} std:{np.std(np.array(RMSEE))}")
print(f"SPN-GP  MAE mean: {np.mean(np.array(MAEE))} std:{np.std(np.array(MAEE))}")
print(f"SPN-GP  NLPD mean: {np.mean(np.array(NLPDD))} std:{np.std(np.array(NLPDD))}")