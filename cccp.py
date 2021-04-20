import dill
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.stats import norm
import random
from learnspngp import  query, build_bins
from spngp import structure,ExactGPModel
import sys
from scipy.io import arff
from gpytorch.likelihoods import *
from torch.optim import *
from gpytorch.mlls import *
import torch
from sklearn.decomposition import PCA
import gc
# d_input= 8

random.seed(23)
np.random.seed(23)
torch.manual_seed(23)
torch.cuda.manual_seed(23)

d_input = 32
# dataarff = arff.loadarff('/home/mzhu/madesi/mzhu_code/scm20d.arff')
# data = pd.DataFrame(dataarff[0])z
# data = pd.read_csv('/home/mzhu/madesi/mzhu_code/windmill14.csv')
# data = pd.read_csv('/home/mzhu/madesi/mzhu_code/windmill14_ab.csv')
data = pd.read_csv('/home/mzhu/madesi/mzhu_code/WECs_DataSet/Adelaide_Data.csv')
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
# pca = PCA(n_components=30)
# x = pca.fit_transform(x)
# x1 = pca.fit_transform(x1)

# x_ = pd.read_csv('/home/mzhu/madesi/datasets/datasets/parkinsons/x_train.csv')
# x1_ = pd.read_csv('/home/mzhu/madesi/datasets/datasets/parkinsons/x_test.csv')
# y_ = pd.read_csv('/home/mzhu/madesi/datasets/datasets/parkinsons/y_train.csv')
# y1_= pd.read_csv('/home/mzhu/madesi/datasets/datasets/parkinsons/y_test.csv')
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

MAEE=[]
RMSEE=[]
NLPDD=[]
rerun = 5
epoch = 150
lr = 0.1
LMM = np.zeros((rerun,epoch))
scores = np.zeros((rerun,3))

for kkk in range(rerun):

    opts = {
        'min_samples':          0,
        'X':                    x,
        'qd':                   9,
        'max_depth':            100,
        'max_samples':       5100,
        'log':               True,
        'jump':              True,
        'reduce_branching':  True
    }
    root_region, gps_ = build_bins(**opts)

    root, gps = structure(root_region, gp_types=['matern1.5_ard'])  #modified

    rmse =0
    mae =0
    nlpd_all =[]
    count=0
    mid_LMM=np.zeros((y_d,epoch))
    for k in range(y_d):
        y_loop = y[:,k].reshape(-1,1)
        # for i, gp in enumerate(gps):
        #     idx = query(x, gp.mins, gp.maxs)
        #     gp.x, gp.y = x[idx], y_loop[idx]
        #     print(f"Training GP set1 {i+1}/{len(gps)} ({len(idx)})") #modified
        #     gp.init(cuda=True)
        # lr = 0.1
        # steps = 200
        # likelihood_scope = GaussianLikelihood().train()
        # tensor_x = torch.from_numpy(np.zeros((100,d_input))).float().to('cuda')
        # tensor_y = torch.from_numpy(np.zeros((100, d_input))).float().to('cuda')
        # model_scope = ExactGPModel(x=tensor_x, y=tensor_y, likelihood=likelihood_scope, type='matern1.5_ard').to('cuda')
        #
        # l0 = list(model_scope.parameters())
        #
        # # for param in l0:
        # #     print(f'value = {param.item()}')
        #
        # optimizer_scope = Adam([{'params': l0}], lr=lr)
        #
        # for i in range(steps):  # 这是优化的大循环，优化共#steps步
        #     optimizer_scope.zero_grad()
        #     for j, gp in enumerate(gps):
        #         if i == 0:
        #             idx = query(x, gp.mins, gp.maxs)
        #             gp.x, gp.y = x[idx], y_loop[idx]
        #         cuda_ = True
        #         temp_device = torch.device("cuda" if cuda_ else "cpu")
        #         if cuda_:
        #             torch.cuda.empty_cache()
        #         x_temp = torch.from_numpy(gp.x).float().to(temp_device)
        #         y_temp = torch.from_numpy(gp.y.ravel()).float().to(temp_device)
        #         model_scope.set_train_data(inputs=x_temp, targets=y_temp, strict=False)
        #         model_scope.train()
        #         gp.likelihood = likelihood_scope
        #         gp.model = model_scope
        #         mll = ExactMarginalLogLikelihood(likelihood_scope, model_scope)
        #         output = model_scope(x_temp)  # Output from model
        #         if i == steps - 1:
        #             gp.mll = mll(output, y_temp).item()
        #         gp.mll_grad = -mll(output, y_temp)
        #
        #         # loss = -mll(output, y_temp)
        #         x_temp.detach()
        #         y_temp.detach()
        #         del x_temp
        #         del y_temp
        #         del gp.model
        #         x_temp = y_temp = None
        #         torch.cuda.empty_cache()
        #         gc.collect()
        #     tree_loss_all = root.update_mll()
        #     # root.update()
        #     print('loss', tree_loss_all.item())
        #     tree_loss_all.backward()
        #     optimizer_scope.step()
        #
        # for i, gp in enumerate(gps):
        #     x_temp = torch.from_numpy(gp.x).float().to('cuda')
        #     y_temp = torch.from_numpy(gp.y.ravel()).float().to('cuda')
        #     model_scope.set_train_data(inputs=x_temp, targets=y_temp, strict=False)
        #     gp.model = model_scope
        #     gp.likelihood = likelihood_scope
        #     mll = ExactMarginalLogLikelihood(gp.likelihood, gp.model)
        #     output = model_scope(x_temp)
        #     gp.mll = mll(output, y_temp).item()
        #     x_temp.detach()
        #     y_temp.detach()
        #     del x_temp
        #     del y_temp
        #     x_temp = y_temp = None
        #     torch.cuda.empty_cache()
        # root.update()
        outer_LMM = np.zeros((len(gps), epoch))
        for i, gp in enumerate(gps):
            idx = query(x, gp.mins, gp.maxs)
            gp.x = x[idx]
            # y_scope = y[:, gp.scope]
            gp.y = y_loop[idx]
            print(f"Training GP {i + 1}/{len(gps)} ({len(idx)})")
            # gp.init(cuda=True,lr = 0.1,steps=200)
            outer_LMM[i, :] = gp.init(cuda=True, lr=lr, steps=epoch, iter=i)
        root.update()
        # mu, cov = root.forward(test.iloc[:, :d_input].values,smudge=0)
        # mu_s1 = mu[:,:].ravel()
        # mu_t1 = test.iloc[:, d_input+k]
        # sqe1 = (mu_s1 - mu_t1.values) ** 2
        # rmse1 = np.sqrt(sqe1.sum() / len(test))
        # mae1 = np.sqrt(sqe1).sum() / len(test)
        # rmse+=rmse1
        # mae+=mae1
        #
        # filename = 'DSMGP_scm.dill'
        # dill.dump(root, open(filename, 'wb'))
        mu, cov = root.forward(x1[:, :], smudge=0)
        mu_s1 = mu[:, :].ravel()
        sqe1 = (mu_s1 - y1[:, k]) ** 2
        rmse1 = np.sqrt(sqe1.sum() / len(y1))
        mae1 = np.sqrt(sqe1).sum() / len(y1)
        mae += mae1
        rmse += rmse1
        nlpd1=0
        for i in range(mu.shape[0]):
            # vals = norm.pdf(mu[i,:],test)
            sigma2 =cov[i,:]
            if sigma2 == 0:
                count += 1
                continue
            # d1 = test.iloc[i, d_input+k]-mu[i,:]
            d1 = y1[i, k] - mu[i, :]
            a = np.sqrt((2*np.pi)*sigma2)
            b=1/a * np.exp(-0.5 * np.power(d1,2)/sigma2)
            if b > 0.0000000001:
                nlpd = -np.log(b)
            else:
                nlpd = 0
            # nlpd = -np.log(1/a * np.exp(-0.5 * np.power(d1, 2) / sigma2))
            nlpd1+=nlpd

        # nlpd2 = nlpd1/len(test)
        nlpd2 = nlpd1 / len(y1)
        # mid_LMM[k, :] = logsumexp(outer_LMM, axis=0)
        mid_LMM[k, :] = np.sum(outer_LMM, axis=0)
        mid_LMM[k, :] = mid_LMM[k, :] / len(gps)

        nlpd_all.append(nlpd2)
    mid_nlpd = logsumexp(nlpd_all)
    mid_nlpd2 = np.sum(nlpd_all)
    LMM[kkk, :] = logsumexp(mid_LMM, axis=0)
    RMSEE.append(rmse/y_d)
    MAEE.append(mae/y_d)
    NLPDD.append(mid_nlpd2)
    scores[kkk, 0] = rmse / y_d
    scores[kkk, 1] = mae / y_d
    scores[kkk, 2] = mid_nlpd2


np.savetxt('DSMGP_scores_adelaide.csv', scores, delimiter=',')
np.savetxt('DSMGP_LMM_adelaide_mean.csv', LMM, delimiter=',')
print(f"SPN-GP  RMSE: {RMSEE}")
print(f"SPN-GP  MAE1: {MAEE}")
print(f"SPN-GP  NLPD1: {NLPDD}")
# print(count)
# print(f"SPN-GP  RMSE mean: {np.mean(np.array(RMSEE))} std:{np.std(np.array(RMSEE))}")
# print(f"SPN-GP  MAE mean: {np.mean(np.array(MAEE))} std:{np.std(np.array(MAEE))}")
# print(f"SPN-GP  NLPD mean: {np.mean(np.array(NLPDD))} std:{np.std(np.array(NLPDD))}")