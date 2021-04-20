import pandas as pd
import numpy as np
import gc
from scipy.io import arff
import torch
import gpytorch
import random
from gpytorch.mlls import *
from scipy.special import logsumexp


random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)

# #specify the number of input dimensions
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

#normalization
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
# d_input=30

# x_ = pd.read_csv('/home/mzhu/madesi/datasets/datasets/parkinsons/x_train.csv')
# x1_ = pd.read_csv('/home/mzhu/madesi/datasets/datasets/parkinsons/x_test.csv')
# y_ = pd.read_csv('/home/mzhu/madesi/datasets/datasets/parkinsons/y_train.csv')
# y1_= pd.read_csv('/home/mzhu/madesi/datasets/datasets/parkinsons/y_test.csv')
#
# print(x_.shape)
# print(x1_.shape)
# mu1,std1 =x_.mean().to_numpy(),x_.std().to_numpy()
#
# mu3,std3 =y_.mean().to_numpy(),y_.std().to_numpy()
#
# x = (x_-mu1)/std1# normalized train_x
# x1 = (x1_-mu1)/std1 # test_x
# y = (y_-mu3)/std3# train_y
# y1 = (y1_-mu3)/std3 #test_y
# x = x.iloc[:,:].values
# x1 = x1.iloc[:,:].values
# y = y.iloc[:,:].values
# y1 = y1.iloc[:,:].values
# y_d = y.shape[1]  # number of output dimensions
# d_input=8

MAEE=[]
RMSEE=[]
NLPDD=[]


class FancyGPWithPriors(gpytorch.models.ExactGP):
    def __init__(self, x, y, likelihood):

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y.ravel()).float()

        xd = x.shape[1]
        active_dims = torch.tensor(list(range(xd)))
        super(FancyGPWithPriors, self).__init__(x, y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        lengthscale_prior = gpytorch.priors.GammaPrior(2, 3)
        outputscale_prior = gpytorch.priors.GammaPrior(2, 3)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=xd, active_dims=active_dims)

        )

        # Initialize lengthscale and outputscale to mean of priors
        self.covar_module.base_kernel.lengthscale = lengthscale_prior.sample()
        self.covar_module.outputscale = outputscale_prior.sample()


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


lr = 0.1
rerun = 1
epoch = 200
batch_size = 2096
LMM = np.zeros((rerun,epoch))
scores = np.zeros((rerun,3))

for kkk in range(rerun):
    rmse_all = 0
    mae_all = 0
    nlpd_all =[]
    all_nlpd = np.zeros((y_d, 1000))
    all_rmse = np.zeros((y_d, 1000))
    count=0
    for k in range(y_d):
        device = torch.device("cuda")
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        y_loop = y[:, k].reshape(-1, 1)
        model = FancyGPWithPriors(x, y_loop, likelihood).to(
            device)
        model.train()
        likelihood.train()
        x_tmp = torch.from_numpy(x).float().to(device)
        y_tmp = torch.from_numpy(y_loop.ravel()).float().to(device)
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=lr)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        training_iter = epoch
        iner_LMM = np.zeros((y_d,epoch))
        for i in range(training_iter):
            torch.cuda.empty_cache()
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(x_tmp)
            # Calc loss and backprop gradients
            loss = -mll(output, y_tmp)
            loss.backward()
            if (i+1) > 0 and (i+1) % 10 == 0:
                print('Iter %d/%d - Loss: %.3f     noise: %.3f' % (
                i+1, training_iter, loss.item(),
                # model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ))
            optimizer.step()
            iner_LMM[k, i] = loss.detach().item()
    # np.savetxt('loss_one_gp.csv', [np.array(loss_cached)], delimiter=',')

        x_tmp.detach()
        y_tmp.detach()

        del x_tmp
        del y_tmp
        x_tmp = y_tmp = None

        model.eval()
        likelihood.eval()
        x_ = x1[:, :]
        y_ = y1[:, k]
        #test for abnormal windmill data
        # data2 = pd.read_csv('/home/mzhu/madesi/mzhu_code/windmill14_ab.csv')
        # data2 = pd.DataFrame(data2).dropna()
        # train2 = data2.sample(frac=0.8, random_state=58)
        # test2 = data2.drop(train2.index)
        # x1_, y1_ = test2.iloc[:, :d_input].values, test2.iloc[:, d_input:].values
        # x_ = (x1_ - mu2) / std1  # test_x
        # y_ = (y1_ - mu1) / std2  # test_y

        mll = ExactMarginalLogLikelihood(likelihood, model)
        x_test = torch.from_numpy(x_).float().to(
            device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(x_test))
            pm, pv = observed_pred.mean.detach().cpu().numpy(), observed_pred.variance.detach().cpu().numpy()

            del model
            torch.cuda.empty_cache()
            gc.collect()
        x_test.detach()
        del x_test

        torch.cuda.empty_cache()
        gc.collect()

        y11 = y1[:,k]
        sqe = (pm- y11) ** 2
        rmse1 = np.sqrt(sqe.sum() / len(y1))
        mae1 = np.sqrt(sqe).sum() / len(y1)

        rmse_all+= rmse1
        mae_all+= mae1

        all_rmse[k, :] = np.sqrt(sqe)
        mu = pm.reshape((-1,1))
        cov = pv.reshape((-1,1))

        nlpd1=0
        for i in range(mu.shape[0]):
            sigma2 =cov[i,:]
            if sigma2 == 0:
                count += 1
                continue
            d1 = y1[i,k] - mu[i, :]
            a = np.sqrt((2*np.pi)*sigma2)
            b=1/a * np.exp(-0.5 * np.power(d1,2)/sigma2)
            if b > 0.0000000001:
                nlpd = -np.log(b)
            else:
                nlpd = 0
            nlpd1+=nlpd
            all_nlpd[k,i] = nlpd
        nlpd2 = nlpd1 / len(y1)
        nlpd_all.append(nlpd2)
    # mid_nlpd = logsumexp(nlpd_all)
    LMM[kkk, :] = logsumexp(iner_LMM, axis=0)
    RMSEE.append(rmse_all / y_d)
    MAEE.append(mae_all / y_d)
    # NLPDD.append(mid_nlpd)
    # scores[kkk, 0] = rmse_all / y_d
    # scores[kkk, 1] = mae_all / y_d
    # scores[kkk, 2] = mid_nlpd
# np.savetxt('GP_scores_parkinsons_prior.csv', scores, delimiter=',')
# np.savetxt('GP_LMM_parkinsons_prior.csv', LMM, delimiter=',')
print(f"SPN-GP  RMSE: {RMSEE}")
print(f"SPN-GP  MAE1: {MAEE}")
print(f"SPN-GP  NLPD1: {NLPDD}")
print(count)
print(f"SPN-GP  RMSE mean: {np.mean(np.array(RMSEE))} std:{np.std(np.array(RMSEE))}")
print(f"SPN-GP  MAE mean: {np.mean(np.array(MAEE))} std:{np.std(np.array(MAEE))}")
print(f"SPN-GP  NLPD mean: {np.mean(np.array(NLPDD))} std:{np.std(np.array(NLPDD))}")
all_rmse = np.sum(all_rmse,axis=0)
all_nlpd = np.sum(all_nlpd,axis=0)
np.savetxt('GP_windmill_rmse.csv', [all_rmse], delimiter=',')

np.savetxt('GP_windmill_nlpd.csv', all_nlpd, delimiter=',')
print(all_rmse.shape)
print(all_nlpd.shape)