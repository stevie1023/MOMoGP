import dill
import pandas as pd
import numpy as np
import gc
from scipy.io import arff
import torch
import gpytorch
import random
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.mlls import *

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)

#specify the number of input dimensions
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
# d_input=x.shape[1]

#for anomaly detection of windmill
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
# x1 = np.concatenate((x1,x_version2_test,x_version3_test),axis=0)
# y1 = np.concatenate((y1,y_version2_test,y_version3_test),axis=0)
# # x1 = x_version2_test
# # y1 = y_version2_test
# y_d = y.shape[1]

MAEE=[]
RMSEE=[]
NLPDD=[]
all_nlpd = []
all_rmse=np.zeros((y_d,1000))
lr = 0.1
rerun = 1
epoch = 100
batch_size = 2096
LMM = np.zeros((rerun,epoch))
scores = np.zeros((rerun,3))

# MOGP
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        xd = train_x.shape[1]
        active_dims = torch.tensor(list(range(xd)))
        yd = train_y.shape[1]
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=yd
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=xd, active_dims=active_dims), num_tasks=yd, rank=1
        )


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
#
# train_x = torch.from_numpy(x).float().to("cuda")
# train_y = torch.from_numpy(y).float().to("cuda")
# train_y = torch.stack([train_y[:,i] for i in range(y_d)], -1)
# train_dataset = TensorDataset(train_x, train_y)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=y_d)
# model = MultitaskGPModel(train_x, train_y, likelihood).to("cuda")
# model.train()
# likelihood.train()
# #
# # Use the adam optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters
#
# # "Loss" for GPs - the marginal log likelihood
# mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)


# for i in range(epoch):
#     optimizer.zero_grad()
#     output = model(train_x)
#     loss = -mll(output, train_y)
#     loss.backward()
#     print('Iter %d/%d - Loss: %.3f' % (i + 1, epoch, loss.item()))
#     optimizer.step()


# filename1 = 'MOGP_windmill14.dill'
# dill.dump(model, open(filename1, 'wb'))
# with open("/home/mzhu/madesi/mzhu_code/MOGP_windmill14.dill", "rb") as dill_file:
#     model = dill.load(dill_file)
# filename2 = 'MOGP_likelihood_windmill14.dill'
# dill.dump(likelihood, open(filename2, 'wb'))
# with open("/home/mzhu/madesi/mzhu_code/MOGP_likelihood_windmill14.dill", "rb") as dill_file:
#     likelihood = dill.load(dill_file)
# model.eval()
# likelihood.eval()

## MOSVGP
num_latents = y_d
num_tasks = y_d
xd = x.shape[1]

active_dims = torch.tensor(list(range(xd)))
#
class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self):
        # Let's use a different set of inducing points for each latent function
        inducing_points = torch.rand(num_latents, 500, d_input)

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=y_d,
            num_latents=num_latents,
            latent_dim=-1
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents])
        )

def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


device = 'cuda'
train_x = torch.from_numpy(x).float().to(device)
train_y = torch.from_numpy(y).float().to(device)
train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
model = MultitaskGPModel().to(device)
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks).to(device)


optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=lr)

# Our loss object. We're using the VariationalELBO, which essentially just computes the ELBO
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

ave_loss = []
for i in range(epoch):
    mid_loss=0
    # Within each iteration, we will go over each minibatch of data
    for x_batch, y_batch in train_loader:
        likelihood(model(x_batch)).rsample()
        model.train()
        likelihood.train()
        optimizer.zero_grad()
        output = model(x_batch)
        loss = -mll(output,y_batch)
        # print('Iter %d/%d - Loss: %.3f' % (i + 1, epoch, loss.item()))
        loss.backward()
        optimizer.step()
        mid_loss+=loss.item()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, epoch, mid_loss))

model.eval()
likelihood.eval()


##test for MOGP and MOSVGP
data2 = pd.read_csv('/home/mzhu/madesi/mzhu_code/windmill14_ab.csv')
data2 = pd.DataFrame(data2).dropna()
train2 = data2.sample(frac=0.8, random_state=58)
test2 = data2.drop(train2.index)
x1_, y1_ = test2.iloc[:, :d_input].values, test2.iloc[:, d_input:].values
x1 = (x1_-mu2)/std1 # test_x
y1 = (y1_-mu1)/std2 #test_y
device = 'cuda'
test_x = torch.from_numpy(x1).float().to(device)
test_y = torch.from_numpy(y1).float().to(device)
# test_y = torch.stack([test_y[:,i] for i in range(y_d)], -1)
test_dataset = TensorDataset(test_x, test_y)
batch_size = 1
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
mu = np.zeros((len(x1),y_d))
cov = np.zeros((len(x1),y_d,y_d))
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    i = 0
    for x_batch, y_batch in test_loader:
        observed_pred = likelihood(model(x_batch))
        mu1, cov1 = observed_pred.mean.detach().cpu().numpy(), observed_pred.covariance_matrix.detach().cpu().numpy()
        mu[i :i+1, :] = mu1
        cov[i:i+1,:,:] = cov1
        lower, upper = observed_pred.confidence_region()
        i+=1

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


nlpd1=0
count=0
for i in range(mu.shape[0]):
    sigma = np.sqrt(np.abs(np.linalg.det(cov[i,:, :])))
    if sigma == 0:
        print("sigma=0")
        continue
    d1 = (y1[i, :].reshape((1, 1, y_d)) - mu[i, :]).reshape((1, y_d))
    a = 1 / (np.power((2 * np.pi), y.shape[1] / 2) * sigma)
    ni = np.linalg.pinv(cov[i, :,:])
    b = a * np.exp(np.dot(np.dot(-1 / 2 * d1, ni), d1.T))
    if b > 0.0000000001:
        nlpd = -np.log(b)
    else:
        count += 1
        nlpd = 0

    nlpd1 += nlpd
    all_nlpd.append(nlpd[0,0])

nlpd2 = nlpd1/len(y1)
print(rmse/y_d)
print(mae/y_d)
print(nlpd2)
print(count)
all_rmse = np.sum(all_rmse,axis=0)
np.savetxt('MOSVGP_windmill_rmse_ab.csv', [all_rmse], delimiter=',')

np.savetxt('MOSVGP_windmill_nlpd_ab.csv', all_nlpd, delimiter=',')