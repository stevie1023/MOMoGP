# prepare data
import gc

import dill
import smp as smp
from gpytorch import ExactMarginalLogLikelihood
from torch.optim import *
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior
from prod_learnspngp import query, build_bins
from prod_gp import structure, ExactGPModel

img_array = np.array(Image.open('baboon1.png'))
d_input  = 2
# load original image, shape (n, n, 3)
H,W,_ = img_array.shape
scale = 2
img_train = img_array[::scale, ::scale,:] # downsample the original image to shape(n/2, n/2, 3)
x_train=[]
y_train = []
for i in range(int(H/scale)):
    for j in range(int(W/scale)):
        x_train.append([i,j])
        y_train.append(img_train[i,j])

x = np.asarray(x_train)/H
# y = np.asarray(y_train-mu1)/std2# train_y
y = np.asarray(y_train)
min= np.min(y,axis=0).reshape((1,3))
max= np.max(y,axis=0).reshape((1,3))
y = (y-min)/(max-min)
y_d = y.shape[1]

y_test =img_array.reshape((H,W,3))
y_test = y_test.reshape((H*W,1,3))
y_test = (y_test-min)/(max-min)
x_test=[]
for i in range(H):
    for j in range(W):
        x_test.append([i*0.5,j*0.5])

x_test = np.asarray(x_test)/H
# train SPGPN
opts = {
        'min_samples': 0,
        'X': x,
        'Y': y,
        'qd': 1,
        'max_depth': 100,
        'max_samples': 300,
        'log': True,
        'jump': True,
        'reduce_branching': True
    }
root_region, gps_ = build_bins(**opts)
1
root, gps = structure(root_region,scope = [i for i in range(y.shape[1])], gp_types=['matern1.5_ard'])
# lr = 0.1
# steps = 150
from learnspngp import  query, build_bins
from spngp import structure,ExactGPModel
import torch
lr = 0.1
steps = 75

likelihood_scope = [GaussianLikelihood().train() for _ in range(y_d)]
tensor_x = torch.from_numpy(np.zeros((100,d_input))).float().to('cuda')
tensor_y = torch.from_numpy(np.zeros((100,d_input))).float().to('cuda')
model_scope = [ExactGPModel(x = tensor_x,y = tensor_y,likelihood = likelihood_scope[i], type='matern1.5_ard') for i in range(y.shape[1])]
# opt = [model_scope[p].parameters() for p in range(y_d)]
# optimizer_scope = [Adam([{'params':opt[p]}], lr=lr) for p in range(y_d)]
l0=[]
for m in range(y_d):
    l0.extend(list(model_scope[m].parameters()))
optimizer_scope = Adam([{'params':l0}], lr=lr)
# optimizer_scope=optim.SGD(l0, lr=0.1, momentum=0.9)
# optimizer_scope = torch.optim.LBFGS(l0)
model_scope = [i.to('cuda') for i in model_scope]

nlpd = []
loss_train = []
for i in range(steps): #这是优化的大循环，优化共#steps步
    # tree_loss = [0] * y.shape[1]
    # tree_scope = [0] * y.shape[1]
    optimizer_scope.zero_grad()

    for j, gp in enumerate(gps):
        if i == 0:
            idx = query(x, gp.mins, gp.maxs)
            gp.x = x[idx]
            y_scope = y[:,gp.scope]
            gp.y = y_scope[idx]
            gp.n = len(gp.x)
        cuda_ = True
        temp_device = torch.device("cuda" if cuda_ else "cpu")
        if cuda_:
            torch.cuda.empty_cache()
        x_temp = torch.from_numpy(gp.x).float().to(temp_device)
        y_temp = torch.from_numpy(gp.y.ravel()).float().to(temp_device)
        model_scope[gp.scope].set_train_data(inputs=x_temp, targets=y_temp, strict=False)
        model_scope[gp.scope].train()
        gp.likelihood = likelihood_scope[gp.scope]
        gp.model = model_scope[gp.scope]
        mll = ExactMarginalLogLikelihood(likelihood_scope[gp.scope], model_scope[gp.scope])

        output = model_scope[gp.scope](x_temp)  # Output from model
        if i == steps-1:
            gp.mll = mll(output, y_temp).item()
        gp.mll_grad = -mll(output, y_temp)
        # loss = -mll(output, y_temp)
        x_temp.detach()
        y_temp.detach()
        del x_temp
        del y_temp
        del gp.model,gp.likelihood
        x_temp = y_temp = None
        torch.cuda.empty_cache()
        gc.collect()
        # loss_scope[gp.scope]+=gp.mll_grad

    tree_loss_all = root.update_mll()
    loss_train.append(tree_loss_all.item())
    # for l in range(y_d):
    #     loss_scope[l].backward()
    #     optimizer_scope[l].step()
    print(f"\t Step {i + 1}/{steps}, -mll(loss): {round(tree_loss_all.item(), 3)}")

    tree_loss_all.backward()
    optimizer_scope.step()

    # root.update()

for i, gp in enumerate(gps):
    x_temp = torch.from_numpy(gp.x).float().to('cuda')
    y_temp = torch.from_numpy(gp.y.ravel()).float().to('cuda')
    # model_mean = ExactGPModel(x=x_temp, y=y_temp, likelihood=likelihood_scope[gp.scope], type='matern2.5_ard')
    model_scope[gp.scope].set_train_data(inputs=x_temp, targets=y_temp, strict=False)
    gp.model = model_scope[gp.scope]
    gp.likelihood = likelihood_scope[gp.scope]
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp.model)
    output = model_scope[gp.scope](x_temp)
    gp.mll = mll(output, y_temp).item()
    x_temp.detach()
    y_temp.detach()
    del x_temp
    del y_temp
    x_temp = y_temp = None
    torch.cuda.empty_cache()
root.update()

# # #
# for i, gp in enumerate(gps):
#     idx = query(x, gp.mins, gp.maxs)
#     gp.x = x[idx]
#     y_scope = y[:,gp.scope]
#     gp.y = y_scope[idx]
#     print(f"Training GP {i + 1}/{len(gps)} ({len(idx)})")
#     gp.init(cuda=True,lr = lr,steps=steps)

# filename = 'graph1.dill'
# dill.dump(root, open(filename, 'wb'))
# root.update()
# # interpolation
filename2 = 'graphglobal.dill'
dill.dump(root, open(filename2, 'wb'))

# with open("/home/mzhu/madesi/mzhu_code/graph2.dill", "rb") as dill_file:
#     root = dill.load(dill_file)
mu,_ = root.forward(np.asarray(x_test), smudge=0,y_d = y_d)
# print(mu.shape)

rmse = 0
mae = 0
for k in range(y.shape[1]):
    mu_s1 = mu[:,0, k]
    sqe1 = (mu_s1 - y_test[:,0,k]) ** 2
    rmse1 = np.sqrt(sqe1.sum() / len(y_test))
    mae1 = np.sqrt(sqe1).sum() / len(y_test)
    mae+=mae1
    rmse+=rmse1
print('rmse',rmse)
print('mae',mae)

for k in range(mu.shape[0]):
    mu[k,0,:]=mu[k,0,:]*(max-min)+min

    # mu[k, 0, 0] = mu[k, 0, 0]+
    # # if mu[k,0,0]>255:
    # #     mu[k, 0, 0]=255
    # # if mu[k, 0, 0] <0:
    # #     mu[k, 0, 0] = 0
    # mu[k, 0, 1] = mu[k, 0, 1]*255
    #
    # mu[k, 0, 2] = mu[k, 0, 2]*255

mu = mu.astype(np.uint8).reshape((H,W,3))
im = Image.fromarray(mu)
im.save("rainbowwei1.png")


im = Image.fromarray(img_train)
im.save("pisaa1.png")

mu[::scale,::scale,:] = img_train
im = Image.fromarray(mu)
im.save("rainbowwe12.png")

