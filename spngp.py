import numpy as np
from learnspngp import Mixture, Separator, GPMixture, Color
import gc
import torch
import gpytorch
from gpytorch.kernels import *
from gpytorch.likelihoods import *
from gpytorch.mlls import *
from scipy.special import logsumexp
from torch.optim import *
from torch.utils.data import TensorDataset, DataLoader


# from scipy.optimize import minimize
# from numpy.linalg   import inv, cholesky

class Sum:
    def __init__(self, **kwargs):
        self.children = []
        self.weights = []
        return None

    def forward(self, x_pred, **kwargs):
        if len(x_pred) == 0:
            return np.zeros(1), np.zeros(1)

        # Sometimes Sums only have one child
        # (when using 1GP for instance) therefore
        # we return fast:
        if len(self.children) == 1:
            r_ = self.children[0].forward(x_pred, **kwargs)
            return r_[0].reshape(-1, 1), r_[1].reshape(-1, 1)

        if len(self.children) == 2:
            _wei = np.array(self.weights).reshape(-1, 1)
            mu_x = np.empty((len(x_pred), len(self.weights)))
            co_x = np.empty((len(x_pred), len(self.weights)))

            r1_ = self.children[0].forward(x_pred, **kwargs)
            r2_ = self.children[1].forward(x_pred, **kwargs)
            mu_x[:, 0], co_x[:, 0] = r1_[0].squeeze(-1), r1_[1].squeeze(-1)
            mu_x[:, 1], co_x[:, 1] = r2_[0].squeeze(-1), r2_[1].squeeze(-1)


            return_mu, return_cov = mu_x @ _wei, co_x @ _wei
            # index_max_mll = np.argmax((np.mean(mll_x1 @_wei), np.mean(mll_x2 @_wei)))
            # return_mu, return_cov = mu_x[:,index_max_mll].reshape(-1, 1), co_x[:, index_max_mll].reshape(-1,1)
            return return_mu, return_cov



    def true_variance(self, x_pred, mu, **kwargs):
        if len(self.children) == 1:
            r_ = self.children[0].true_variance(x_pred, mu, **kwargs)
            return r_[0].reshape(-1, 1), r_[1].reshape(-1, 1)

        _wei = np.array(self.weights).reshape(-1, 1)
        mu_x = np.empty((len(mu), len(self.weights)))
        co_x = np.empty((len(mu), len(self.weights)))

        for i, child in enumerate(self.children):
            mu_c, co_c = child.true_variance(x_pred, mu, **kwargs)
            mu_x[:, i], co_x[:, i] = mu_c.squeeze(-1), co_c.squeeze(-1)

        return mu_x @ _wei, ((mu_x - mu) ** 2 + co_x) @ _wei

    def update(self):
        c_mllh = np.array([c.update() for c in self.children])
        # new_weights = np.exp(c_mllh)
        # self.weights = new_weights / np.sum(new_weights)
        # _wei = np.array(self.weights).reshape((-1, 1))
        # self.mll = c_mllh @_wei
        lw = logsumexp(c_mllh)
        logweights = c_mllh - lw

        self.weights = np.exp(logweights)

        return lw

    def update_mll(self):
        if len(self.children)==2:

            a = torch.stack((self.children[0].update_mll(), self.children[1].update_mll()))
            w_log = torch.logsumexp(a, 0) + torch.log(torch.tensor(0.5))
            return w_log

        elif len(self.children)==1:
            w_exp1 = torch.exp(self.children[0].update_mll())
            w_sum = w_exp1 * self.weights[0]
            w_log = torch.log(w_sum)
            return w_log
        else:
            raise Exception(1)

    def __repr__(self, level=0, **kwargs):
        _wei = dict.get(kwargs, 'extra')
        _wei = Color.flt(_wei) if _wei else ""
        _sel = " " * (level) + f"{_wei} ✚ Sum"
        for i, child in enumerate(self.children):
            _wei = self.weights[i]
            _sel += f"\n{child.__repr__(level + 2, extra=_wei)}"

        return f"{_sel}"


class Split:
    def __init__(self, **kwargs):
        self.children = []
        self.split = kwargs['split']
        self.dimension = kwargs['dimension']
        self.depth = kwargs['depth']
        self.splits = kwargs['splits']
        return None

    def forward(self, x_pred, **kwargs):
        smudge = dict.get(kwargs, 'smudge', 0)

        mu_x = np.zeros((len(x_pred), 1))
        co_x = np.zeros((len(x_pred), 1))

        for i, child in enumerate(self.children):

            idx = np.where((x_pred[:, self.dimension] > self.splits[i]) & (x_pred[:, self.dimension] <= self.splits[i + 1]))[0]
            mu_x[idx], co_x[idx] = child.forward(x_pred[idx], **kwargs)

        return mu_x, co_x

    # left, right = self.children[0], self.children[1]
    #
    #     smudge_scales = dict.get(kwargs, 'smudge_scales', {})
    #     smudge_scale = dict.get(smudge_scales, self.depth, 1)
    #
    #     left_idx = np.where(x_pred[:, self.dimension] <= (self.split + smudge * smudge_scale))[0]
    #     right_idx = np.where(x_pred[:, self.dimension] > (self.split - smudge * smudge_scale))[0]
    #
    #     if not smudge:
    #         mu_x[left_idx], co_x[left_idx]= left.forward(x_pred[left_idx],
    #                                                                        **kwargs)
    #         mu_x[right_idx], co_x[right_idx]= right.forward(x_pred[right_idx],
    #                                                                            **kwargs)
    #     else:
    #         inter_idx = np.intersect1d(left_idx, right_idx)
    #
    #         mu_x[left_idx], co_x[left_idx], mll_x[left_idx] = left.forward(x_pred[left_idx],
    #                                                                        **kwargs)
    #         left_mu, left_co, left_mll = mu_x[inter_idx], co_x[inter_idx], mll_x[inter_idx]
    #
    #         mu_x[right_idx], co_x[right_idx], mll_x[right_idx] = right.forward(x_pred[right_idx],
    #                                                                            **kwargs)
    #         right_mu, right_co, right_mll = mu_x[inter_idx], co_x[inter_idx], mll_x[inter_idx]
    #
    #         ss = (np.sqrt(left_co) ** -2 + np.sqrt(right_co) ** -2) ** -1
    #         mu = ss * (left_mu / left_co + right_mu / right_co)
    #         # modified
    #         mll_inter = ss * (left_mll / left_co + right_mll / right_co)
    #
    #         mu_x[inter_idx], co_x[inter_idx], mll_x[inter_idx] = mu, ss, mll_inter
    #

    def update(self):
        return np.sum([c.update() for c in self.children])
    def update_mll(self):
        # a = torch.sum(torch.tensor([c.update_mll() for c in self.children],requires_grad=True))
        sum=0
        for c in self.children:
            sum+=c.update_mll()

        return sum



    def __repr__(self, level=0, **kwargs):
        _wei = dict.get(kwargs, 'extra')
        _wei = Color.flt(_wei) if _wei else ""
        _spl = Color.val(self.split, f=1, color='yellow')
        _dim = Color.val(self.dimension, f=0, color='orange', extra="dim=")
        _dep = Color.val(self.depth, f=0, color='blue', extra="dep=")
        _sel = " " * (level - 1) + f"{_wei} ⓛ Split {_spl} {_dim} {_dep}"
        for child in self.children:
            _sel += f"\n{child.__repr__(level + 10)}"

        return f"{_sel}"


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, **kwargs):
        x, y = kwargs['x'], kwargs['y']

        likelihood = kwargs['likelihood']
        gp_type = kwargs['type']
        xd = x.shape[1]

        active_dims = torch.tensor(list(range(xd)))

        super(ExactGPModel, self).__init__((x), (y), likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        if gp_type == 'mixed':
            m = MaternKernel(nu=1.5, ard_num_dims=xd, active_dims=active_dims)
            l = gpytorch.kernels.PeriodicKernel()
            self.covar_module = ScaleKernel(ScaleKernel(m) + l)
            return
        elif gp_type == 'matern0.5':
            k = MaternKernel(nu=0.5)
        elif gp_type == 'matern1.5':
            k = MaternKernel(nu=1.5)
        elif gp_type == 'matern2.5':
            k = MaternKernel(nu=2.5)
        elif gp_type == 'rbf':
            k = RBFKernel()
        elif gp_type == 'matern0.5_ard':
            k = MaternKernel(nu=0.5, ard_num_dims=xd, active_dims=active_dims)
        elif gp_type == 'matern1.5_ard':
            k = MaternKernel(nu=1.5, ard_num_dims=xd, active_dims=active_dims)
        elif gp_type == 'matern2.5_ard':
            k = MaternKernel(nu=2.5, ard_num_dims=xd, active_dims=active_dims)
        elif gp_type == 'rbf_ard':
            k = RBFKernel(ard_num_dims=xd)
        elif gp_type == 'linear':
            k = LinearKernel(ard_num_dims=xd)  # ard_num_dims for linear doesn't actually work
        else:
            raise Exception("Unknown GP type")


        self.covar_module = ScaleKernel(k)
        lengthscale_prior = gpytorch.priors.GammaPrior(2, 3)
        outputscale_prior = gpytorch.priors.GammaPrior(2, 3)
        self.covar_module.base_kernel.lengthscale = lengthscale_prior.sample()
        self.covar_module.outputscale = outputscale_prior.sample()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


#
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GP:
    def __init__(self, **kwargs):
        self.type = kwargs['type']
        self.mins = kwargs['mins']
        self.maxs = kwargs['maxs']

        self.x = dict.get(kwargs, 'x', [])
        self.y = dict.get(kwargs, 'y', [])
        self.n = None

    def forward(self, x_pred,  **kwargs):
        ##modified
        mu_gp, co_gp= self.predict(x_pred)


        return mu_gp, co_gp

    def true_variance(self, x_pred, mu, **kwargs):
        mu_gp, co_gp = self.predict(x_pred)
        return mu_gp, co_gp
        # return self.stored_mu, self.stored_cov

    def update(self):
        return self.mll  # np.log(self.n)*0.01 #-self.mll - 1/np.log(self.n)
    def update_mll(self):
        return self.mll_grad

    def predict(self, X_s):
        self.model.eval()
        self.likelihood.eval()
        x = torch.from_numpy(X_s).float().to('cuda')  # .to('cpu') #.cuda(
        # x=X_s


        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(x))
            pm, pv = observed_pred.mean, observed_pred.variance


        x.detach()

        del x
        del self.model

        gc.collect()
        # modified
        return pm.detach().cpu(), pv.detach().cpu()

    def init(self, **kwargs):
        lr = dict.get(kwargs, 'lr', 0.20)
        steps = dict.get(kwargs, 'steps', 100)

        self.n = len(self.x)
        self.cuda = dict.get(kwargs, 'cuda') and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        if self.cuda:
            torch.cuda.empty_cache()

        self.x = torch.from_numpy(self.x).float().to(self.device)

        self.y = torch.from_numpy(self.y.ravel()).float().to(self.device)

        # noises = torch.ones(self.n) * 1
        # self.likelihood = FixedNoiseGaussianLikelihood(noise=noises, learn_additional_noise=True)
        self.likelihood = GaussianLikelihood()
        # noise_constraint=gpytorch.constraints.LessThan(1e-2))
        # (noise_prior=gpytorch.priors.NormalPrior(3, 20))
        self.model = ExactGPModel(x=self.x, y=self.y, likelihood=self.likelihood, type=self.type).to(
            self.device)  # .cuda()
        self.optimizer = Adam([{'params': self.model.parameters()}], lr=lr)
        self.model.train()
        self.likelihood.train()

        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
        # test_dataset = TensorDataset(self.x, self.y)
        # test_loader = DataLoader(test_dataset)
        iner_LMM = np.zeros((1, steps))
        print(f"\tGP {self.type} init completed. Training on {self.device}")
        for i in range(steps):
            self.optimizer.zero_grad()  # Zero gradients from previous iteration
            output = self.model(self.x)  # Output from model
            loss = -mll(output, self.y)  # Calc loss and backprop gradients
            if i > 0 and i % 10 == 0:
                print(f"\t Step {i + 1}/{steps}, -mll(loss): {round(loss.item(), 3)}")

            loss.backward()
            self.optimizer.step()
            iner_LMM[:, i] = loss.detach().item()

        # LOG LIKELIHOOD NOW POSITIVE
        self.mll = -loss.detach().item()

        print(f"\tCompleted. +mll: {round(self.mll, 3)}")

        self.x.detach()
        self.y.detach()

        # del self.likelihood
        # self.likelihood = None
        del self.x
        del self.y
        self.x = self.y = None

        # self.model = self.model.to('cpu')
        torch.cuda.empty_cache()
        gc.collect()
        return iner_LMM


def structure(root_region, **kwargs):
    root = Sum()
    to_process, gps = [(root_region, root)], dict()
    gp_types = dict.get(kwargs, 'gp_types', ['matern1.5_ard'])

    while len(to_process):
        gro, sto = to_process.pop()  # gro = graph region object
        # sto = structure object

        if type(gro) is Mixture:
            for child in gro.children:
                if type(child) == Separator:
                    _child = Split(split=child.split, depth=child.depth, dimension=child.dimension,splits=child.splits)
                elif type(child) == Mixture:
                    _child = Sum()
                else:
                    _child = Sum()
                sto.children.append(_child)
            _cn = len(sto.children)
            sto.weights = np.ones(_cn) / _cn
            to_process.extend(zip(gro.children, sto.children))
        elif type(gro) is Separator:  # sto is Split
            for child in gro.children:
                sto.children.append(Sum())
            to_process.extend(zip(gro.children, sto.children))
        elif type(gro) is GPMixture:  # sto is Sum
            gp_type = gp_types[0]
            key = (*gro.mins, *gro.maxs, gp_type)
            if not dict.get(gps, key):
                gps[key] = GP(type=gp_type, mins=gro.mins, maxs=gro.maxs)
            else:
                pass
            sto.children.append(gps[key])
            _cn = len(sto.children)
            sto.weights = np.ones(_cn) / _cn

    return root, list(gps.values())