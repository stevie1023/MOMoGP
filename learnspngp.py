#!/usr/bin/python
# -*- coding: <encoding name> -*-
import numpy as np
from collections import Counter
from scipy.stats import beta, iqr


class Color():
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    LIGHTBLUE = '\033[96m'
    FADE = '\033[90m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def flt(flt):
        r = "%.4f" % flt
        return f"{Color.FADE}{r}{Color.ENDC}"

    @staticmethod
    def bold(txt):
        return f"{Color.OKGREEN}{txt}{Color.ENDC}"

    @staticmethod
    def val(flt, **kwargs):
        c = kwargs.get('color', 'yellow')
        e = kwargs.get('extra', '')
        f = kwargs.get('f', 4)

        if flt != float('-inf'):
            r = f"%.{f}f" % flt if flt != None else None
        else:
            r = '-∞'

        colors = {
            'yellow': Color.WARNING,
            'blue': Color.OKBLUE,
            'orange': Color.FAIL,
            'green': Color.OKGREEN,
            'lightblue': Color.LIGHTBLUE
        }
        return f"{colors.get(c)}{e}{r}{Color.ENDC}"


class Mixture:
    def __init__(self, **kwargs):
        self.maxs = kwargs['maxs']
        self.mins = kwargs['mins']
        self.deltas = dict.get(kwargs, 'deltas', [])
        self.spreads = self.maxs - self.mins
        self.dimension = dict.get(kwargs, 'dimension', None)
        self.children = dict.get(kwargs, 'children', [])
        self.depth = dict.get(kwargs, 'depth', 0)
        self.n = kwargs['n']
        self.parent = dict.get(kwargs, 'parent', None)
        self.splits = dict.get(kwargs, 'splits', [])
        self.idx = dict.get(kwargs, 'idx', [])
        # assert np.all(self.spreads > 0)

    def __repr__(self, level=0):
        _dim = Color.val(self.dimension, f=0, color='orange', extra="dim=")
        _dep = Color.val(self.depth, f=0, color='yellow', extra="dep=")
        _nnn = Color.val(self.n, f=0, color='green', extra="n=")
        _rng = [f"{round(self.mins[i], 2)} - {round(self.maxs[i], 2)}" for i, _ in enumerate(self.mins)]
        _rng = ", ".join(_rng)

        if self.mins.shape[0] > 4:
            _rng = "..."

        _sel = " " * (level) + f"✚ Mixture [{_rng}] {_dim} {_dep} {_nnn}"

        if level <= 100:
            for split in self.children:
                _sel += f"\n{split.__repr__(level + 2)}"
        else:
            _sel += " ..."
        return f"{_sel}"


class Separator:
    def __init__(self, **kwargs):
        self.split = kwargs['split']
        self.dimension = kwargs['dimension']
        self.depth = kwargs['depth']
        self.children = kwargs['children']
        self.parent = kwargs['parent']
        self.splits = kwargs['splits']

    def __repr__(self, level=0):
        _sel = " " * (level) + f"ⓛ Separator dim={self.dimension} split={round(self.split, 2)}"

        for child in self.children:
            _sel += f"\n{child.__repr__(level + 2)}"

        return _sel


class GPMixture:
    def __init__(self, **kwargs):
        self.mins = kwargs['mins']
        self.maxs = kwargs['maxs']
        self.idx = dict.get(kwargs, 'idx', [])
        self.parent = kwargs['parent']

    def __repr__(self, level=0):
        _rng = [f"{round(self.mins[i], 2)} - {round(self.maxs[i], 2)}" for i, _ in enumerate(self.mins)]
        _rng = ", ".join(_rng)
        if self.mins.shape[0] > 4:
            _rng = "..."

        return " " * (level) + f"⚄ GPMixture [{_rng}] n={len(self.idx)}"


def _cached_gp(cache, **kwargs):
    min_, max_ = list(kwargs['mins']), list(kwargs['maxs'])
    cached = dict.get(cache, (*min_, *max_))
    if not cached:
        cache[(*min_, *max_)] = GPMixture(**kwargs)

    return cache[(*min_, *max_)]


def query(X, mins, maxs, skipleft=False):
    mask, D = np.full(len(X), True), X.shape[1]
    for d_ in range(D):
        if not skipleft:
            mask = mask & (X[:, d_] >= mins[d_]) & (X[:, d_] <= maxs[d_])
        else:
            mask = mask & (X[:, d_] > mins[d_]) & (X[:, d_] <= maxs[d_])

    return np.nonzero(mask)[0]

def get_splits(X, dd, **kwargs):
    meta = dict.get(kwargs, 'meta', [""] * X.shape[1])
    max_depth = dict.get(kwargs, 'max_depth', 8)
    log = dict.get(kwargs, 'log', False)

    features_mask = np.zeros(X.shape[1])
    splits = np.zeros((X.shape[1], dd - 1))
    quantiles = np.quantile(X, np.arange(0, 1, 1 / dd)[1:], axis=0).T
    for i, var in enumerate(quantiles):
        include = False
        if dd == 2:
            spread = np.sum(X[:, i] < var[0]) - np.sum(X[:, i] >= var[0])

            if np.abs(spread) < X.shape[0] / 12:
                include = True
        elif len(np.unique(np.round(var, 8))) == len(var):
            include = True

        if include:
            features_mask[i] = 1
            splits[i] = np.array(var)

            if np.sum(features_mask) <= max_depth and meta and log:
                print(i, "\t", meta[i], var)
            else:
                pass  # print('.', end = '')

    return splits, features_mask


def build_bins(**kwargs):
    X = kwargs['X']
    max_depth = dict.get(kwargs, 'max_depth', 8)
    min_samples = dict.get(kwargs, 'min_samples', 0)
    max_samples = dict.get(kwargs, 'max_samples', 1001)
    log = dict.get(kwargs, 'log', False)
    jump = dict.get(kwargs, 'jump', False)
    ddd = dict.get(kwargs,'qd',1)


    root_mixture_opts = {
        'mins': np.min(X, 0),  # min & max of every feature
        'maxs': np.max(X, 0),
        'n': len(X),  # here the valid length of the data(for every feature)
        'parent': None,
        'dimension': np.argmax(np.var(X, axis=0)), # the index of the features
        'idx': X
    }

    nsplits = Counter()
    root_node = Mixture(**root_mixture_opts)
    to_process, cache = [root_node], dict()


    while len(to_process):
        node = to_process.pop()

        if type(node) is not Mixture:
            continue

        d = node.dimension
        x_node = node.idx

        mins_node, maxs_node = np.min(x_node, 0), np.max(x_node, 0)
        d_selected = np.argsort(-np.var(x_node, axis=0))

        d2 = d_selected[1]
        d3 = d_selected[2]

        node_splits_all = [1, 2]
        quantiles = np.quantile(x_node, np.linspace(0, 1, num=ddd + 2), axis=0).T

        d = [d, d2]

        m = 0
        for split in node_splits_all:  # again operate in splits in one dimension

            u = np.unique(quantiles[d[m]])
            loop = []
            # if len(u) == 1:
            #     loop.append(x_node)
            for i in range(len(u) - 1):
                new_maxs, new_mins = maxs_node.copy(), mins_node.copy()
                skipleft = True
                if i == 0:
                    skipleft = False

                new_mins[d[m]] = u[i]
                new_maxs[d[m]] = u[i + 1]
                idx_i = query(x_node, new_mins, new_maxs, skipleft=skipleft)
                if len(idx_i)==0:
                    continue
                loop.append(idx_i)
            next_depth = node.depth + 1
            results = []
            for idx in loop:
                x_idx = x_node[idx]
                maxs_loop = np.max(x_idx, axis=0)
                mins_loop = np.min(x_idx, axis=0)
                # y_idx = y[idx]
                next_dimension = np.argsort(-np.var(x_idx, axis=0))[0]

                mixture_opts = {
                    'mins': mins_loop,
                    'maxs': maxs_loop,
                    'depth': next_depth,
                    'dimension': next_dimension,
                    'idx': x_idx,
                    'n': len(idx)  # the number of left/right new splits
                }  # newly genenrated mixture nodes for the next dimension

                if len(idx) >= max_samples:
                    results.append(Mixture(**mixture_opts))
                else:
                    gp = _cached_gp(cache, mins=mins_loop, maxs=maxs_loop, idx=idx, parent=None)
                    # gp1 = _cached_gp(cache, mins=mins, maxs=maxs, idx=idx, parent=None)#modified
                    # gp2 = _cached_gp(cache, mins=mins, maxs=maxs, idx=idx, parent=None)
                    results.append(gp)

            if len(results) != 1:
                to_process.extend(results)  # results are put into the root_reigon
                separator_opts = {
                    'depth': node.depth,
                    'dimension': d[m],
                    'split': split,
                    'children': results,
                    'parent': None,
                    'splits': quantiles[d[m]]
                }
                node.children.append(Separator(**separator_opts))  # create product nodes for every mixture node

            elif len(results) == 1:
                node.children.extend(results)
                to_process.extend(results)
            else:
                raise Exception('1')
            m += 1

    gps = list(cache.values())
    aaa = [len(gp.idx) for gp in gps]
    # aaa = len((list(gps))[0].idx)
    c = (np.mean(aaa) ** 3) * len(aaa)
    r = 1 - (c / (len(X) ** 3))
    #
    # gpss=[]
    # for gp in gps:
    #     gp=list(gp)
    #     gpss.extend(gp)
    print("Full:\t\t", len(X) ** 3, "\nOptimized:\t", int(c), f"\n#GP's:\t\t {len(gps)} ({np.min(aaa)}-{np.max(aaa)})",
          "\nReduction:\t", f"{round(100 * r, 4)}%")
    print(f"nsplits:\t {nsplits}")
    print(f"Lengths:\t {aaa}\nSum:\t\t {sum(aaa)} (N={len(X)})")
    # #
    return root_node, gps