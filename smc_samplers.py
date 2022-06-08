# -*- coding: utf-8 -*-

"""
Classical and waste-free SMC samplers.

Overview
========

This module implements SMC samplers, that is, SMC algorithms that may sample
from a sequence of arbitrary probability distributions (and approximate their
normalising constants).  Applications include sequential and non-sequential
Bayesian inference, rare-event simulation, etc.  For more background on
(standard) SMC samplers, see Chapter 17 (and references therein). For the
waste-free variant, see Dau & Chopin (2020).

The following type of sequences of distributions are implemented: 

* SMC tempering: target distribution at time t as a density of the form
  mu(theta) L(theta)^{gamma_t}, with gamma_t increasing from 0 to 1. 

* IBIS: target distribution at time t is the posterior of parameter theta
  given data Y_{0:t}, for a given model. 

* SMC^2: same as IBIS, but a for state-space model. For each
  theta-particle, a local particle filter is run to approximate the
  likelihood up to time t; see Chapter 18 in the book.

SMC samplers for binary distributions (and variable selection) are implemented
elsewhere, in module `binary_smc`.

Before reading the documentation below, you might want to have a look at the
following notebook tutorial_, which may be a more friendly introduction.

.. _tutorial: notebooks/SMC_samplers_tutorial.ipynb

Target distribution(s)
======================

If you want to use a SMC sampler to perform Bayesian inference, you may specify
your model by sub-classing `StaticModel`, and defining method `logpyt` (the log
density of data Y_t, given previous datapoints and parameter values) as
follows::

    class ToyModel(StaticModel):
        def logpyt(self, theta, t):  # density of Y_t given parameter theta
            return -0.5 * (theta['mu'] - self.data[t])**2 / theta['sigma2']

In this example, theta is a structured array, with fields named after the
different parameters of the model. For the sake of consistency, the prior
should be a `distributions.StructDist` object (see module `distributions` for
more details), whose inputs and outputs are structured arrays with the same
fields::

    from particles import distributions as dists

    prior = dists.StructDist(mu=dists.Normal(scale=10.),
                             sigma2=dists.Gamma())

Then you can instantiate the class as follows::

    data = np.random.randn(20)  # simulated data
    my_toy_model = ToyModel(prior=prior, data=data)

This object may be passed as an argument to the `FeynmanKac` classes that
define SMC samplers, see below.

Under the hood, class `StaticModel` defines methods ``loglik`` and ``logpost``
which computes respectively the log-likelihood and the log posterior density of
the model at a certain time.

What if I don't want to do Bayesian inference
=============================================

This is work in progress, but if you just want to sample from some target
distribution, using SMC tempering, you may define your target as follows::

    class ToyBridge(TemperingBridge):
        def logtarget(self, theta):
            return -0.5 * np.sum(theta**2, axis=1)

and then define::

    base_dist = dists.MvNormal(scale=10., cov=np.eye(10))
    toy_bridge = ToyBridge(base_dist=base_dist)

Note that, this time, we went for standard, bi-dimensional numpy arrays for
argument theta. This is fine because we use a prior object that also uses
standard numpy arrays.

FeynmanKac objects
==================

SMC samplers are represented as `FeynmanKac` classes. For instance, to perform
SMC tempering with respect to the bridge defined in the previous section, you
may do::

    fk_tpr = AdaptiveTempering(model=toy_bridge, len_chain=100)
    alg = SMC(fk=fk_tpr, N=200)
    alg.run()

This piece of code will run a tempering SMC algorithm such that:

* the successive exponents are chosen adaptively, so that the ESS between two
  successive steps is cN, with c=1/2 (use parameter ESSrmin to change the value
  of c).
* the waste-free version is implemented; that is, the actual number of
  particles is 100 * 200, but only 200 particles are resampled at each time,
  and then moved through 100 MCMC steps (parameter len_chain)
  (set parameter wastefree=False to run a standard SMC sampler).
* the default MCMC strategy is random walk Metropolis, with a covariance
  proposal set to a fraction of the empirical covariance of the current
  particle sample. See next section for how to use a different MCMC kernel.

To run IBIS instead, you may do::

    fk_ibis = IBIS(model=toy_model, len_chain=100)
    alg = SMC(fk=fk_ibis, N=200)

Again see the notebook tutorials for more details and examples. 

Under the hood
==============

`ThetaParticles`
----------------

In a SMC sampler, a particle sample is represented as a `ThetaParticles`
object  ``X``, which contains several attributes such as, e.g.:

* ``X.theta``: a structured array of length N, representing the N
  particles (or alternatively a numpy arrray of shape (N,d))

* ``X.lpost``: a numpy float array of length N, which stores the
  log-target density of the N particles.

* ``X.shared``: a dictionary that contains meta-information on the N
  particles; for instance it may be used to record the successive acceptance
  rates of the Metropolis steps.

Details may vary in a given algorithm; the common idea is that attribute
``shared`` is the only one which not behave like an array of length N.
The main point of ``ThetaParticles`` is to implement fancy indexing, which is
convenient for e.g. resampling::

    from particles import resampling as rs

    A = rs.resampling('multinomial', W)  # an array of N ints
    Xp = X[A]  # fancy indexing

MCMC schemes
------------

A MCMC scheme (e.g. random walk Metropolis) is represented as an
`ArrayMCMC` object, which has two methods: 

* ``self.calibrate(W, x)``: calibrate (tune the hyper-parameters of)
  the MCMC kernel to the weighted sample (W, x).

* ``self.step(x)``: apply a single step to the `ThetaParticles` object
  ``x``, in place. 

Furthermore, the different ways one may repeat a given MCMC kernel  is
represented by a `MCMCSequence` object, which you may pass as an argument 
when instantiating the `FeynmanKac` object that represents the algorithm you
want to run::

    move = MCMCSequenceWF(mcmc=ArrayRandomWalk(), len_chain=100)
    fk_tpr = AdaptiveTempering(model=toy_bridge, len_chain=100, move=move)
    # run a waste-free SMC sampler where the particles are moved through 99
    # iterations of a random walk Metropolis kernel

Such objects may either keep all intermediate states (as in waste-free SMC, see
sub-class `MCMCSequenceWF`) or only the states of the last iteration (as in
standard SMC, see sub-class `AdaptiveMCMCSequence`).  

The bottom line is: if you wish to implement a different MCMC scheme to move
the particles, you should sub-class `ArrayMCMC`. If you wish to implement a new
strategy to repeat several MCMC steps, you should sub-cass MCMCSequence (or one
of its sub-classes). 


References
==========

Dau, H.D. and Chopin, N (2020). Waste-free Sequential Monte Carlo,
`arxiv:2011.02328 <https://arxiv.org/abs/2011.02328>`_


"""

from __future__ import absolute_import, division, print_function
import copy as cp
import itertools
import numpy as np
from numpy import random
from scipy import optimize, stats, linalg
from particles import distributions 
import matplotlib.pyplot as plt
import particles
from particles import resampling as rs
from particles.state_space_models import Bootstrap

import os
###################################
# Static models


class StaticModel(object):
    """Base class for static models.

    To define a static model, sub-class `StaticModel`, and define method
    `logpyt`.

    Example
    -------
    ::

        class ToyModel(StaticModel):
            def logpyt(self, theta, t):
                return -0.5 * (theta['mu'] - self.data[t])**2

        my_toy_model = ToyModel(data=x, prior=pi)

    See doc of `__init__` for more details on the arguments
    """

    def __init__(self, data=None, prior=None, autodiff = False):
        """
        Parameters
        ----------
        data: list-like
            data
        prior: `StructDist` object
            prior distribution of the parameters
        """
        self.data = data
        self.prior = prior
        self.autodiff = autodiff
    @property
    def T(self):
        return 0 if self.data is None else len(self.data)

    def logpyt(self, theta, t):
        """log-likelihood of Y_t, given parameter and previous datapoints.

        Parameters
        ----------
        theta: dict-like
            theta['par'] is a ndarray containing the N values for parameter par
        t: int
            time
        """
        raise NotImplementedError('StaticModel: logpyt not implemented')

    def loglik(self, theta, t=None):
        """ log-likelihood at given parameter values.

        Parameters
        ----------
        theta: dict-like
            theta['par'] is a ndarray containing the N values for parameter par
        t: int
            time (if set to None, the full log-likelihood is returned)

        Returns
        -------
        l: float numpy.ndarray
            the N log-likelihood values
        """
        if t is None:
            t = self.T - 1
        l = np.zeros(shape=theta.shape[0])
        for s in range(t + 1):
            l += self.logpyt(theta, s)
        np.nan_to_num(l, copy=False, nan=-np.inf) 
        return l

    def logpost(self, theta, t=None):
        """Posterior log-density at given parameter values.

        Parameters
        ----------
        theta: dict-like
            theta['par'] is a ndarray containing the N values for parameter par
        t: int
            time (if set to None, the full posterior is returned)

        Returns
        -------
        l: float numpy.ndarray
            the N log-likelihood values
        """
        return self.prior.logpdf(theta) + self.loglik(theta, t)

class TemperingBridge(StaticModel):
    def __init__(self, base_dist=None):
        self.prior = base_dist

    def loglik(self, theta):
        return np.nan_to_num(self.logtarget(theta) - self.prior.logpdf(theta), copy=False, nan=-np.inf)

    def logpost(self, theta):
        return self.logtarget(theta)

###############################
# Theta Particles


def all_distinct(l, idx):
    """
    Returns the list [l[i] for i in idx] 
    When needed, objects l[i] are replaced by a copy, to make sure that
    the elements of the list are all distinct.

    Parameters
    ---------
    l: iterable
    idx: iterable that generates ints (e.g. ndarray of ints)

    Returns
    -------
    a list
    """
    out = []
    deja_vu = [False for _ in l]
    for i in idx:
        to_add = cp.deepcopy(l[i]) if deja_vu[i] else l[i]
        out.append(to_add)
        deja_vu[i] = True
    return out


class FancyList:
    """A list that implements fancy indexing, and forces elements to be
    distinct.
    """
    def __init__(self, data):
        self.data = [] if data is None else data

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if isinstance(key, np.ndarray):
            return self.__class__(all_distinct(self.data, key))
        else:
            return self.data[key]

    def __add__(self, other):
        return self.__class__(self.data + other.data)

    @classmethod
    def concatenate(cls, *ls):
        return cls(list(itertools.chain(*[l.data for l in ls])))

    def copy(self):
        return cp.deepcopy(self)

    def copyto(self, src, where=None):
        """
        Same syntax and functionality as numpy.copyto

        """
        for n, _ in enumerate(self.data):
            if where[n]:
                self.data[n] = src.data[n]  # not a copy


def view_2d_array(theta):
    """Returns a view to record array theta which behaves
    like a (N,d) float array.
    """
    v = theta.view(np.float)
    N = theta.shape[0]
    v.shape = (N, - 1)
    # raise an error if v cannot be reshaped without creating a copy
    return v

def gen_concatenate(*xs):
    if isinstance(xs[0], np.ndarray):
        return np.concatenate(xs)
    else:
        return xs[0].concatenate(*xs)


class ThetaParticles(object):
    """Base class for particle systems for SMC samplers.

    This is a rather generic class for packing together information on N
    particles; it may have the following attributes:

    * ``theta``: a structured array (an array with named variables);
      see `distributions` module for more details on structured arrays.
    * a bunch of numpy arrays such that shape[0] = N; for instance an array
      ``lpost`` for storing the log posterior density of the N particles;
    * lists of length N; object n in the list is associated to particle n;
      for instance a list of particle filters in SMC^2; the name of each
      of of these lists must be put in class attribute *Nlists*.
    * a common attribute (shared among all particles).

    The whole point of this class is to mimic the behaviour of a numpy array
    containing N particles. In particular this class implements fancy
    indexing::

        obj[array([3, 5, 10, 10])]
        # returns a new instance that contains particles 3, 5 and 10 (twice)

    """
    def __init__(self, shared=None, **fields):
        self.shared = {} if shared is None else shared
        self.__dict__.update(fields)

    @property
    def N(self):
        return len(next(iter(self.dict_fields.values())))

    @property
    def dict_fields(self):
        return {k: v for k, v in self.__dict__.items() if k != 'shared'}

    def __getitem__(self, key):
        fields = {k: v[key] for k, v in self.dict_fields.items()}
        if isinstance(key, int):
            return fields
        else:
            return self.__class__(shared=self.shared.copy(), **fields)

    def __setitem__(self, key, value):
        for k, v in self.dict_fields.item():
            v[key] = getattr(value, k)

    def copy(self):
        """Returns a copy of the object."""
        fields = {k: v.copy() for k, v in self.dict_fields.items()}
        return self.__class__(shared=self.shared.copy(), **fields)

    @classmethod
    def concatenate(cls, *xs):
        fields = {k: gen_concatenate(*[getattr(x, k) for x in xs])
                  for k in xs[0].dict_fields.keys()}
        return cls(shared=xs[0].shared.copy(), **fields)

    def copyto(self, src, where=None):
        """Emulates function copyto in NumPy.

       Parameters
       ----------

       where: (N,) bool ndarray
            True if particle n in src must be copied.
       src: (N,) `ThetaParticles` object
            source

       for each n such that where[n] is True, copy particle n in src
       into self (at location n)
        """
        for k, v in self.dict_fields.items():
            if isinstance(v, np.ndarray):
                # takes care of arrays with ndims > 1
                wh = np.expand_dims(where, tuple(range(1, v.ndim)))
                np.copyto(v, getattr(src, k), where=wh)
            else:
                v.copyto(getattr(src, k), where=where)

    def copyto_at(self, n, src, m):
        """Copy to at a given location.

        Parameters
        ----------
        n: int
            index where to copy
        src: `ThetaParticles` object
            source
        m: int
            index of the element to be copied

        Note
        ----
        Basically, does self[n] <- src[m]
        """
        for k, v in self.dict_fields.items():
            v[n] = getattr(src, k)[m]

#############################
# Basic importance sampler

class ImportanceSampler(object):
    """Importance sampler.

    Basic implementation of importance sampling, with the same interface
    as SMC samplers.

    Parameters
    ----------
    model: `StaticModel` object
        The static model that defines the target posterior distribution(s)
    proposal: `StructDist` object
        the proposal distribution (if None, proposal is set to the prior)

    """
    def __init__(self, model=None, proposal=None):
        self.proposal = model.prior if proposal is None else proposal
        self.model = model

    def run(self, N=100):
        """

        Parameter
        ---------
        N: int
            number of particles

        Returns (as attributes)
        -------
        wgts: Weights object
            The importance weights (with attributes lw, W, and ESS)
        X: ThetaParticles object
            The N particles (with attributes theta, lpost)
        log_norm_cst: float
            Estimate of the log normalising constant of the target
        """
        th = self.proposal.rvs(size=N)
        self.X = ThetaParticles(theta=th)
        self.X.lpost = self.model.logpost(th)
        lw = self.X.lpost - self.proposal.logpdf(th)
        self.wgts = rs.Weights(lw=lw)
        self.log_norm_cst = self.wgts.log_mean

##################################
# MCMC steps (within SMC samplers)

class ArrayMCMC(object):
    """Base class for a (single) MCMC step applied to an array.

    To implement a particular MCMC scheme, subclass ArrayMCMC and define method
    ``step(self, x, target=None)``, which applies one step to all the particles
    in object ``xx``, for a given target distribution ``target``).
    Additionally, you may also define method ``calibrate(self, W, x)`` which will
    be called before resampling in order to tune the MCMC step on the weighted
    sample (W, x).

    """
    def __init__(self):
        pass

    def calibrate(self, W, x):
        """
        Parameters
        ----------
        W:  (N,) numpy array
            weights
        x:  ThetaParticles object
            particles
        """
        pass

    def step(self, x, target=None):
        """
        Parameters
        ----------
        x:   particles object
            current particle system (will be modified in-place)
        target: callable
            compute fields such as x.lpost (log target density)

        Returns
        -------
        mean acceptance probability

        """
        raise NotImplementedError

class ArrayTHMC(ArrayMCMC):
    def proposal(self, xprop, eps, target):
        arr_prop = view_2d_array(xprop.theta)
        arr = np.copy(arr_prop)
        L = linalg.cholesky(xprop.shared['chol_cov'], lower = True)
        p = stats.norm.rvs(size = arr_prop.shape)@(eps*L.T)
        # plt.plot(arr[:,0], arr[:,1], c = 'r')
        arr_prop[:,:] = arr + p + eps*xprop.shared['grad']
        grad_x = xprop.shared['grad']
        target(xprop)
        # print(grad_x -xprop.shared["grad"])
        # for i in range(100):
        #     plot = [[arr[i,0], arr_prop[i,0]], [arr[i,1], arr_prop[i,1]]]
        #     plt.plot([arr[i,0], arr_prop[i,0]],[arr[i,1],arr_prop[i,1]], c = 'black')
        # plt.scatter(arr[:100,0], arr[:100,1], c = 'b')
        # plt.scatter(arr_prop[:100,0], arr_prop[:100,1], c = 'r')     
        return xprop, grad_x

    def step(self, x, target = None, weights = None, test = False):
        cov = x.shared['chol_cov']
        T = 5
        L = linalg.cholesky(cov, lower=True)
        grad = x.shared['grad']
        eps = x.shared['eps']
        xprop = x.__class__(theta=np.empty_like(x.theta))
        arr = view_2d_array(x.theta)
        xprop.theta = x.theta.copy()
        xprop.shared = x.shared
        for t in range(T):
            xprop, grad_x = self.proposal(xprop, eps, target)
            arr_prop = view_2d_array(xprop.theta)
            proba_q_y = -np.diag(((arr_prop- arr)/(4*eps)-grad_x)@np.linalg.inv(cov)@((arr_prop- arr)/(4*eps)-grad_x).T)
            proba_q_x = -np.diag(((-arr_prop + arr)/(4*eps)-xprop.shared['grad'])@np.linalg.inv(cov)@((-arr_prop + arr)/(4*eps)-xprop.shared['grad']).T)
            pb_accept = np.minimum(0, xprop.lpost - x.lpost + proba_q_x - proba_q_y)
        accept = (np.log(random.rand(x.N)) < pb_accept)
        x.copyto(xprop, where=accept)
        target(x)
        # arr = view_2d_array(x.theta)
        # plt.scatter(arr[:100,0], arr[:100,1], c = 'g')
        # plt.show()
        return np.mean(np.exp(pb_accept))

    def calibrate(self,W,x):
        arr = view_2d_array(x.theta)
        N, d = arr.shape
        m, cov = rs.wmean_and_cov(W, arr)
        # cov = np.linalg.inv(np.diag(np.diag(cov)))
        x.shared['chol_cov'] = np.eye(cov.shape[0])
        optimal_alpha = 0.5
        x.shared['eps'] = optimal_alpha

class ArrayHMC(ArrayMCMC):
    # def proposal(self, xprop, pprop, eps, target):
    #     arr = view_2d_array(xprop.theta)
    #     tampon = arr - xprop.shared['grad']@(eps@eps.T)/2 + pprop
    #     tampon2 = arr
    #     pb_accept_1 = np.array([distributions.MvNormal(loc = arr[i] + eps@xprop.shared['grad'][i], cov = eps@eps.T).logpdf(x = tampon[i]) for i in range(len(arr))])
    #     #pb_accept_1 = -0.5*np.diag(pprop@np.linalg.inv(eps@eps.T)@pprop.T)
    #     arr[:,:] = tampon
    #     target(xprop)
    #     pb_accept_2 = np.array([distributions.MvNormal(loc = arr[i] + eps@xprop.shared['grad'][i], cov = eps@eps.T).logpdf(x = tampon2[i]) for i in range(len(arr))])
    #     #pb_accept_2 = -0.5*np.diag((tampon - tampon2 + xprop.shared['grad'])@np.linalg.inv(eps@eps.T)@(tampon - tampon2 + xprop.shared['grad']).T)
    #     #print(- pb_accept_1 + pb_accept_2)
    #     return xprop, pprop, + pb_accept_1 - pb_accept_2

    # def proposal(self, xprop, pprop, eps, target):
    #     arr = view_2d_array(xprop.theta)
    #     cov = xprop.shared['chol_cov']
    #     pprop = pprop + eps*xprop.shared['grad']/2
    #     arr[:,:] = arr + (eps*np.linalg.inv(cov)@pprop.T).T
    #     target(xprop)
    #     pprop = pprop + eps*xprop.shared['grad']/2
    #     return xprop, pprop
    def proposal(self, xprop, eps, target):
        arr_prop = view_2d_array(xprop.theta)
        arr = np.copy(arr_prop)
        L = linalg.cholesky(xprop.shared['chol_cov'], lower = True)
        p = stats.norm.rvs(size = arr_prop.shape)@(eps*L.T)
        # plt.plot(arr[:,0], arr[:,1], c = 'r')
        arr_prop[:,:] = arr + p + eps*xprop.shared['grad']
        grad_x = xprop.shared['grad']
        target(xprop)
        # print(grad_x -xprop.shared["grad"])
        # for i in range(100):
        #     plot = [[arr[i,0], arr_prop[i,0]], [arr[i,1], arr_prop[i,1]]]
        #     plt.plot([arr[i,0], arr_prop[i,0]],[arr[i,1],arr_prop[i,1]], c = 'black')
        # plt.scatter(arr[:100,0], arr[:100,1], c = 'b')
        # plt.scatter(arr_prop[:100,0], arr_prop[:100,1], c = 'r')     
        return xprop, grad_x

    def step(self, x, target = None, weights = None, test = False):
        cov = x.shared['chol_cov']
        T = 1
        L = linalg.cholesky(cov, lower=True)
        grad = x.shared['grad']
        eps = x.shared['eps']
        xprop = x.__class__(theta=np.empty_like(x.theta))
        arr = view_2d_array(x.theta)
        xprop.theta = x.theta.copy()
        xprop.shared = x.shared
        #p = stats.norm.rvs(size=arr_prop.shape)@L.T
        #p = distributions.MvNormal(cov = cov).rvs(arr_prop.shape[0])
        # pprop = p.copy()
        # for t in range(T):
            # xprop, pprop = self.proposal(xprop, pprop, eps, target) 
        xprop, grad_x = self.proposal(xprop, eps, target)
        arr_prop = view_2d_array(xprop.theta)
        proba_q_y = -np.diag(((arr_prop- arr)/(4*eps)-grad_x)@np.linalg.inv(cov)@((arr_prop- arr)/(4*eps)-grad_x).T)
        proba_q_x = -np.diag(((-arr_prop + arr)/(4*eps)-xprop.shared['grad'])@np.linalg.inv(cov)@((-arr_prop + arr)/(4*eps)-xprop.shared['grad']).T)
        # pb_accept = np.minimum(0, (xprop.lpost - x.lpost + np.diag((xprop.theta - x.theta) - )@np.linalg.inv(cov)@pprop.T)
        #                     - 0.5*np.diag(p@np.linalg.inv(cov)@p.T)))
        #print(-xprop.lpost + x.lpost)
        #pb_accept = np.minimum(1, np.exp(xprop.lpost - x.lpost + log_pb_accept))
        pb_accept = np.minimum(0, xprop.lpost - x.lpost + proba_q_x - proba_q_y)
        accept = (np.log(random.rand(x.N)) < pb_accept)
        x.copyto(xprop, where=accept)
        target(x)
        # arr = view_2d_array(x.theta)
        # plt.scatter(arr[:100,0], arr[:100,1], c = 'g')
        # plt.show()
        return np.mean(np.exp(pb_accept))

    def calibrate(self,W,x):
        arr = view_2d_array(x.theta)
        N, d = arr.shape
        m, cov = rs.wmean_and_cov(W, arr)
        # cov = np.linalg.inv(np.diag(np.diag(cov)))
        x.shared['chol_cov'] = np.eye(cov.shape[0])
        optimal_alpha = 0.5
        x.shared['eps'] = optimal_alpha

    def calibrate2(self, W,x):
        arr = view_2d_array(x.theta)
        N, d = arr.shape
        cov = np.eye(d)
        x_etoile = arr[np.argmax(x.lpost)]
        x_grad_etoile = x.shared["grad"][np.argmax(x.lpost)]
        pi_x_etoile = max(x.lpost)
        def compute_ESS(x, weights, alpha, cov):
            arr = view_2d_array(x.theta)
            x_etoile = arr[np.argmax(x.lpost)]
            x_grad_etoile = x.shared["grad"][np.argmax(x.lpost)]
            pi_x_etoile = max(x.lpost)
            divisor_q = (2*np.pi)**(d/2)*alpha**(d/2)*np.sqrt(np.linalg.det(cov))
            calc_accept_y = np.array([-(y - (x_etoile + alpha/2*x_grad_etoile))@np.linalg.inv(cov)@(y - (x_etoile - alpha/2*x_grad_etoile)).T for y in arr])
            new_weights = weights/divisor_q*np.exp(-x.lpost + calc_accept_y/(2*alpha))
            print("right use")
            return np.sum(new_weights)**2/np.sum(new_weights**2)
        def func(alpha, x, weights, cov, x_etoile, x_grad_etoile, pi_x_etoile, verbose = False):
            arr = view_2d_array(x.theta)
            if verbose:
                print("alpha :", alpha)
            if alpha >0:
                
                # calc_accept_y = np.array([-(y - (x_etoile + alpha/2*x_grad_etoile))@np.linalg.inv(cov)@(y - (x_etoile - alpha/2*x_grad_etoile)).T for y in arr])
                #calc_accept_y = np.diag(((-x_etoile+x.theta-(alpha/2*cov@x_grad_etoile))@np.linalg.inv(cov))@(-x_etoile + x.theta -(alpha/2*cov@x_grad_etoile)).T)/(2*alpha)
                #calc_accept_x_etoile = np.diag((x_etoile-x.theta -(alpha/2*x.shared['grad']@cov))@np.linalg.inv(cov)@(x_etoile - x.theta - (alpha/2*x.shared['grad']@cov)).T)/(2*alpha)
                # calc_accept_x_etoile = np.array([-(-arr[i] + x_etoile - alpha/2*x.shared['grad'][i])@np.linalg.inv(cov)@(-arr[i] + x_etoile - alpha/2*x.shared['grad'][i]).T for i in range(len(arr))])
                proba_q_y = -np.diag(((arr- x_etoile)/(4*alpha)-x_grad_etoile)@np.linalg.inv(cov)@((arr- x_etoile)/(4*alpha)-x_grad_etoile).T)
                proba_q_x = -np.diag(((-arr + x_etoile)/(4*alpha)-x.shared['grad'])@np.linalg.inv(cov)@((-arr + x_etoile)/(4*alpha)-x.shared['grad']).T)
        
                d = arr.shape[1]
                divisor_q = alpha**(d/2)
                #print(np.exp(- x.lpost + np.minimum(x.lpost + (calc_accept_x_etoile- calc_accept_y)/(2*alpha) - pi_x_etoile, 0) + calc_accept_y/(2*alpha) + pi_x_etoile))
                #sum_other = np.sum(weights/divisor_q*np.exp(calc_accept_y/(2*alpha) - x.lpost + pi_x_etoile))
                #return np.sum(weights*np.exp(calc_accept_y/(2*alpha))*np.linalg.norm(arr - x_etoile)**2)
                try:
                    
                    return np.sum(weights/divisor_q*np.exp(-x.lpost + np.minimum(x.lpost - pi_x_etoile +(proba_q_x - proba_q_y), 0) + proba_q_y + pi_x_etoile)*np.linalg.norm(arr - x_etoile, axis = 1)**2)
                except:
                    import pdb; pdb.set_trace()
            else:
                print("ALPHA négatif!!")
                return 0
        alphas = np.linspace(0.05,1,50)
        tot = []
        # for alpha in alphas:
        #     print(alpha)
        #     tot.append(func(alpha, x, W, cov, x_etoile, x_grad_etoile, pi_x_etoile))
        #     #tot.append(compute_ESS(alpha = alpha, x = x, weights = W, cov = cov))
        # plt.plot(alphas, tot)
        # plt.show()
        # import pdb; pdb.set_trace()
        def func_final(alpha, x = x, weights = W, cov = cov, x_etoile = x_etoile, x_grad_etoile = x_grad_etoile, pi_x_etoile = pi_x_etoile):
            return -func(alpha, x, weights, cov, x_etoile, x_grad_etoile, pi_x_etoile, False)
            #return compute_ESS(alpha = alpha, x = x, weights = weights, cov = cov)
        try:
            optimal_alpha = optimize.minimize_scalar(func_final, bounds = (0.0, 1.), method = 'bounded', options = {'xatol' : 1e-2}).x
            #optimal_alpha = optimize.newton(func, 0.2, args = (x, W, cov, post), maxiter = 300, tol = 0.1)
        except:
            print("Exception Raised")
            optimal_alpha = 0.5
        x.shared['chol_cov'] = np.eye(cov.shape[1])
        print("optimal alpha",optimal_alpha)
        #optimal_alpha = 0.16*50**(1/3)/arr.shape[1]**(1/3)
        x.shared['eps'] = max(0,optimal_alpha)
        
class ArrayPCN(ArrayMCMC):
    def proposal(self, x, xprop):
        cov = x.shared['chol_cov']
        eps = x.shared['eps']
        arr = view_2d_array(x.theta)
        arr_prop = view_2d_array(xprop.theta)   
        add = eps*stats.norm.rvs(size=arr.shape)
        arr_prop[:,:] = np.sqrt(1 - eps**2)*(arr - np.mean(arr)) + add + np.mean(arr)
        return 0

    def step(self,x, target = None):
        xprop = x.__class__(theta=np.empty_like(x.theta))
        delta_lp = self.proposal(x, xprop)
        target(xprop)
        lp_acc = xprop.lpost - x.lpost
        pb_acc = np.exp(np.clip(lp_acc, None, 0.))
        mean_acc = np.mean(pb_acc)
        accept = (random.rand(x.N) < pb_acc)
        x.copyto(xprop, where=accept)
        # arr = view_2d_array(x.theta)
        # plt.scatter(arr[:100,0], arr[:100,1], c = 'g')
        # plt.show()
        if np.isnan(mean_acc):
            import pdb; pdb.set_trace()
        return mean_acc

    def calibrate(self, W, x):
        arr = view_2d_array(x.theta)
        N, d = arr.shape
        m, cov = rs.wmean_and_cov(W, arr)
        cov = np.eye(cov.shape[0])  
        scale = 0.5
        try:
            #x.shared['chol_cov'] = linalg.cholesky(cov, lower=True)
            x.shared['chol_cov'] = np.eye(cov.shape[0])
        except:
            x.shared['chol_cov'] = np.eye(cov.shape[0])
        x.shared['eps'] = scale
    
    def calibrate2(self, W,x):
        arr = view_2d_array(x.theta)
        N, d = arr.shape
        m, cov = rs.wmean_and_cov(W, arr)
        cov = np.eye(cov.shape[0])
        argmaxi = np.argmax(x.lpost)
        pi_x_etoile = x.lpost[argmaxi]
        x_etoile = arr[np.argmax(x.lpost)]
        sqrt_det = np.sqrt(np.linalg.det(cov))
        x.shared['chol_cov'] = np.eye(cov.shape[0])
        max_lp_y = np.diag(((arr - np.sqrt(1 - 0.05**2)*(x_etoile - np.mean(arr))) + np.mean(arr))@np.linalg.inv(0.05*cov)@(arr - np.sqrt(1 - 0.05**2)*(x_etoile - np.mean(arr)) + np.mean(arr)).T)

        def func(eps, x, weights, cov, pi_x_etoile, sqrt_det, x_etoile, max_lp_y):
            arr = view_2d_array(x.theta)
            d = cov.shape[0]
            divisor_q = eps**(d/2)*(2*np.pi)**(d/2)*sqrt_det
            lp_acc_y = np.diag(((arr - np.sqrt(1 - eps**2)*(x_etoile - np.mean(arr)) + np.mean(arr)))@np.linalg.inv(eps*cov)@((arr - np.sqrt(1 - eps**2)*(x_etoile - np.mean(arr)) + np.mean(arr))).T)
            # lp_acc_x = np.diag(((x_etoile - np.sqrt(1 - eps**2)*(arr - np.mean(arr)) + np.mean(arr)))@np.linalg.inv(eps*cov)@((x_etoile - np.sqrt(1 - eps**2)*(arr - np.mean(arr)) + np.mean(arr))).T)
            # lp_acc_y = np.array([distributions.MvNormal(loc = np.sqrt(1- eps**2)*x_etoile, cov = eps*cov).logpdf(x = arr[i]) for i in range(len(arr))])
            # lp_acc_x = np.array([distributions.MvNormal(loc = np.sqrt(1- eps**2)*arr[i], cov = eps*cov).logpdf(x = x_etoile) for i in range(len(arr))])
            return np.sum(weights/divisor_q*np.exp(-x.lpost + pi_x_etoile + np.minimum(x.lpost - pi_x_etoile,0) - lp_acc_y)* np.linalg.norm(x_etoile - arr, axis = 1)**2)
        alphas = np.linspace(0.05,1,50)
        tot = []
        # for alpha in alphas:
        #     tot.append(func(alpha,x,W,cov, pi_x_etoile, sqrt_det, x_etoile, max_lp_y))
        #     #tot.append(compute_ESS(alpha = alpha, x = x, weights = W, cov = cov))
        # plt.plot(alphas, tot)
        # plt.show()
        def func_final(alpha, x = x, weights = W, cov = cov, pi_x_etoile = pi_x_etoile, sqrt_det = sqrt_det, x_etoile = x_etoile, max_lp_y = max_lp_y):
            return -func(alpha, x, weights, cov, pi_x_etoile, sqrt_det, x_etoile, max_lp_y)
        #try:
        optimal_alpha = optimize.minimize_scalar(func_final, bounds = (0.05, 1.), method = 'bounded', options = {'xatol' : 1e-2}).x
        # except:
        #     print("Exception Raised")
        #     optimal_alpha = 0.1
        print("optimal alpha",optimal_alpha)
        x.shared['eps'] = max(0,optimal_alpha)

class ArrayMetropolis(ArrayMCMC):
    """Base class for Metropolis steps (whatever the proposal).
    """
    def proposal(self, x, xprop):
        raise NotImplementedError

    def step(self, x, target=None):
        xprop = x.__class__(theta=np.empty_like(x.theta))
        delta_lp = self.proposal(x, xprop)
        target(xprop)
        lp_acc = xprop.lpost - x.lpost + delta_lp
        pb_acc = np.exp(np.clip(lp_acc, None, 0.))
        mean_acc = np.mean(pb_acc)
        accept = (random.rand(x.N) < pb_acc)
        x.copyto(xprop, where=accept)
        if np.isnan(mean_acc):
            import pdb; pdb.set_trace()
        return mean_acc

class ArrayRandomWalk(ArrayMetropolis):
    """Gaussian random walk Metropolis.
    """
    def calibrate(self, W, x):
        arr = view_2d_array(x.theta)
        N, d = arr.shape
        m, cov = rs.wmean_and_cov(W, arr)
        cov = np.eye(cov.shape[0])  
        scale = 2.38 / np.sqrt(d)
        try:
            x.shared['chol_cov'] = scale * linalg.cholesky(cov, lower=True)
        except:
            x.shared['chol_cov'] = scale * np.eye(cov.shape[0])

    def calibrate2(self, W, x):
        arr = view_2d_array(x.theta)
        N, d = arr.shape
        m, cov = rs.wmean_and_cov(W, arr)
        argmaxi = np.argmax(x.lpost)
        pi_x_etoile = x.lpost[argmaxi]
        x_etoile = arr[np.argmax(x.lpost)]
        sqrt_det = np.sqrt(np.linalg.det(cov))
        dans_exp = -np.diag((arr-x_etoile)@np.linalg.inv(cov)@(arr-x_etoile).T)
        #dans_exp = np.array([-(y - x_etoile)@np.linalg.inv(cov)@(y-x_etoile).T for y in arr])
        #arr = view_2d_array(x.theta)
        alpha_opt = (2.38 / np.sqrt(d))**2
        print(alpha_opt, "test")
        def new_func(alpha, x, weights, cov, dans_exp, alpha_opt):
            arr = view_2d_array(x.theta)
            new_arr = arr + alpha_opt*stats.norm.rvs(size=arr.shape)
            d = arr.shape[1]
            dans_exp = -np.diag((new_arr - arr)@np.linalg.inv(cov)@(arr-x_etoile).T)
            calc_exp = dans_exp/(2*alpha)
            calc_exp_opt = dans_exp/(2*alpha_opt)
            divisor_q_opt = alpha_opt**(d/2)
            divisor_q = alpha**(d/2)
            return np.sum(divisor_q/divisor_q_opt*np.exp(-calc_exp + calc_exp_opt))     
        def compute_ESS_mean(alpha,x, weights, cov, pi_x_etoile, sqrt_det, dans_exp, x_etoile, verbose = False):
            arr = view_2d_array(x.theta)
            d = arr.shape[1]
            calc_exp = dans_exp/(2*alpha)
            divisor_q = alpha**(d/2)*(2*np.pi)**(d/2)*sqrt_det
            new_weights = weights/divisor_q*np.exp(-x.lpost + calc_exp )
            return x.N/(np.sum(new_weights)**2/np.sum(new_weights**2)) - 1
        def compute_ESS(alpha,x, weights, cov, pi_x_etoile, sqrt_det, dans_exp, x_etoile, verbose = False):
            arr = view_2d_array(x.theta)
            d = arr.shape[1]
            calc_exp = dans_exp/(2*alpha)
            divisor_q = alpha**(d/2)*(2*np.pi)**(d/2)*sqrt_det
            new_weights = weights/divisor_q*np.exp(-x.lpost + calc_exp )
            return (np.sum(new_weights)**2/np.sum(new_weights**2))

        def func(alpha, x, weights, cov, pi_x_etoile, sqrt_det, dans_exp, x_etoile, verbose = False):
            arr = view_2d_array(x.theta)
            if verbose:
                print('alpha', alpha)
            calc_exp = dans_exp/(2*alpha)
            d = arr.shape[1]
            divisor_q = alpha**(d/2)*(2*np.pi)**(d/2)*sqrt_det
            #maxi = max(x.lpost)
            #print((weights/post)/divisor_q*exp_q*np.minimum(post/pi_x_etoile, 1)*np.linalg.norm(arr - x_etoile, axis = 1)**2)
            #sum_other = np.sum(weights/divisor_q*np.exp(calc_exp -x.lpost))
            #print(np.sum((weights/post)/divisor_q/*exp_q*np.minimum(post/pi_x_etoile, 1)*np.linalg.norm(arr - x_etoile, axis = 1)**2))
            # if alpha < 2:
            #     print(weights/divisor_q*np.exp(calc_exp)/sum_other, alpha)
            return np.sum(weights/divisor_q*np.exp( - x.lpost+ pi_x_etoile + np.minimum(x.lpost - pi_x_etoile,0) + calc_exp)* np.linalg.norm(x_etoile - arr, axis = 1)**2)
            return np.sum((weights/post)*1/divisor_q*exp_q*np.minimum(post/pi_x_etoile, 1))/sum_other - 0.3
        def func_final(alpha, x = x, weights = W, cov = cov, pi_x_etoile = pi_x_etoile, sqrt_det = sqrt_det, dans_exp = dans_exp, x_etoile = x_etoile):
            return -new_func(alpha, x, weights, cov, pi_x_etoile, sqrt_det, dans_exp, x_etoile, False)

        def new_func_final(alpha, x = x, weights = W, cov = cov, dans_exp = dans_exp, alpha_opt = alpha_opt):
            return -new_func(alpha, x, weights, cov, dans_exp, alpha_opt)

        # def f_final(alpha, x = x, weights = W, cov = cov, post = post):
        #     return -f_compare(alpha, x, weights, cov, post, False)
        optimal_alpha = optimize.minimize_scalar(new_func_final, bounds = (0.0, 3), method = 'bounded').x - 0.1
        #print("somme:",optimize.minimize_scalar(f_final, bounds = (0.0, 5.), method = 'bounded').x)
            #optimal_alpha = optimize.newton(f_prime, 0.3, args = (x, W, cov, post, True), maxiter = 300, tol = 0.05, fprime = f_second)
        # except:
        #     print("Exception Riased")
        #     optimal_alpha = 2.38 / np.sqrt(d)
        if optimal_alpha <= 0 or optimal_alpha > 10:
            print("Wrong alpha: ", optimal_alpha)
            optimal_alpha = (2.38 / np.sqrt(d))**2
        
        if True == True:
            tot = []
            diff = []
            diff2diff = []
            post = np.exp(x.lpost)
            #pre_requis = calc_pre_requis(alpha,x,W, cov,post)
            alphas = np.linspace(0.05,3, 100)
            for alpha in alphas:
                tot.append(new_func(alpha,x,W,cov, dans_exp, (2.38 / np.sqrt(d))**2))
                #tot.append(new_func(alpha, x, W, cov, pi_x_etoile, sqrt_det, dans_exp, x_etoile, False))
                # diff.append(compute_ESS_mean(alpha, x, W, cov, pi_x_etoile, sqrt_det, dans_exp, x_etoile, False))
                # diff2diff.append(compute_ESS(alpha, x, W, cov, pi_x_etoile, sqrt_det, dans_exp, x_etoile, False))
            names = ["func", "ESS"]
            # plt.subplot(3, 1, 1)
            # plt.plot(np.linspace(0.05,5,500), tot, color = 'black')
            # plt.subplot(3, 1, 2)
            # plt.plot(np.linspace(0.05, 5, 500), diff, color = 'red')
            # plt.subplot(3,1,3)
            # plt.plot(np.linspace(0.05,5,500), diff2diff, color = 'blue')
            #plt.vlines([2.38/x.theta.shape[1]], ymax = 0.5, ymin = 0, color = 'r')
            #plt.hlines(0, xmin = 0.05, xmax = 1, color = 'r')
            plt.plot(alphas, tot, color = 'black')
            plt.show()
            # plt.plot(np.linspace(0.05, 5, 500), diff, color = 'red')
            # plt.show()
            last_list = 0
            # for plot in os.listdir("/home/benj/Téléchargements/Stage CREST/plot/"):
            #     if int(plot[:-4]) >last_list:
            #         last_list = int(plot[:-4])
            # last_list = str(last_list +1)
            # plt.savefig("/home/benj/Téléchargements/Stage CREST/plot/" + last_list)
        try:
            x.shared['chol_cov'] = np.sqrt(optimal_alpha)*linalg.cholesky(cov, lower=True)
        except:
            print("exception in cholesky")
            x.shared['chol_cov'] = np.sqrt(optimal_alpha)*np.eye(cov.shape[0])
        print(optimal_alpha)
        
    def proposal(self, x, xprop):
        L = x.shared['chol_cov']
        arr = view_2d_array(x.theta)
        arr_prop = view_2d_array(xprop.theta)   
        arr_prop[:, :] = (arr + stats.norm.rvs(size=arr.shape) @ L.T)
        return 0.

class ArrayIndependentMetropolis(ArrayMetropolis):
    """Independent Metropolis (Gaussian proposal).
    """
    def __init__(self, scale=1.):
        self.scale = scale

    def calibrate(self, W, x):
        m, cov = rs.wmean_and_cov(W, view_2d_array(x.theta))
        x.shared['mean'] = m
        x.shared['chol_cov'] = self.scale * linalg.cholesky(cov, lower=True)

    def proposal(self, x, xprop):
        mu = x.shared['mean']
        L = x.shared['chol_cov']
        arr = view_2d_array(x.theta)
        arr_prop = view_2d_array(xprop.theta)
        z = stats.norm.rvs(size=arr.shape)
        zx = linalg.solve_triangular(L, np.transpose(arr - mu), lower=True)
        delta_lp = 0.5 * (np.sum(z * z, axis=1) - np.sum(zx * zx, axis=0))
        arr_prop[:, :] = mu + z @ L.T
        return delta_lp


class MCMCSequence:
    """Base class for a (fixed length or adaptive) sequence of MCMC steps.
    """
    def __init__(self, mcmc=None, len_chain=10):
        self.mcmc = ArrayRandomWalk() if mcmc is None else mcmc
        self.nsteps = len_chain - 1

    def calibrate(self, W, x):
        self.mcmc.calibrate(W, x)

    def __call__(self, x, target):
        raise NotImplementedError


class MCMCSequenceWF(MCMCSequence):
    """MCMC sequence for a waste-free SMC sampler (keep all intermediate states).
    """
    def __call__(self, x, target):
        xs = [x]
        xprev = x
        ars = []
        for _ in range(self.nsteps):
            x = x.copy()
            ar = self.mcmc.step(x, target=target)
            ars.append(ar)
            xs.append(x)
        xout = x.concatenate(*xs)
        prev_ars = x.shared.get('acc_rates', [])
        xout.shared['acc_rates'] = prev_ars + [ars]  # a list of lists
        return xout


class AdaptiveMCMCSequence(MCMCSequence):
    """MCMC sequence for a standard SMC sampler (keep only final states).
    """
    def __init__(self, mcmc=None, len_chain=10, adaptive=False, delta_dist=0.1):
        super().__init__(mcmc=mcmc, len_chain=len_chain)
        self.adaptive = adaptive
        self.delta_dist = delta_dist

    def __call__(self, x, target):
        xout = x.copy()
        ars = []
        dist = 0.
        for _ in range(self.nsteps):  # if adaptive, nsteps is max nb of steps
            ar = self.mcmc.step(xout, target)
            ars.append(ar)
            if self.adaptive:
                prev_dist = dist
                diff = view_2d_array(xout.theta) - view_2d_array(x.theta)
                dist = np.mean(linalg.norm(diff, axis=1))
                if np.abs(dist - prev_dist) < self.delta_dist * prev_dist:
                    break
        prev_ars = x.shared.get('acc_rates', [])
        xout.shared['acc_rates'] = prev_ars + [ars]  # a list of lists
        return xout


#############################
# FK classes for SMC samplers
class FKSMCsampler(particles.FeynmanKac):
    """Base FeynmanKac class for SMC samplers.

    Parameters
    ----------
    model: `StaticModel` object
        The static model that defines the target posterior distribution(s)
    wastefree: bool (default: True)
        whether to run a waste-free or standard SMC sampler
    len_chain: int (default=10)
        length of MCMC chains (1 + number of MCMC steps)
    move:   MCMCSequence object
        type of move (a sequence of MCMC steps applied after resampling)

    """
    def __init__(self, model=None, wastefree=True, len_chain=10, move=None):
        self.model = model
        self.wastefree = wastefree
        self.len_chain = len_chain
        if move is None:
            if wastefree:
                self.move = MCMCSequenceWF(len_chain=len_chain)
            else:
                self.move = AdaptiveMCMCSequence(len_chain=len_chain)
        else:
            self.move = move

    @property
    def T(self):
        return self.model.T

    def default_moments(self, W, x):
        return rs.wmean_and_var_str_array(W, x.theta)

    def summary_format(self, smc):
        if smc.rs_flag:
            ars = np.array(smc.X.shared['acc_rates'][-1])
            to_add = ', Metropolis acc. rate (over %i steps): %.3f' % (
                ars.size, ars.mean())
        else:
            to_add = ''
        return 't=%i%s, ESS=%.2f' % (smc.t, to_add, smc.wgts.ESS)

    def time_to_resample(self, smc):
        rs_flag = (smc.aux.ESS < smc.X.N * smc.ESSrmin)
        smc.X.shared['rs_flag'] = rs_flag
        if rs_flag:
            self.move.calibrate(smc.W, smc.X)
        return rs_flag

    def M0(self, N):
        N0 = N * self.len_chain if self.wastefree else N
        return self._M0(N0)


class IBIS(FKSMCsampler):
    def logG(self, t, xp, x):
        lpyt = self.model.logpyt(x.theta, t)
        x.lpost += lpyt
        return lpyt

    def current_target(self, t):
        def func(x):
            x.lpost = self.model.logpost(x.theta, t=t)
        return func

    def _M0(self, N):
        x0 = ThetaParticles(theta=self.model.prior.rvs(size=N))
        self.current_target(0)(x0)
        return x0

    def M(self, t, xp):
        if xp.shared['rs_flag']:
            return self.move(xp, self.current_target(t - 1))
            # in IBIS, target at time t is posterior given y_0:t-1
        else:
            return xp


class AdaptiveTempering(FKSMCsampler):
    """Feynman-Kac class for adaptive tempering SMC.

    Parameters
    ----------
    ESSrmin: float
        Sequence of tempering dist's are chosen so that ESS ~ N * ESSrmin at
        each step

    See base class for other parameters.
    """
    def __init__(self, model=None, wastefree=True, len_chain=10, move=None,
                 ESSrmin=0.5):
        super().__init__(model=model, wastefree=wastefree,
                         len_chain=len_chain, move=move)
        self.ESSrmin = ESSrmin

    def time_to_resample(self, smc):
        self.move.calibrate(smc.W, smc.X)
        return True  # We *always* resample in tempering

    def done(self, smc):
        if smc.X is None:
            return False  # We have not started yet
        else:
            return smc.X.shared['exponents'][-1] >= 1.

    def update_path_sampling_est(self, x, delta):
        grid_size = 10
        binwidth = delta / (grid_size - 1)
        new_ps_est = x.shared['path_sampling'][-1]
        for i, e in enumerate(np.linspace(0., delta, grid_size)):
            mult = 0.5 if i==0 or i==grid_size-1 else 1.
            new_ps_est += (mult * binwidth *
                           np.average(x.llik,
                                      weights=rs.exp_and_normalise(e * x.llik)))
            x.shared['path_sampling'].append(new_ps_est)

    def logG_tempering(self, x, delta):
        dl = delta * x.llik
        x.lpost += dl
        self.update_path_sampling_est(x, delta)
        return dl

    def logG(self, t, xp, x):
        ESSmin = self.ESSrmin * x.N
        f = lambda e: rs.essl(e * x.llik) - ESSmin
        epn = x.shared['exponents'][-1]
        if f(1. - epn) > 0:  # we're done (last iteration)
            delta = 1. - epn
            new_epn = 1.
            # set 1. manually so that we can safely test == 1.
        else:
            delta = optimize.brentq(f, 1.e-12, 1. - epn)  # secant search
            # left endpoint is >0, since f(0.) = nan if any likelihood = -inf
            if delta == 0. :
                delta = 0.001
            new_epn = epn + delta
        x.shared['exponents'].append(new_epn)
        return self.logG_tempering(x, delta)

    def current_target(self, epn):
        def func(x):
            if self.model.autograd:
                import jax
                import jax.numpy as jnp
                def tempered_potential(theta):
                    lprior = self.model.prior.logpdf(theta)
                    llik = self.model.loglik(theta)
                    out = lprior + jnp.nan_to_num(epn * llik)
                    return out, (lprior, llik)
                grad_and_val_fun = jax.vmap(jax.value_and_grad(tempered_potential, has_aux=True))
                (x.lpost, (x.lprior, x.llik)), x.shared['grad'] = grad_and_val_fun(x.theta)
            else:
                arr = view_2d_array(x.theta)
                x.lprior = self.model.prior.logpdf(x.theta)
                x.llik = self.model.loglik(x.theta)
                if hasattr(self.model, "gradlogpdf") and hasattr(self.model, "gradloglik"):
                    x.shared['grad'] = (1-epn)*self.model.gradlogpdf(arr) + epn*self.model.gradloglik(arr, self.model.data)
                if epn > 0.:
                    x.lpost = x.lprior + epn * x.llik
                else:  # avoid having 0 x Nan
                    x.lpost = x.lprior.copy()
        return func

    def _M0(self, N):
        x0 = ThetaParticles(theta=self.model.prior.rvs(size=N))
        x0.shared['exponents'] = [0.]
        x0.shared['path_sampling'] = [0.]
        self.current_target(0.)(x0)
        return x0

    def M(self, t, xp):
        epn = xp.shared['exponents'][-1]
        target = self.current_target(epn)
        return self.move(xp, target)

    def summary_format(self, smc):
        msg = FKSMCsampler.summary_format(self, smc)
        return msg + ', tempering exponent=%.3g' % smc.X.shared['exponents'][-1]


#####################################
# SMC^2

def rec_to_dict(arr):
    """ Turns record array *arr* into a dict """

    return dict(zip(arr.dtype.names, arr))


class SMC2(FKSMCsampler):
    """ Feynman-Kac subclass for the SMC^2 algorithm.

    Parameters
    ----------
    ssm_cls: `StateSpaceModel` subclass
        the considered parametric state-space model
    prior: `StructDist` object
        the prior
    data: list-like
        the data
    smc_options: dict
        options to be passed to each SMC algorithm
    fk_cls: Feynman-Kac class (default: Bootstrap)
    init_Nx: int
        initial value for N_x
    ar_to_increase_Nx: float
        Nx is increased (using an exchange step) each time
        the acceptance rate is above this value (if negative, Nx stays
        constant)
    wastefree:  bool
        whether to use the waste-free version (default: True)
    len_chain:  int
        length of MCMC chain (default: 10)
    move:   MCMCSequence object
        MCMC sequence
    """
    def __init__(self, ssm_cls=None, prior=None, data=None, smc_options=None,
                 fk_cls=None, init_Nx=100, ar_to_increase_Nx=-1.,
                 wastefree=True, len_chain=10, move=None):
        super().__init__(self, wastefree=wastefree, len_chain=len_chain,
                         move=move)
        # switch off collection of basic summaries (takes too much memory)
        self.smc_options = {'collect': 'off'}
        if smc_options is not None:
            self.smc_options.update(smc_options)
        self.fk_cls = Bootstrap if fk_cls is None else fk_cls
        if 'model' in self.smc_options or 'data' in self.smc_options:
            raise ValueError(
                'SMC2: options model and data are not allowed in smc_options')
        self.ssm_cls = ssm_cls
        self.prior = prior
        self.data = data
        self.init_Nx = init_Nx
        self.ar_to_increase_Nx = ar_to_increase_Nx

    @property
    def T(self):
        return 0 if self.data is None else len(self.data)

    def logG(self, t, xp, x):
        # exchange step (should occur only immediately after a move step)
        try:
            ar = np.mean(x.shared['acc_rates'][-1])
        except:  # either list does not exist or is of length 0
            ar = 1.
        low_ar = ar < self.ar_to_increase_Nx
        we_increase_Nx = low_ar & x.shared.get('rs_flag', False)
        if we_increase_Nx:
            liw_Nx = self.exchange_step(x, t, 2 * x.pfs[0].N)
        # compute (estimate of) log p(y_t|\theta,y_{0:t-1})
        lpyt = np.empty(shape=x.N)
        for m, pf in enumerate(x.pfs):
            next(pf)
            lpyt[m] = pf.loglt
        x.lpost += lpyt
        if t > 0:
            x.shared['Nxs'].append(x.pfs[0].N)
        if we_increase_Nx:
            return lpyt + liw_Nx
        else:
            return lpyt

    def alg_instance(self, theta, N):
        return particles.SMC(fk=self.fk_cls(ssm=self.ssm_cls(**theta),
                                            data=self.data),
                          N=N, **self.smc_options)

    def current_target(self, t, Nx):
        def func(x):
            x.pfs = FancyList([self.alg_instance(rec_to_dict(theta), Nx)
                               for theta in x.theta])
            x.lpost = self.prior.logpdf(x.theta)
            is_finite = np.isfinite(x.lpost)
            if t >= 0:
                for m, pf in enumerate(x.pfs):
                    if is_finite[m]:
                        for _ in range(t + 1):
                            next(pf)
                        x.lpost[m] += pf.logLt
        return func

    def _M0(self, N):
        x0 = ThetaParticles(theta=self.prior.rvs(size=N),
                            shared={'Nxs': [self.init_Nx]})
        self.current_target(0, self.init_Nx)(x0)
        return x0

    def M(self, t, xp):
        if xp.shared['rs_flag']:
            return self.move(xp, self.current_target(t - 1, xp.pfs[0].N))
            # in IBIS, target at time t is posterior given y_0:t-1
        else:
            return xp

    def exchange_step(self, x, t, new_Nx):
        old_lpost = x.lpost.copy()
        # exchange step occurs at beginning of step t, so y_t not processed yet
        self.current_target(t - 1, new_Nx)(x)
        return x.lpost - old_lpost

    def summary_format(self, smc):
        msg = FKSMCsampler.summary_format(self, smc)
        return msg + ', Nx=%i' % smc.X.pfs[0].N