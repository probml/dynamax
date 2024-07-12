---
title: 'Dynamax: A Python package for probabilistic state space modeling with JAX'
tags:
  - Python
  - State space models
  - dynamics
  - JAX

authors:
  - name: Scott W. Linderman
    orcid: 0000-0002-3878-9073
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
    corresponding: true
  - name: Peter Chang
    affiliation: "3"
  - name: Giles Harper-Donnelly
    affiliation: "4"
  - name: Aleyna Kara
    affiliation: "5"
  - name: Xinglong Li
    affiliation: "6"
  - name: Kevin Murphy
    affiliation: "2"
    corresponding: true
affiliations:
 - name: Department of Statistics and Wu Tsai Neurosciences Insitute, Stanford University, USA
   index: 1
 - name: Google Research, USA
   index: 2
 - name: CSAIL, Massachusetts Institute of Technology, USA
   index: 3
 - name: Cambridge University, UK
   index: 4
 - name: Boğaziçi University, Turkey
   index: 5
 - name: University of British Columbia, Canada
   index: 6
 
date: 12 July 2024
bibliography: paper.bib

---

# Summary

State space models (SSMs) are fundamental tools for modeling sequential data. They are broadly used across engineering disciplines like signal processing and control theory, as well as scientific domains like neuroscience, genetics, ecology, and climate science. Fast and robust tools for state space modeling are crucial to researchers in all of these application areas.

State space models specify a probability distribution over a sequence of observations, $y_1, \ldots y_T$, where $y_t$ denotes the observation at time $t$. The key assumption of an SSM is that the observations arise from a sequence of _latent states_, $z_1, \ldots, z_T$, which evolve according to a _dynamics model_ (aka transition model). An SSM may also use inputs (aka controls or covariates), $u_1,\ldots,u_T$, to steer the latent state dynamics and influence the observations. 

For example, SSMs are often used in neuroscience to model the dynamics of neural spike train recordings [@vyas2020computation]. Here, $y_t$ is a vector of spike counts from each of, say, 100 measured neurons. The activity of nearby neurons is often correlated, and SSMs can capture that correlation through a lower dimensional latent state, $z_t$, which may change slowly over time. If we know that certain sensory inputs may drive the neural activity, we can encode them in $u_t$. A common goal in neuroscience is to infer the latent states $z_t$ that best explain the observed neural spike train; formally, this is called _state inference_. Another goal is to estimate the dynamics that govern how latent states evolve; formally, this is part of the _parameter estimation_ process. `Dynamax` provides algorithms for state inference and parameter estimation in a variety of SSMs. 

The key design choices when constructing an SSM include the type of latent state (is $z_t$ a continuous or discrete random variable?), the dynamics that govern how latent states evolve over time (are they linear or nonlinear?), and the conditional distribution of the observations (are they Gaussian, Poisson, etc.?). Canonical examples of SSMs include hidden Markov models (HMM), which have discrete latent states, and linear dynamical systems (LDS), which have continuous latent states with linear dynamics and additive Gaussian noise. `Dynamax` supports these canonical examples as well as more complex models. 

More information about state space models and algorithms for state inference and parameter estimation can be found in the textbooks by @murphy2023probabilistic and @sarkka2023bayesian. 


# Statement of need

`Dynamax` is an open-source Python pacakge for state space modeling. Since it is built with `JAX` [@jax], it supports just-in-time (JIT) compilation for hardware acceleration on CPU, GPU, and TPU machines. It also supports automatic differentiation for gradient-based model learning. While other libraries exist for state space modeling in Python (and some also use `JAX`), this library provides a unique combination of low-level inference algorithms and high-level modeling objects that can support a wide range of research applications.

The API for `Dynamax` is divided into two parts: a set of core, functionally pure, low-level inference algorithms, and a high-level, object oriented module for constructing and fitting probabilistic SSMs. The low-level inference API provides message passing algorithms for several common types of SSMs. For example, `Dynamax` provides `JAX` implementations for:

- Forward-Backward algorithms for discrete-state hidden Markov models (HMMs), 
- Kalman filtering and smoothing algorithms for linear Gaussian SSMs, 
- Extended and unscented Kalman filtering and smoothing for nonlinear Gaussian SSMs, 
- Conditional moment filtering and smoothing algorithms for models with non-Gaussian emissions, and
- Parallel message passing routines that leverage GPU or TPU acceleration to perform message passing in sublinear time. 

The high-level model API makes it easy to construct, fit, and inspect HMMs and linear Gaussian SSMs. Finally, the online `Dynamax` documentation and tutorials provide a wealth of resources for state space modeling experts and newcomers alike.

`Dynamax` has supported several publications. The low-level API has been used in machine learning research [@zhao2023revisiting; @lee2023switching; @chang2023low]. More sophisticated, special purpose models on top of `Dynamax`, like the Keypoint-MoSeq library for modeling postural dynamics of animals [@weinreb2024keypoint]. Finally, the `Dynamax` tutorials are used as reference examples in a major machine learning textbook [@murphy2023probabilistic].  

# Acknowledgements

A significant portion of this library was developed while S.W.L. was a Visiting Faculty Researcher at Google and P.C., G.H.D., A.K., and X.L. were Google Summer of Code participants. 

# References