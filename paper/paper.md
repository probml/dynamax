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
    affiliation: "1" # (Multiple affiliations must be quoted)
    corresponding: true
  - name: Peter Chang
    affiliation: "2"
  - name: Giles Harper-Donnelly
    affiliation: "3"
  - name: Aleyna Kara
    affiliation: "4"
  - name: Xinglong Li
    affiliation: "5"
  - name: Gerardo Duran-Martin
    affiliation: "6"
  - name: Kevin Murphy
    affiliation: "7"
    corresponding: true
affiliations:
 - name: Department of Statistics and Wu Tsai Neurosciences Institute, Stanford University, USA
   index: 1
 - name: CSAIL, Massachusetts Institute of Technology, USA
   index: 2
 - name: Cambridge University, England, UK
   index: 3
 - name: Computer Science Department, Technical University of Munich Garching, Germany
   index: 4
 - name: Statistics Department, University of British Columbia, Canada
   index: 5
 - name: Queen Mary University of London, England, UK
   index: 6
 - name: Google DeepMind, USA
   index: 7
 
date: 19 July 2024
bibliography: paper.bib

---

# Summary

State space models (SSMs) are fundamental tools for modeling sequential data. They are broadly used across engineering disciplines like signal processing and control theory, as well as scientific domains like neuroscience [@vyas2020computation], genetics [@durbin1998biological], ecology [@patterson2008state], computational ethology [@weinreb2024keypoint], economics [@jacquier2002bayesian], and climate science [@ott2004local]. Fast and robust tools for state space modeling are crucial to researchers in all of these application areas.

State space models specify a probability distribution over a sequence of observations, $y_1, \ldots y_T$, where $y_t$ denotes the observation at time $t$. The key assumption of an SSM is that the observations arise from a sequence of _latent states_, $z_1, \ldots, z_T$, which evolve according to a _dynamics model_ (aka transition model). An SSM may also use inputs (aka controls or covariates), $u_1,\ldots,u_T$, to steer the latent state dynamics and influence the observations. 
For example, in a neuroscience application from @vyas2020computation, $y_t$ represents a vector of spike counts from $\sim 1000$ measured neurons, and $z_t$ is a lower dimensional latent state that changes slowly over time and captures correlations among the measured neurons. If sensory inputs to the neural circuit are known, they can be encoded in $u_t$. 
In the computational ethology application of @weinreb2024keypoint, $y_t$ represents a vector of 3D locations for several key points on an animal's body, and $z_t$ is a discrete behavioral state that specifies how the animal's posture changes over time.
In both examples, there are two main objectives: First, we aim to infer the latent states $z_t$ that best explain the observed data; formally, this is called _state inference_. 
Second, we need to estimate the dynamics that govern how latent states evolve; formally, this is part of the _parameter estimation_ process. 
`Dynamax` provides algorithms for state inference and parameter estimation in a variety of SSMs. 

There are a few key design choices to make when constructing an SSM:

- What is the type of latent state? E.g., is $z_t$ a continuous or discrete random variable?
- How do the latent states evolve over time? E.g., are the dynamics linear or nonlinear?
- How are the observations distributed? E.g., are they Gaussian, Poisson, etc.?

Some design choices are so common they have their own names. Hidden Markov models (HMM) are SSMs with discrete latent states, and linear dynamical systems (LDS) are SSMs with continuous latent states, linear dynamics, and additive Gaussian noise.  `Dynamax` supports canonical SSMs and allows the user to construct bespoke models as needed.

Finally, even for canonical models, there are several algorithms for state inference and parameter estimation. `Dynamax` provides robust implementations of several low-level inference algorithms to suit a variety of applications, allowing users to choose among a host of models and algorithms for their application. More information about state space models and algorithms for state inference and parameter estimation can be found in the textbooks by @murphy2023probabilistic and @sarkka2023bayesian. 


# Statement of need

`Dynamax` is an open-source Python package for state space modeling. Since it is built with `JAX` [@jax], it supports just-in-time (JIT) compilation for hardware acceleration on CPU, GPU, and TPU machines. It also supports automatic differentiation for gradient-based model learning. While other libraries exist for state space modeling in Python [@pyhsmm; @ssm; @eeasensors; @seabold2010statsmodels; @hmmlearn],  `Dynamax` provides a unique combination of low-level inference algorithms and high-level modeling objects that can support a wide range of research applications in JAX.

The API for `Dynamax` is divided into two parts: a set of core, functionally pure, low-level inference algorithms, and a high-level, object oriented module for constructing and fitting probabilistic SSMs. The low-level inference API provides message passing algorithms for several common types of SSMs. For example, `Dynamax` provides `JAX` implementations for:

- Forward-Backward algorithms for discrete-state hidden Markov models (HMMs), 
- Kalman filtering and smoothing algorithms for linear Gaussian SSMs, 
- Extended and unscented generalized Kalman filtering and smoothing for nonlinear and/or non-Gaussian SSMs, and
- Parallel message passing routines that leverage GPU or TPU acceleration to perform message passing in sublinear time [@stone1975parallel; @sarkka2020temporal; @hassan2021temporal]. 

The high-level model API makes it easy to construct, fit, and inspect HMMs and linear Gaussian SSMs. Finally, the online `Dynamax` documentation and tutorials provide a wealth of resources for state space modeling experts and newcomers alike.

`Dynamax` has supported several publications. The low-level API has been used in machine learning research [@zhao2023revisiting; @lee2023switching; @chang2023low]. Special purpose libraries have been built on top of `Dynamax`, like the Keypoint-MoSeq library for modeling animal behavior [@weinreb2024keypoint] and the Structural Time Series in JAX library, `sts-jax` [@sts-jax]. Finally, the `Dynamax` tutorials are used as reference examples in a major machine learning textbook [@murphy2023probabilistic].  

# Acknowledgements

A significant portion of this library was developed while S.W.L. was a Visiting Faculty Researcher at Google and P.C., G.H.D., A.K., and X.L. were Google Summer of Code participants. 

# References
