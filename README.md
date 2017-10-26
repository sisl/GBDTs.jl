# GBDTs.jl -- Grammar-Based Decision Trees

Grammar-based decision tree (GBDT) is an interpretable machine learning model that can be used for the classification and categorization of heterogeneous multivariate time series data. GBDTs combine decision trees with a grammar framework. Each split of the decision tree is governed by a logical expression derived from a user-supplied grammar. The flexibility of the grammar framework enables GBDTs to be applied to a wide range of problems. In particular, GBDT has been previously applied to analyze multivariate heterogeneous time series data of failures in aircraft collision avoidance systems [1].

[1] Lee et al. "Interpretable Categorization of Heterogeneous Time Series Data", preprint, 2018.

## Main Dependencies

* sisl/ExprOptimization.jl
* sisl/MultivariateTimeSeries.jl

## Usage

Please see the [example notebook](http://nbviewer.ipython.org/github/sisl/GBDTs.jl/blob/master/examples/Auslan.ipynb).

## Maintainers:

* Ritchie Lee, ritchie.lee@sv.cmu.edu

[![Build Status](https://travis-ci.org/sisl/GBDTs.jl.svg?branch=master)](https://travis-ci.org/sisl/GBDTs.jl) [![Coverage Status](https://coveralls.io/repos/sisl/GBDTs.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/sisl/GBDTs.jl?branch=master)
