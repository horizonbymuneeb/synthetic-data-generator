"""Gaussian copula generator for mixed-type tables.

The copula separates two questions that a naive sampler answers at once: *which
values co-occur* and *what the marginal distribution of each column is*.

1. Each numeric column is rank-normalised: replace every value by its empirical
   quantile and push that through the inverse normal CDF. The result is a
   column that is standard normal by construction, whatever the original shape
   (income is log-normal, tenure is clipped and discrete).
2. A correlation matrix is fitted on the normal scores, not on the raw values,
   so a single heavy tail cannot dominate it. The matrix is repaired to be
   positive semi-definite before sampling.
3. Samples are drawn from the multivariate normal, then each column is mapped
   back through its own empirical quantile function, which restores the exact
   marginal shape - including the discreteness of a column like tenure.
4. Categorical columns are not sampled independently. Each one is conditioned on
   the first numeric column, binned by quantile, so ``plan`` stays correlated
   with income after the synthetic rows are generated.
"""
