# PCA on a Synthetic Dataset using Generalized EigenGame

This example provides a simple experiment running PCA on a synthetic dataset.

Recall that PCA can be modeled as a generalized eigenvalue problem, $` A v =
\lambda B v `$, by defining the $` A `$ and $` B `$ matrices as follows:
$` A=\mathbb{E}[x x^\top], B=I `$.

We generate a dataset by first generating a random matrix
$` M \in \mathbb{R}^{d \times n} `$ and then using
$` M^\top M \in \mathbb{R}^{d \times d} `$ as a covariance matrix. We then draw
$` d `$ dimensional vectors from a multivariate normal with this covariance.

For $` d < 5 \times 10^3 `$ we can solve for the ground truths using scipy and
compare with the eigenvector results from EigenGame using cosine error.
