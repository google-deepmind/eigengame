# CCA on a Synthetic Dataset using Generalized EigenGame

This example provides a simple experiment running CCA on a synthetic dataset.

Recall that CCA can be modeled as a generalized eigenvalue problem, $$ A v =
\lambda B v $$, by defining the $$ A $$ and $$ B $$ matrices as follows:
$$ A=\begin{bmatrix}0&\mathbb{E}[x y^\top]\\\mathbb{E}[y x^\top]&0\end{bmatrix},
B=\begin{bmatrix}\mathbb{E}[x x^\top]&0\\0&\mathbb{E}[y y^\top]\end{bmatrix} $$.

We generate a dataset by first generating a random matrix
$$ M \in \mathbb{R}^{d \times n} $$ and then using
$$ M^\top M \in \mathbb{R}^{d \times d} $$ as a covariance matrix. We then draw
$$ d $$ dimensional vectors from a multivariate normal with this covariance. The
first $$ d / 2 $$ entries denote the ``$$ x $$'' vector while the last
$$ d / 2 $$ entries denote the ``$$ y $$'' vector.

For $$ d < 5 \times 10^3 $$ we can solve for the ground truths using scipy and
compare with the eigenvector results from EigenGame using cosine error.
