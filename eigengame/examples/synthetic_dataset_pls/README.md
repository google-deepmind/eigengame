# PLS on a Synthetic Dataset using Generalized EigenGame

This example provides a simple experiment running PLS on a synthetic dataset.

Recall that PLS can be modeled as a generalized eigenvalue problem, $$ A v =
\lambda B v $$, by defining the $$ A $$ and $$ B $$ matrices as follows:
$$ A=\begin{bmatrix}0&\mathbb{E}[x y^\top]\\\mathbb{E}[y x^\top]&0\end{bmatrix},
B=\begin{bmatrix}\mathbb{E}[x x^\top]&0\\0&Id\end{bmatrix} $$. Notice that it is
equivalent to the CCA formulation except where $$ \mathbb{E}[y y^\top] $$ has
been replaced by the identity matrix.

We generate a dataset by by following the same steps in the demo included
on the [sklearn website](https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_compare_cross_decomposition.html)

For $$ d < 5 \times 10^3 $$ we can solve for the ground truths using scipy and
compare with the eigenvector results from EigenGame using cosine error.
