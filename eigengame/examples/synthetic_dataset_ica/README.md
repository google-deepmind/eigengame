# ICA on a Synthetic Dataset using Generalized EigenGame

This example provides a simple experiment running ICA on a synthetic dataset.

Recall that ICA can be modeled as a generalized eigenvalue problem, $$ A v =
\lambda B v $$, by defining the $$ A $$ and $$ B $$ matrices as follows:
$$ A=\mathbb{E}[\langle x, x \rangle x x^\top] - tr(B)B - B^2,
B=\mathbb{E}[x x^\top] $$.

We generate 8 synthetic periodic signals starting from the basic demo included
on the [sklearn website](https://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_blind_source_separation.html)

We can solve for the ground truths using scipy and compare with the eigenvector
results from EigenGame using cosine error.

In this case, we are searching for the vectors with maximum 'negative'
eigenvalues. There are only 6 negative eigenvalues. Hence, this example also
demonstrates a limitation of the algorithm. Any eigenvector can always 'copy' an
'uncopied' parent in order to achieve zero utility. Therefore, we only expect to
find the bottom 6 eigenvectors when solving top-6 SGEP(-A, B). If we want to
recover the two positive eigenvectors, we need to run SGEP(A, B) in a separate
run.
