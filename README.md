# Calculating dark matter halo shapes in Illustris

This is a script to calculates dark matter halo shapes for the Illustris and IllustrisTNG data.
The results have been published in [Chua K. T. E., Pillepich A., Vogelsberger M., Hernquist L., 2019, MNRAS,
484, 476][http://adsabs.harvard.edu/abs/2019MNRAS.484..476C].
Halo shapes for [IllustrisTNG][www.tng-project.org] is also available at the website.

The algorithm uses the iterative algorithm based on the shape tensor (e.g. Bailin & Steinmetz 2005; Zemp et al. 2011):
\begin{equation}
    S_{ij} = \frac{1}{\sum_k m_k} \sum_k  \frac{1}{w_k} m_k\, r_{k,i} \,r_{k,j}
\end{equation}
$m_k$ is the mass of the $k$th particle, and $r_{k,i}$ is the $i$th component of its position vector.
$w_k$ is a parameter that can be used to weight the contribution of each particle to $S_{ij}$.

The choice of $w_k$ can be dependent on the aspect of halo shape that is under examination.
Common choices of $w_k$ are $w_k$ = 1 and $w_k = r^2_{\rm ell,k}$ where
\begin{equation}
    r_{\rm ell}^2 = x^2 + \frac{y^2}{(b/a)^2} + \frac{z^2}{(c/a)^2}.
\end{equation}
with $(x,y,z)$ being the position of the particle in its principal frame
and $a$, $b$ and $c$ the lengths of the semi-axes.
Here, we use $w_k = 1$, where all particles are unweighted and $S_{ij}$ is proportional to the inertia tensor.

To calculate the shape, the shape tensor is diagonalized to compute its eigenvectors and eigenvalues $\lambda_a$, $\lambda_b$ and $\lambda_c$, with $\lambda_a>\lambda_b>\lambda_c$. The eigenvectors denote the  directions of the principal axes while the eigenvalues denote the principal axes lengths ($a\propto\sqrt{\lambda_a}$, $b\propto\sqrt{\lambda_b}$ and $c\propto\sqrt{\lambda_c}$; where $a > b > c$).
The iterative method begins with particles selected in a spherical shell (i.e. $q=s=1$). In each iteration, the shape tensor of particles in radial bins are computed and diagonalised. Eigenvectors are then used to rotate all particles into the computed principal frame. The process is repeated keeping the semi-major length constant (fixed $r_{\rm ell}$) until $q$ and $s$ converge.
