# Background
+ multi-objects optimization
  + regard it as NE of Game but a local minimum
+ stable fixed point in general
# outline
+ Nash Equilibria in nPlayers Games
+ access to NE with Gradient Descent
+ Potential and Hamiltonian Games
+ Neural Net as player

# Stability of NE
  (zero-sum game in the bimatrix case)

# Symetic Gradient Descent
## Potential Games
+ why call *potential*
+ Monderer and Shapley, 1998
+ a game that the losses of players $l_i(\bold w)$can be discribed by a *potential function* $\phi(\bold w)$, so the players act in policy space just as particles act in potential field
+ reflect **hypercoalitional**
### convergency of GD on it
## Hamiltonian Games
+ why *Hamiltonian*
+ reflects **hyperadversarial**
### convergency of GD on it
## simultaneous gradient first
$$\bold{\xi(w)}=(\dots,\nabla_\bold w l,\dots)$$

## Hessian of a game
$$\mathbf H(\bold w)=\nabla_\bold w \cdot\bold{\xi(w)^\top}$$
+ Helmholtz decomposition
  + $\bold A$ is a vector field, there is a unique decomposition $\bold{A=A_P+A_R}$, where $\bold{A_P}$is longitudinal (diverging) field **[grandient of scale field]** and $\bold{A_R}$ is transverse (curling) field **[curl of vector field]**
  + generalized </br> $\bold{H(w)=S(w)+A(w)}$, where $\bold{S(w)}$ is symmetric and $\bold{A(w)}$ is antisymmetric
    + to proof it can write down the unique decomposition
    + ***Potential Game if $\bold{A(w)}\equiv0$***
    + ***Hamiltonian Game if $\bold{S(w)\equiv}0$***
      + are they necessary?
    + both can be solved by GD
+ if players imporve their policies by GD, they must consider $\bold{\xi(w)}$ and $\bold{H(w)}$
# Fixed Points
## Stability
+ (the point that $\text{SGA}(\bold w)=\bold w$? Or just where GD stops?)
+ if $\bold{S(w)\succeq}0$, the fixed point $\bold{w^*}$ (with $\bold{\xi(w^*)}$) is stable
## NE
+ if $\bold{w^*}$ is stable then it is the **local** NE

# Algorithm
## finding fixed point 
+ Hamiltonian: $\mathcal{H}(\bold w):=\frac12\Vert\bold{\xi(w)}\Vert^2$ (how to construct?)
+ Gradient on it $\nabla\mathcal H=\mathbf{H^\top\xi}$
### consensus optimization
+ $$\xi\leftarrow\xi+\lambda\mathbf H^\top\xi$$
  + actually $\xi=\xi+\lambda\nabla\mathcal H$
+ does not work well in general games
## stable
+ **adjustment** of the game dynamics: $\xi_\lambda$
+ it should satisfy:
  + compability:
    1. dynamics: $\langle\xi_\lambda,\xi\rangle=\alpha_1\Vert\xi\Vert^2$
    2. potential dynamics: in PG $\langle\xi_\lambda,\nabla\phi\rangle=\alpha_2\Vert\nabla\phi\Vert^2$
    3. Hamiltonian dunamics: in HG $\langle\xi_\lambda,\nabla\mathcal H\rangle=\alpha_3\Vert\nabla\mathcal H\Vert^2$
  + atttract: in neighborhoods where $\bold S\succ0$, $\theta(\xi_\lambda,\nabla\mathcal H)\leq\theta(\xi,\nabla\mathcal H)$
  + repel: in neighborhoods where $\bold S\prec0$, $\theta(\xi_\lambda,\nabla\mathcal H)\geq\theta(\xi,\nabla\mathcal H)$
## Symplectic Grandient Adjustment (SGA)
+ $\xi_\lambda:=\xi+\lambda\mathbf A^\top\xi$
## $\lambda$
+ how to pick a suitable $\lambda$
## alignment
+ to point the direction of $\xi_\lambda$ (see the figure)
+ $\text{align}(\xi_\lambda,\bold w):=\frac{d}{d\lambda}\{\cos^2[\theta(\xi_\lambda,\bold w)]\}|_{\lambda=0}$
+ check the properties of fixed-point by$\langle\xi,\nabla\mathcal H\rangle$ (stability of fixed-point) and$\langle\bold A^\top\xi,\nabla\mathcal H\rangle$ (towards or away from the fixed-point) 
+ therefore $\lambda$ should satisfy $\lambda\langle\xi,\nabla\mathcal H\rangle\langle\bold A^\top\xi,\nabla\mathcal H\rangle\geq0$
### aligned consensus optimization
+ $\xi\leftarrow\xi+|\lambda|\text{sign}(\langle\xi,\nabla\mathcal H\rangle)\bold H^\top\xi$
#### Performence
+ behaves strangely in PG ('inverse' of Newton's Method)
+ may imporves performance in GANs (fgiure 9)
+ drop the first term $\xi$ performs poorly (attracts to saddle?)
# Experiments
+ compare with Simultaneous GD, Optimistic Mirror Descent & Consensus Optimization
## 1. strong rotational
+ compare with SGD
+ the losses are
$$
l_1=\frac12x^2+10xy\qquad l_2=\frac12y^2-10xy
$$
with fixed point $(0,0)$ and $\xi=(x+10y\ ,\ y-10x)$
## 2. zero-sum bimatrix game
+ compare with OMD
+ the losses are
$$
l_1=\bold w_1^\top\bold w_2\qquad l_2=-\bold w_1^\top\bold w_2
$$
## 3. GANs
+ compare with SGD & CO (and the alignment)
+ ![figure8](https://s2.loli.net/2022/03/02/5rVm38xSzLAMIXG.png)
# Advantage
+ the analysis is indifferent to the number of players
+ regardless of opponents' policy and what players need is only their own strategy and performance

