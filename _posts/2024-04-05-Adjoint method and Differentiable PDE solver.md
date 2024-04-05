---
layout: post
title: Adjoint method and Differentiable PDE solver
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/POD_thumb.png
share-img: /assets/img/POD_thumb.png
tags: [Theory]
---

#### This content is based on *"Differentiable Turbulence II"* by Varun Shankar et al. [^1]. 
<br>

1. Adjoint method
   - Purpose
   - Derivation<br>
2. Differentiable PDE solver
   - Purpose
   - Implementation
   
<br>
### 1. Adjoint method
#### 1.1. Purpose
   The method of Lagrangian multiplier provides a way to find optimal solutions in the optimization problem with constraints. However, for complex systems like systems of PDE, it is really difficult to get the optimal solution directly from the method of Lagrangian multiplier. The Adjoint method helps to find solutions to complex optimization problems by providing a gradient of the objective function with respect to optimization variables.
   $test$

#### 1.2. Derivation
 Let's say we have the objective function 
 $f(<b>x</b>, m)$ and constraint $g(x, m)$, where **x** is the state variable and **m** is the optimization variable.
 And define Lagrangian $L(**x**, **m**, \lambda)$ as following.
 $$L(**x**, **m**, \lambda) = f(**x**, **m**) + \lambda^Tg(**x**, **m**) $$
 ,where $\lambda$ is the adjoint variable or Lagrangian multiplier.


2. Examples

3. Outlook




[^1]: [Shankar, Varun, Romit Maulik, and Venkatasubramanian Viswanathan. "Differentiable Turbulence II." arXiv preprint arXiv:2307.13533 (2023).](
https://doi.org/10.48550/arXiv.2307.13533) 

