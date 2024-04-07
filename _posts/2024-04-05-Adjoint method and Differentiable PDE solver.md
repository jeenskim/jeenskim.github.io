---
layout: post
title: Adjoint method and Differentiable PDE solver
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/POD_thumb.png
share-img: /assets/img/POD_thumb.png
tags: [Theory]
---

#### This content is based on *"Differentiable Turbulence II"* by Varun Shankar et al. [^1]. 

1. Adjoint method
   - Purpose
   - Derivation<br>
2. Differentiable PDE solver
   - Purpose
   - Implementation
   
<br>
### 1. Adjoint method
#### 1.1. Purpose
&nbsp; The method of Lagrangian multiplier provides a way to find optimal solutions in the optimization problem with constraints. However, for complex systems like systems of PDE, it is really difficult to get the optimal solution directly from the method of Lagrangian multiplier. The Adjoint method helps to find solutions to complex optimization problems by providing a gradient of the objective function with respect to optimization variables.
   $test$

#### 1.2. Derivation
&nbsp; Let's say we have the objective function $f(x, m)$ and constraint $g(x, m)$, where **x** is the state variable, and **m** is the optimization variable. Our goal is to calculate the total derivative of the objective function with respect to the optimization variable, $\frac{df}{dm}$. <br/><br/>
And define Lagrangian $L(x, m, \lambda)$ as follows. <br/><br/>
 $$
 L(x, m, \lambda) = f(x, m) + \lambda^Tg(x, m) 
 $$ 
 <br>
 ,where $\lambda$ is the adjoint variable or Lagrangian multiplier.<br/><br/>
 
 Sicne $g(x,m) = 0$ everywhere, $f(x,m)$ is equivalent to $L(x, m, \lambda)$, and we can choose $\lambda$ freely. Also, we can think $f(x,m)$ as $f(x(m))$. <br/><br/>
 Then, $\frac{df}{dm}$ becomes <br/><br/>
 $$ \frac{df}{dm} = \frac{dL}{dm} = \frac{\partial f}{\partial x}\frac{dx}{dm} + \frac{d\lambda^T}{dm}g + \lambda^T(\frac{\partial g}{\partial m} + \frac{\partial g}{\partial x}\frac{dx}{dm})$$ <br/><br/>

 Since $g(x,m)=0$, it becomes <br/><br/>
 $$
 \frac{df}{dm} = (\frac{\partial f}{\partial x}+\lambda^T\frac{\partial g}{\partial x})\frac{dx}{dm}+\lambda^T\frac{\partial g}{\partial m}
 $$
 <br/><br/>

 Generally, it is difficult to know $\frac{dx}{dm}$. Therefore, by setting <br/><br/>
 $$
 \frac{\partial f}{\partial x} + \lambda^T \frac{\partial g}{\partial x} = 0
 $$
 <br/><br/>
 we don't need to compute $\frac{dx}{dm}$ <br/><br/>
 Then,
 $$
 \frac{df}{dm} = \lambda^T\frac{\partial g}{\partial m}
 $$
<br/><br/>
,which can be calculated using $\lambda$ from the above equation. <br>

In summary, by utilizing Lagrangian and setting the adjoint variable properly, we can compute a gradient of the objective function with regard to parameters.

### 2. Differentiable PDE solver
#### 2.1. Purpose
&nbsp; Turbulence modeling with machine learning techniques usually has taken _a-priori_ learning. While _a-priori_ approach in modeling turbulence is easy to implement, it can't guarantee accurate prediction for unseen data. This is because when a learned model is embedded into the PDE solver, numerical artifacts, such as numerical errors and temporal errors, are produced by the PDE solver. These artifacts are not included in the training process, thus deteriorating prediction capability for unseen data. <br>

&nbsp: However, differentiable PDE solver enables end-to-end training of machine learning models with backpropagation, enhancing _a-posteriori_ prediction capability. A differentiable PDE solver can be realized by implementing a governing equation in an automatic differentiation (AD) framework, which is a necessary tool for training deep neural networks. <be>

#### 2.2. Implementation with the adjoint method
&nbsp; The AD framework's basic principle is that gradients of a composition of differentiable functions can be exactly computed using the chain rule. Deep neural networks are composed of straightforward operations, such as linear or pointwise nonlinear operations. Therefore, gradients can be easily computed in the training of deep neural networks using the AD framework. However, writing a PDE solver in an AD-enabled way can add enormous complexity, which makes it impractical. Instead, by implementing a vector-jacobian product using the discrete adjoint method, we can integrate a PDE solver within the AD framework.




[^1]: [Shankar, Varun, Romit Maulik, and Venkatasubramanian Viswanathan. "Differentiable Turbulence II." arXiv preprint arXiv:2307.13533 (2023).](https://doi.org/10.48550/arXiv.2307.13533) 

