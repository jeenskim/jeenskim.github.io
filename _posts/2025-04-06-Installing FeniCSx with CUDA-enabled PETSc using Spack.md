---
layout: post
title: How to install FeniCSx with CUDA-enabled PETSc using Spack
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/Fenicsx-spack.png
share-img: /assets/img/Fenicsx-spack.png
tags: [HPC, GPU, FeniCSx]
---

_The thumbnail image is created by chatGPT-4o_
###### This content explains how to to install FeniCSx with CUDA-enabled PETSc using Spack on HPC systems (ALCF Polaris).
<br/>

<br/>

### 1. Acclerating FenicsX simulation using GPUs
Let's say we are working on FEM simulation and want to get solution field at the next timestep. 
This process includes two steps:
1) `Assemble`
2) Solving linear system, $Ax = b$

Here, `Assemble` computes element matrix $A_e$ and element vector $b_e$ for each element, implements boundary conditions for $A_e$ & $b_e$, and constructs global matrix $A$ and global vector $b$. 

And a sparse matrix solver from different linear algebra backgrounds (e.g. PETSc, Eigen, CuPy, and etc.) solves the linear system to get the solution at the next timestep.

Generally, for systems with large dofs such as LES simulations, `Assemble` process takes 10~20 % of the total computation time and Solving linear system takes 80~90%. Therefore, it is more efficient to accelerate the process of solving linear systems.

PETSc provides GPU support for solving linear systems. To that end, the type of `petsc4py.PETSc.Vec` and 'petsc4py.PETSc.Mat' should be set as a GPU-compatible type using `petsc4py.PETSc.Vec.SetType('CUDA')` and `petsc4py.PETSc.Vec.SetType('AIJCUSPARSE')`, respectively.

```
A.setType("AIJCUSPARSE")
x.setType("CUDA")
b.setType("CUDA")
```

To use this option, PETSc should be configured with cuda option. In addition, hypre option should be activated while configuring PETSc to use the hypre preconditioner. 

In spack, PETSc with cuda and hypre option can be installed using this `spack.yaml` file:

```
# This is a Spack Environment file.
#
# It describes a set of packages to be installed, along with
# configuration settings.
spack:
  specs:
    - petsc +cuda +hypre cuda_arch=80
    - hypre +cuda cuda_arch=80
    - fenics-dolfinx+adios2
    - py-fenics-dolfinx cflags=-O3 fflags=-O3
    - py-torch+cuda cuda_arch=80
    - py-pip
  view: true
  concretizer:
    unify: true
```

