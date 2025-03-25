---
layout: post
title: GPU acceleration on FeniCS using PETSc
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/GPU_acceleration_petsc.png
share-img: /assets/img/GPU_acceleration_petsc.png
tags: [HPC, GPU]
---

_The thumbnail image is created by chatGPT-4o_
###### This content explains how to implement GPU acceleration on FeniCS using PETSc+GPUs.
<br/>

<br/>

_Will be updated soon..._

Let's say we are working on FEM simulation and want to get solution field at the next timestep. 
This process includes two steps:
1) `Assemble`
2) Solving linear system, $Ax = b$

Here, `Assemble` computes element matrix $A_e$ and element vector $b_e$ for each element, implements boundary conditions for $A_e$ & $b_e$, and constructs global matrix $A$ and global vector $b$. 

And a sparse matrix solver from different linear algebra backgrounds (e.g. PETSc, Eigen, CuPy, and etc.) solves the linear system to get the solution at the next timestep.

Generally, `Assemble` process takes 10~20 % of the total computation time and Solving linear system takes 80~90%.

For one possible way to accelerate this simulation using GPU, this `Assemble` process can be run on GPUs. For this, `assemble()` function in FeniCS should be implemented using CUDA kernel because this function includes conducting integration on different meshes across domain. And this is not officially supported in FeniCSx yet, and it is non-trivial to implement this function using CUDA kernel. There are some ongoing works on GPU implementation of finite element assembly.
(<https://www.sciencedirect.com/science/article/pii/S0167819123000571>)
(<https://www.youtube.com/watch?v=HV8zgxN9SFI>)

On the other hand, for solving a lienar system with a sparse matrix, many linear algebra backends support GPU acceleration. For example, CuPy converts assembled $A$ and $b$ to CSR(Compact Sparse Row) format on a GPU and solves a sparse linear system.

```

```

However, since CuPy doesn't support multi-GPUs in its basic setting, scaling up to larger problems can be difficult. 

<br/>
On the other hand, _'PETSc'_ linear algebra backend in FeniCSX supports multi-GPUs setting and enables executing $assemble()$ function on GPUs.
There is recent paper on this: 

(<https://www.sciencedirect.com/science/article/pii/S0167819123000571>)
(<https://www.youtube.com/watch?v=HV8zgxN9SFI>)
