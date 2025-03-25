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

On the other hand, for solving a lienar system with a sparse matrix, many linear algebra backends support GPU acceleration. For example, CuPy converts assembled $A$ and $b$ to CSR (Compact Sparse Row) format on a GPU and solves a sparse linear system.

```
A1_gpu = cupyx.scipy.sparse.csr_matrix(tran2SparseMatrix(A1))

b1_gpu = cupy.zeros_like(cupy.array(assemble(L1)[:]))
b2_gpu = cupy.zeros_like(cupy.array(assemble(L2)[:]))
b3_gpu = cupy.zeros_like(cupy.array(assemble(L3)[:]))

b1 = assemble(L1)
[bc.apply(b1) for bc in bcu]
b1_gpu = cupy.asarray(b1[:])
    
u_.vector()[:] = cupy.asnumpy(cupyx.scipy.sparse.linalg.cg(A1_gpu, b1_gpu)[0])
```

One limitation of CuPy is that it doesn't support multi-GPUs in its basic setting, so scaling up to larger problems can be difficult.

On the other hand, PETSc support multi-GPUs, and it is better to use PETSc for large scale problems. In addition, PETSc provides different types of preconditioners for linear solvers.
