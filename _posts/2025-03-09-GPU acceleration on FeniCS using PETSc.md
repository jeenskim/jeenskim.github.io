---
layout: post
title: GPU acceleration on FeniCS using PETSc+GPUs
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

Let's say we are working on FEM simulation using FeniCS and want to solve a linear system as follows:

$Ax = b$.

When trying to accelerate this with _'Eigen'_ linear algebra backend and Cupy, 

The LHS matrix can be precomputed by using $assemble()$ and converted to CSR(Compact Sparse Row) format on a GPU.

However, the RHS vector, $b$, should be computed using $assemble()$ and boundary conditons and converted to CSR format at each time step using CPU. This is becuase $assemble()$ function includes conducting integration on different meshes across domain and it is non-trivial to implement this function using CUDA kernel. As a result, this can cause a bottleneck in the whole process.

In addition, since Cupy doesn't support multi-GPUs in its basic setting, scaling up to larger problems can be difficult.


