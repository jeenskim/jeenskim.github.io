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

### 2. Istalling FenicsX with CUDA-enabled PETSc
To use this option, PETSc should be configured with cuda option. In addition, hypre option should be activated while configuring PETSc to use the hypre preconditioner. 

In spack, PETSc with cuda and hypre option can be installed using this `spack.yaml` file. In addition, the cuda-enbaled version of mpich should be installed together.

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
    - mpich +cuda
    - py-pip
  view: true
  concretizer:
    unify: true
```

The installation of `py-torch` from the source code could take 40-60 minutes.


We can test Fenicsx with the cuda-enbaled PETSc with the following code:


```
from petsc4py import PETSc

vec = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
vec.setSizes(100)
vec.setType("mpicuda")
vec.setFromOptions()
vec.set(1.0)

print("Vec type:", vec.getType())



mat = PETSc.Mat().createAIJ([100, 100])
mat.setFromOptions()
mat.setType("mpiaijcusparse")
mat.setUp()
print("Mat type:", mat.getType())
```

Result
```
hjkim@x3206c0s25b0n0:/grand/NeuralDE/hjkim> mpiexec -n 2 python mpitest.py 
Warning: Permanently added 'x3206c0s25b0n0.hsn.cm.polaris.alcf.anl.gov,10.201.0.255' (ECDSA) to the list of known hosts.
Vec type: mpicuda
Mat type: mpiaijcusparse
Vec type: mpicuda
Mat type: mpiaijcusparse
```

### 3. Performance check
We can check the performance of GPU acceleration by comparing it with CPUs.

CPU execution code

```
from petsc4py import PETSc
from mpi4py import MPI
import time

comm = PETSc.COMM_WORLD
mpi_comm = MPI.COMM_WORLD
rank = comm.getRank()

mpi_comm.Barrier()
t0 = MPI.Wtime()


n = 1000000
A = PETSc.Mat().createAIJ([n, n], comm=comm)
A.setType("mpiaij")  # CPU matrix
A.setFromOptions()
A.setUp()

start, end = A.getOwnershipRange()
for i in range(start, end):
    A.setValue(i, i, 2.0)
    if i > 0:
        A.setValue(i, i - 1, -1.0)
    if i < n - 1:
        A.setValue(i, i + 1, -1.0)
A.assemble()

b = PETSc.Vec().create(comm=comm)
b.setSizes(n)
b.setFromOptions()
b.set(1.0)

x = b.duplicate()

ksp = PETSc.KSP().create(comm=comm)
ksp.setOperators(A)
ksp.setType("cg")
ksp.getPC().setType("jacobi")
ksp.setFromOptions()

ksp.solve(b, x)

mpi_comm.Barrier()
t1 = MPI.Wtime()

elapsed = t1 - t0
max_elapsed = mpi_comm.reduce(elapsed, op=MPI.MAX, root=0)
sum_elapsed = mpi_comm.reduce(elapsed, op=MPI.SUM, root=0)

if rank == 0:
    print(f"[CPU] Total Wall Time (Wtime): {max_elapsed:.4f} sec")
    print(f"[CPU] Sum of All Ranks' Time: {sum_elapsed:.4f} sec")
    print(f"[CPU] Residual norm: {ksp.getResidualNorm():.2e}")

```

GPU execution code

```
import os
import sys
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()

from petsc4py import PETSc
import time
import torch  


comm = PETSc.COMM_WORLD
mpi_comm = MPI.COMM_WORLD
rank = comm.getRank()

mpi_comm.Barrier()
t0 = MPI.Wtime()


if torch.cuda.is_available():
    print(f"[Rank {rank}] Using GPU {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print(f"[Rank {rank}] No GPU available")



n = 10000000
A = PETSc.Mat().createAIJ([n, n], comm=comm)
A.setType("mpiaijcusparse")  # GPU matrix
A.setFromOptions()
A.setUp()

start, end = A.getOwnershipRange()
for i in range(start, end):
    A.setValue(i, i, 2.0)
    if i > 0:
        A.setValue(i, i - 1, -1.0)
    if i < n - 1:
        A.setValue(i, i + 1, -1.0)
A.assemble()

b = PETSc.Vec().create(comm=comm)
b.setSizes(n)
b.setFromOptions()
b.setType("cuda")  # GPU vector
b.set(1.0)

x = b.duplicate()

ksp = PETSc.KSP().create(comm=comm)
ksp.setOperators(A)
ksp.setType("cg")
ksp.getPC().setType("jacobi")
ksp.setFromOptions()

ksp.solve(b, x)

mpi_comm.Barrier()
t1 = MPI.Wtime()

elapsed = t1 - t0
max_elapsed = mpi_comm.reduce(elapsed, op=MPI.MAX, root=0)
sum_elapsed = mpi_comm.reduce(elapsed, op=MPI.SUM, root=0)

if rank == 0:
    print(f"[GPU] Total Wall Time (Wtime): {max_elapsed:.4f} sec")
    print(f"[GPU] Sum of All Ranks' Time: {sum_elapsed:.4f} sec")
    print(f"[GPU] Residual norm: {ksp.getResidualNorm():.2e}")
```

Results

$n=10^6$
```
CPU (# of ranks = # of CPUs)
# of ranks 1: 50.46 sec
# of ranks 2: 26.09 sec
# of ranks 8: 6.60 sec
# of ranks 16: 3.47 sec 
# of ranks 32: 2.36 sec
```

```
GPU (# of ranks = # of GPUs)
# of ranks 1: 3.23 sec
# of ranks 4: 2.6 sec
```


$n = 10^7$
```
GPU (# of ranks = # of GPUs)
# of ranks 1: 284.67 sec
# of ranks 4: 73.04 sec
```

