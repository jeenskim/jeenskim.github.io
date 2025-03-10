---
layout: post
title: GPU acceleration on FeniCS using Eigen + Cupy
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/GPU_acceleration.png
share-img: /assets/img/GPU_acceleration.png
tags: [HPC, GPU]
---

_The thumbnail image is created by chatGPT-4o_
###### This content explains how to implement GPU acceleration on FeniCS using Eigen + Cupy.
<br/>

<br/>


### 1. Install Cupy based on the installed CUDA version
(<https://pypi.org/project/cupy-cuda12x/>)

```
pip install cupy-cuda12x
```

<br/>


### 2. Import required library

```
from fenics import * 
import matplotlib.pyplot as plt 
import numpy as np 
import time
import matplotlib.pyplot as plt
import cupy
import cupyx.scipy.sparse
import cupyx.scipy.sparse.linalg
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
mempool = cupy.get_default_memory_pool()
with cupy.cuda.Device(0):
    mempool.set_limit(size=40*1024**3)
parameters['linear_algebra_backend'] = 'Eigen'
```

<br/>


### 3. Define helper function

```
def tran2SparseMatrix(A):
    row, col, val = as_backend_type(A).data()
    return sps.csr_matrix((val, col, row))
```

<br/>


### 4. Define simulation parameters

```
T = 2.0            # final time
num_steps = 5000   # number of time steps
dt = T / num_steps # time step size
mu = 0.001         # dynamic viscosity
rho = 1            # density
```

<br/>

### 5. Load mesh

```
msh = Mesh()
with XDMFFile(MPI.comm_world, 'cylinder_dense.xdmf') as xdmf_file:
    xdmf_file.read(msh)
```

<br/>


### 6. Define Function Space

```
V = VectorFunctionSpace(msh, 'P', 2)
Q = FunctionSpace(msh, 'P', 1)
```

<br/>

### 7. Define Boundaries

```
inflow   = 'near(x[0], 0)'
outflow  = 'near(x[0], 2.2)'
walls    = 'near(x[1], 0) || near(x[1], 0.41)'
cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'

# Define inflow profile
inflow_profile = ('4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0')

# Define boundary conditions
bcu_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), inflow)
bcu_walls = DirichletBC(V, Constant((0, 0)), walls)
bcu_cylinder = DirichletBC(V, Constant((0, 0)), cylinder)
bcp_outflow = DirichletBC(Q, Constant(0), outflow)
bcu = [bcu_inflow, bcu_walls, bcu_cylinder]
bcp = [bcp_outflow]
```

<br/>

### 8. Define trial and test functions

```
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

u_n = Function(V)
u_  = Function(V)
p_n = Function(Q)
p_  = Function(Q)
```

<br/>

