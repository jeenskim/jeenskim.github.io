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

Cupy vs. Cupyx
- Cupy: GPU version of Numpy
- Cupyx: GPU version of Scipy

<br/>


### 3. Define helper function

```
def tran2SparseMatrix(A):
    row, col, val = as_backend_type(A).data()
    return sps.csr_matrix((val, col, row))
```

``` scipy.sparse.csr_matrix ``` : convert sparse matrix into the CSR(Compressed Sparse Row) matrix 
- More efficient memory
- Faster matrix-vector multiplication

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


### 9. Define variational forms

```
# Define expressions used in variational forms
U  = 0.5*(u_n + u)
n  = FacetNormal(msh)
f  = Constant((0, 0))
k  = Constant(dt)
mu = Constant(mu)
rho = Constant(rho)

# Define symmetric gradient
def epsilon(u):
    return sym(nabla_grad(u))

# Define stress tensor
def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))

# Define variational problem for step 1
F1 = rho*dot((u - u_n) / k, v)*dx \
   + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
   + inner(sigma(U, p_n), epsilon(v))*dx \
   + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
   - dot(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Define variational problem for step 2
a2 = dot(nabla_grad(p), nabla_grad(q))*dx
L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

# Define variational problem for step 3
a3 = dot(u, v)*dx
L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx


```

<br/>


### 10. Assemble to form matrices

```
# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)


# Apply boundary conditions to matrices
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]
```

<br/>

### 11. Create XDMF files for visualization output

```
xdmffile_u = XDMFFile('navier_GPU/velocity_dense.xdmf')
xdmffile_p = XDMFFile('navier_GPU/pressure_dense.xdmf')
```

<br/>


### 12. Convert assembled matrices to sparse format

```
A1_gpu = cupyx.scipy.sparse.csr_matrix(tran2SparseMatrix(A1))
A2_gpu = cupyx.scipy.sparse.csr_matrix(tran2SparseMatrix(A2))
A3_gpu = cupyx.scipy.sparse.csr_matrix(tran2SparseMatrix(A3))
```

```cupyx.scipy.sparse.csr_matrix```: convert CSR matrix at CPU to CSR matrix at GPU

<br/>

### 13. Allocate GPU memory for RHS vectors

```
b1_gpu = cupy.zeros_like(cupy.array(assemble(L1)[:]))
b2_gpu = cupy.zeros_like(cupy.array(assemble(L2)[:]))
b3_gpu = cupy.zeros_like(cupy.array(assemble(L3)[:]))
```

``` cupy.array ```: convert numpy array at CPU to cupy array at GPU


### 14. Timestep integration

```
t = 0
start = time.time()
for n in range(num_steps):

    start_timestep = time.time()

    # Update current time
    t += dt

    # Step 1: Tentative velocity step
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    b1_gpu = cupy.asarray(b1[:])
    
    u_.vector()[:] = cupy.asnumpy(cupyx.scipy.sparse.linalg.cg(A1_gpu, b1_gpu)[0])

    # Step 2: Pressure correction step
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    b2_gpu = cupy.asarray(b2[:])
    
    p_.vector()[:] = cupy.asnumpy(cupyx.scipy.sparse.linalg.cg(A2_gpu, b2_gpu)[0])

    # Step 3: Velocity correction step
    b3_gpu[:] = cupy.asarray(assemble(L3)[:])
    u_.vector()[:] = cupy.asnumpy(cupyx.scipy.sparse.linalg.cg(A3_gpu, b3_gpu)[0])

    # Save solution
    xdmffile_u.write(u_, t)
    xdmffile_p.write(p_, t)

    # Update previous solution
    u_n.assign(u_)
    p_n.assign(p_)
    # print('Time:', t)
    # print('u max:', u_.vector().max())
    # print('p max:', p_.vector().max())

    end_timestep = time.time()
    # print('GPU(s)/iteration', end_timestep-start_timestep)
    print(n, end_timestep - start, end_timestep - start_timestep)

end = time.time()
print("Total GPU execution time:", end - start)
```

<br/>

#### 14.1. IPCS algorithm
```
# Step 1: Tentative velocity step
b1 = assemble(L1)
[bc.apply(b1) for bc in bcu]
b1_gpu = cupy.asarray(b1[:])
    
u_.vector()[:] = cupy.asnumpy(cupyx.scipy.sparse.linalg.cg(A1_gpu, b1_gpu)[0])

# Step 2: Pressure correction step
b2 = assemble(L2)
[bc.apply(b2) for bc in bcp]
b2_gpu = cupy.asarray(b2[:])
    
p_.vector()[:] = cupy.asnumpy(cupyx.scipy.sparse.linalg.cg(A2_gpu, b2_gpu)[0])

# Step 3: Velocity correction step
b3_gpu[:] = cupy.asarray(assemble(L3)[:])
u_.vector()[:] = cupy.asnumpy(cupyx.scipy.sparse.linalg.cg(A3_gpu, b3_gpu)[0])
```

<br/>

```cupyx.scipy.sparse.linalg.cg```: solving linear system using cg solver

<br/>

```cupy.asnumpy```: convert the solution vector to CPU

<br/>

```assemble```: assembling across domain to form the RHS vector at CPU
