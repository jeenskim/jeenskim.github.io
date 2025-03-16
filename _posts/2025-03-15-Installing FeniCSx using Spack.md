---
layout: post
title: How to install FeniCSx using Spack
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/Fenicsx-spack.png
share-img: /assets/img/Fenicsx-spack.png
tags: [HPC, GPU]
---

_The thumbnail image is created by chatGPT-4o_
###### This content explains how to to install FeniCSx using Spack on HPC systems (ALCF Polaris).
<br/>

<br/>

### 1. Install Spack and FeniCSx environment
"Spack is a package management tool designed to support multiple versions and configurations of software on a wide variety of platforms and environments. It was designed for large supercomputing centers, where many users and application teams share common installations of software on clusters with exotic architectures, using libraries that do not have a standard ABI. Spack is non-destructive: installing a new version does not break existing installations, so many configurations can coexist on the same system."

(<https://spack.readthedocs.io/en/latest/>)

```
git clone https://github.com/spack/spack.git
. ./spack/share/spack/setup-env.sh
spack env create fenicsx-env
spack env activate fenicsx-env
spack add fenics-dolfinx+adios2 py-fenics-dolfinx cflags="-O3" fflags="-O3"
spack install
```

### 2. Solve 2D Poisson equation

```
from mpi4py import MPI
from dolfinx import mesh
import socket
import time


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"Hello from rank {rank} out of {size} processes")



domain = mesh.create_unit_square(MPI.COMM_WORLD, 1024, 1024, mesh.CellType.quadrilateral)

from dolfinx.fem import functionspace
V = functionspace(domain, ("Lagrange", 1))


from dolfinx import fem
uD = fem.Function(V)
uD.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)


import numpy
# Create facet to cell connectivity required to determine boundary facets
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)


boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(uD, boundary_dofs)


import ufl
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)


from dolfinx import default_scalar_type
f = fem.Constant(domain, default_scalar_type(-6))


a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx


if rank == 0:
    start = time.time()

from dolfinx.fem.petsc import LinearProblem
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

MPI.COMM_WORLD.Barrier()

if rank ==0:
    print(f'time = {time.time()-start}')

V2 = fem.functionspace(domain, ("Lagrange", 2))
uex = fem.Function(V2)
uex.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)


L2_error = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
error_local = fem.assemble_scalar(L2_error)
error_L2 = numpy.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))


error_max = numpy.max(numpy.abs(uD.x.array-uh.x.array))
# Only print the error on one process
if domain.comm.rank == 0:
    print(f"Error_L2 : {error_L2:.2e}")
    print(f"Error_max : {error_max:.2e}")
    
    
from dolfinx import io
from pathlib import Path
results_folder = Path("results")
results_folder.mkdir(exist_ok=True, parents=True)
filename = results_folder / "fundamentals"
with io.VTXWriter(domain.comm, filename.with_suffix(".bp"), [uh]) as vtx:
    vtx.write(0.0)
with io.XDMFFile(domain.comm, filename.with_suffix(".xdmf"), "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

```


### 3. Excute with MPI

```
mpirun --oversubscribe -n 32 python poisson.py
```


```
the number of MPI ranks: 32
computation time = 9.24938178062439 sec
Error_L2 : 5.03e-07
Error_max : 6.44e-11
```

```
the number of MPI ranks: 1
computation time = 21.489859342575073 sec
Error_L2 : 5.03e-07
Error_max : 6.11e-11
```
