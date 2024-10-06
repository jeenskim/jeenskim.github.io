---
layout: post
title: Installing Fenics Software on HPC(Polaris)
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/HPC.png
share-img: /assets/img/HPC.png
tags: [HPC]
---

_The thumbnail image is created by chatGPT-4o_
###### This content explains how to install the Finite element method (FEM) based open-source software Fenics from source code on HPC environment. 
###### Especially, we are going to focus on how to install Fenics on Polaris.
<br/>

Contents
1. Basic settings
2. Pytorch & torch-related libraries
3. Install Fenics
4. Enjoy

   
<br>
### 1. Basic settings
#### 1.1. Proxy settings on Polaris
This enables for node to get access to external website (pip, wandb, etc)

```
# proxy settings
export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"
export no_proxy="admin,polaris-adminvm-01,localhost,*.cm.polaris.alcf.anl.gov,polaris-*,*.polaris.alcf.anl.gov,*.alcf.anl.gov"
```
<br/>
#### 1.2. Module setup
This initialize loaded modules and load some necessary modules

```
module restore
module use /soft/modulefiles/
module load jax/0.4.29-dev
module load cmake
export MPICH_GPU_SUPPORT_ENABLED=0
```
<br/>
### 2. Pytorch & torch-related libraries
#### 2.1. Install PyTorch, other torch-related libraries, torch_geometric
Install torch, torchvision and torch audio

```
pip install --user --force-reinstall --no-cache-dir torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

Install pyg_lib, torch_scatter, torch_sparse, torch_cluster, torch_spline_conv
```
pip install --user --force-reinstall --no-cache-dir pyg_lib==0.4.0+pt112cu113 torch_scatter==2.1.0+pt112cu113 torch_sparse==0.6.16+pt112cu113 torch_cluster==1.6.0+pt112cu113 torch_spline_conv==1.2.1+pt112cu113 -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
```

Install torch_geometric
```
pip install --user torch_geometric
```
<br/>
#### 2.2. Install functorch(https://pytorch.org/functorch/versions.html)
This enables the use of additional torch function (e.g. functorch.vmap)
```
pip install --user functorch==0.2.1 
```
<br/>

#### 2.3. Change in train.py (Ignore this if already changed)
torch.concatenate -> torch.cat

torch.vmap -> functorch.vmap
<br/>

#### 2.4. Install mpi4torch
This enables MPI calculation for torch.tensor
```
pip install --user mpi4torch --no-build-isolation
```
<br/>

### 3. Install Fenics
- Installing Fenics is composed of three big steps
    * Install necessary and optional packages required by Fenics
    * Compilation of dolfin source code with those pre-installed packages
    * Installation of dolfin packages
<br/>
- Pre-installed necessary and optional packages required by Fenics
    * Necessary packages
        * Pybind11
        * Eigen3
        * Boost
    * Optional packages
        * PETSc
        * SLEPc
        * SCOTCH
        * PARMETIS
        * MPI
        * Zlib
        * HDF5
        * UMFPACK
        * CHOLMOD
        * BLAS
<br/>
#### 3.0. Create a directory (let’s call this fenics_legacy) where the installation-from-source will take place, and go into that directory
```
mkdir [your_path]/fenics_legacy
cd [your_path]/fenics_legacy
```
<br/>
#### 3.1. Install Pybind11 (in [your_path]/fenics_legacy)
Downloading .tar file of Pybind and untar

```
PYBIND11_VERSION=2.2.4
wget -nc --quiet https://github.com/pybind/pybind11/archive/v${PYBIND11_VERSION}.tar.gz
tar -xf v${PYBIND11_VERSION}.tar.gz
```

Make directories for build and install
```
cd pybind11-${PYBIND11_VERSION}
mkdir build install_dir
```
Compile source file with cmake
```
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install_dir/ -DPYBIND11_TEST=off
```
Install
```
make install
```
<br/>

#### 3.2. Install Eigen3

Downloading git folder
```
cd [your_path]/fenics_legacy
git clone --recursive https://gitlab.com/libeigen/eigen.git
```

Get Eigen3==3.3.0 version and make directories for build and install
```
cd eigen
git checkout tags/3.3.0
mkdir build install_dir
```
Compile source code with cmake
```
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install_dir/
```
Install
```
make install
```
<br/>

#### 3.3. Install Boost
First, download Boost 1.70.0 from here: https://archives.boost.io/release/1.70.0/source/boost_1_70_0.tar.gz, and place it in [your_path]/fenics_legacy

Untar istalled .tar file
```
cd [your_path]/fenics_legacy
tar -xvf boost_1_70_0.tar.gz
```

Make directory for install
```
cd boost_1_70_0
mkdir install_dir
```

Compile source code with bash(bootstrap.sh) file
```
./bootstrap.sh
```

Install
```
./b2 install --prefix=install_dir/
```

<br/>


#### 3.4. Install PETSc (https://petsc.org/release/install/install_tutorial/, https://petsc.org/release/manualpages/Mat/MATSOLVERMUMPS/, https://github.com/PrincetonUniversity/EDIPIC-2D/blob/main/Instructions/installing_PETSc.md)
We can install some of other optional packages(SCOTCH, PARMETIS) together
Downloading git folder
```
git clone -b release https://gitlab.com/petsc/petsc
```

Configure source code with ./configure and other package options
```
cd petsc
./configure --with-single-library=1 --download-fblaslapack --download-mumps --download-scalapack --download-parmetis --download-metis --download-ptscotch --download-hypre


./configure --with-single-library=1 --download-openblas --download-mumps --download-scalapack --download-parmetis --download-metis --download-ptscotch --download-hypre --download-suitesparse --download-fftw --download-superlu --download-superlu_dist


./configure --with-single-library=1 --with-blas-lib=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/compilers/lib/libblas.a --with-lapack-lib=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/compilers/lib/liblapack.a --download-mumps --download-scalapack --download-parmetis --download-metis --download-ptscotch --download-hypre --download-suitesparse --download-fftw --download-superlu --download-superlu_dist
```

Install and check
```
make PETSC_DIR=[your_path]/fenics_legacy/petsc PETSC_ARCH=arch-linux-c-debug all check
```

Make linkage of ‘libpetsc.so.3.21.5’ with ‘libpetsc.so.3.17’
```
cd [your_path]/fenics_legacy/petsc/arch-linux-c-debug/lib
 ln -s libpetsc.so.3.21.5 libpetsc.so.3.17
```


Install python wrapper for petsc using pip
```
cd [your_path]/fenics_legacy/petsc/src/binding/petsc4py
export PETSC_DIR=/grand/NeuralDE/hjkim/fenics_legacy/petsc-3.17.3
export PETSC_ARCH=arch-linux-c-debug
pip install --user .
```


<br/>


#### 3.5. Install SLEPc (https://slepc.upv.es/download/) (https://slepc.upv.es/documentation/slepc.pdf)
Downloading git folder
```
git clone -b release https://gitlab.com/slepc/slepc
```

Configure source code with ./configure
```
cd slepc export PETSC_DIR=[your_path]/fenics_legacy/petsc
export PETSC_ARCH=arch-linux-c-debug
./configure
```
Install and check
```
make 
make check
```

Make linkage of ‘libslepc.so.3.21.5’ with ‘libslepc.so.3.17’
```
cd [your_path]/fenics_legacy/slepc/arch-linux-c-debug/lib
ln -s libslepc.so.3.21.1 libslepc.so.3.17
```

Install python wrapper for slepc using pip
```
cd [your_path]/fenics_legacy/petsc/src/binding/petsc4py
export SLEPC_DIR=/grand/NeuralDE/hjkim/fenics_legacy/slepc-3.17.3
export PETSC_ARCH=arch-linux-c-debug
pip install --user .
```

<br/>


#### 3.6. Building and Installing dolfin

Downloading git folder
```
cd [your_path]/fenics_legacy
FENICS_VERSION=$(python3 -c"import ffc; print(ffc.__version__)")
git clone --branch=$FENICS_VERSION https://bitbucket.org/fenics-project/dolfin
```

Make build and install directories
```
cd dolfin
mkdir build install_dir
cd build
```

Set environment variables for pre-installed packages
```
export PETSC_DIR=[your_path]/fenics_legacy/petsc
export PETSC_ARCH=arch-linux-c-debug

export SLEPC_DIR=/grand/NeuralDE/hjkim/fenics_legacy/slepc
export SLEPC_ARCH=arch-linux-c-debug


export SCOTCH_DIR=/grand/NeuralDE/hjkim/fenics_legacy/petsc/arch-linux-c-debug
export SCOTCH_LIBRARIES=/grand/NeuralDE/hjkim/fenics_legacy/petsc/arch-linux-c-debug/lib
export SCOTCH_LIBRARIES=/grand/NeuralDE/hjkim/fenics_legacy/petsc/arch-linux-c-debug/include

export PARMETIS_DIR=//grand/NeuralDE/hjkim/fenics_legacy/petsc/arch-linux-c-debug
export PARMETIS_LIBRARIES=/grand/NeuralDE/hjkim/fenics_legacy/petsc/arch-linux-c-debug/lib
export PARMETIS_INCLUDE_DIRS=/grand/NeuralDE/hjkim/fenics_legacy/petsc/arch-linux-c-debug/include



export PETSC_DIR=/grand/NeuralDE/hjkim/fenics_legacy/petsc_new
export PETSC_ARCH=arch-linux-c-debug

export SLEPC_DIR=/grand/NeuralDE/hjkim/fenics_legacy/slepc_new
export SLEPC_ARCH=arch-linux-c-debug


export SCOTCH_DIR=/grand/NeuralDE/hjkim/fenics_legacy/petsc_new/arch-linux-c-debug
export SCOTCH_LIBRARIES=/grand/NeuralDE/hjkim/fenics_legacy/petsc_new/arch-linux-c-debug/lib
export SCOTCH_LIBRARIES=/grand/NeuralDE/hjkim/fenics_legacy/petsc_new/arch-linux-c-debug/include

export PARMETIS_DIR=//grand/NeuralDE/hjkim/fenics_legacy/petsc_new/arch-linux-c-debug
export PARMETIS_LIBRARIES=/grand/NeuralDE/hjkim/fenics_legacy/petsc_new/arch-linux-c-debug/lib
export PARMETIS_INCLUDE_DIRS=/grand/NeuralDE/hjkim/fenics_legacy/petsc_new/arch-linux-c-debug/include


export UMFPACK_DIR=/grand/NeuralDE/hjkim/fenics_legacy/petsc_new/arch-linux-c-debug
export UMFPACK_LIBRARIES=/grand/NeuralDE/hjkim/fenics_legacy/petsc_new/arch-linux-c-debug/lib
export UMFPACK_INCLUDE_DIR=/grand/NeuralDE/hjkim/fenics_legacy/petsc_new/arch-linux-c-debug/include/suitesparse

export CHOLMOD_DIR=/grand/NeuralDE/hjkim/fenics_legacy/petsc_new/arch-linux-c-debug
export CHOLMOD_LIBRARIES=/grand/NeuralDE/hjkim/fenics_legacy/petsc_new/arch-linux-c-debug/lib
export CHOLMOD_INCLUDE_DIR=/grand/NeuralDE/hjkim/fenics_legacy/petsc_new/arch-linux-c-debug/include/suitesparse



export BLAS_DIR=/grand/NeuralDE/hjkim/fenics_legacy/petsc_new/arch-linux-c-debug
export BLAS_LIBRARIES=/grand/NeuralDE/hjkim/fenics_legacy/petsc_new/arch-linux-c-debug/lib
export BLAS_INCLUDE_DIR=/grand/NeuralDE/hjkim/fenics_legacy/petsc_new/arch-linux-c-debug/include

```

Load HDF5 module required to compile
```
module load cray-hdf5-parallel/1.12.2.9
```

Compile source code with cmake
```
cmake .. -DBOOST_ROOT=/grand/NeuralDE/hjkim/fenics_legacy/boost_1_70_0/install_dir -DEIGEN3_INCLUDE_DIR=/grand/NeuralDE/hjkim/fenics_legacy/eigen/install_dir/include/eigen3 -DCMAKE_INSTALL_PREFIX=../install_dir/ -DHDF5_ROOT=/opt/cray/pe/hdf5-parallel/1.12.2.9 

cmake .. -DBOOST_ROOT=/grand/NeuralDE/hjkim/fenics_legacy/boost_1_70_0/install_dir -DEIGEN3_INCLUDE_DIR=/grand/NeuralDE/hjkim/fenics_legacy/eigen/install_dir/include/eigen3 -DCMAKE_INSTALL_PREFIX=../install_dir_new/ -DHDF5_ROOT=/opt/cray/pe/hdf5-parallel/1.12.2.9
```

Install
```
make install
```


(If this error occurs while installing)
Error message : _[your_path]/fenics_legacy/dolfin/dolfin/io/HDF5Interface.cpp:285:22: error: too few arguments to function ‘herr_t H5Oget_info_by_name3(hid_t, const char*, H5O_info2_t*, unsigned int, hid_t)’_


`vi fenics_legacy/dolfin/dolfin/io/HDF5Interface.cpp  @ line 285 `  
Replace //H5Oget_info_by_name(hdf5_file_handle, group_name.c_str(), &object_info,
//                    lapl_id);
With _H5Oget_info_by_name3(hdf5_file_handle, group_name.c_str(), &object_info, H5O_INFO_ALL, H5P_DEFAULT);_

<br/>


#### 3.7. Install python wrapper for dolfin
```
source [your_path]/fenics_legacy/dolfin/install_dir/share/dolfin/dolfin.conf 
export pybind11_DIR=[your_path]/fenics_legacy/pybind11-2.2.4/install_dir/share/cmake/pybind11

export pybind11_DIR=/grand/NeuralDE/hjkim/fenics_legacy/pybind11-2.2.4/install_dir/share/cmake/pybind11
```

Install python wrapper for dolfin using pip
```
cd [your_path]/fenics_legacy/dolfin/python
pip install --user .
```

### 4. Enjoy

Whenever restarting your session,
```
module restore
module use /soft/modulefiles/
module load jax/0.4.29-dev
module load cmake
source [your_path]/fenics_legacy/dolfin/install_dir/share/dolfin/dolfin.conf
```

Or when submitting job using bash script,
```
# proxy settings
export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"
export no_proxy="admin,polaris-adminvm-01,localhost,*.cm.polaris.alcf.anl.gov,polaris-*,*.polaris.alcf.anl.gov,*.alcf.anl.gov"

module restore
module use /soft/modulefiles/
module load jax/0.4.29-dev
export MPICH_GPU_SUPPORT_ENABLED=0
source /grand/NeuralDE/hjkim/fenics_legacy/dolfin/install_dir/share/dolfin/dolfin.conf

source /grand/NeuralDE/hjkim/fenics_legacy/dolfin/install_dir_new/share/dolfin/dolfin.conf
```


&nbsp; GNNs use the graph structure and node features $X_v$ to learn a representation vector of a node, $h_v$, or the entire graph, $h_G$. Modern GNNs follow a neighborhood aggregation strategy, where we iteratively update the representation of a node by aggregating representations of its neighbors. 
After $k$ iterations of aggregation, a node's representation captures the structural information within its k-hop network neighborhood.
Formally, the $k$-th layer of a GNN is as follows:  <br/><br/>
$a_v^{(k)} = AGGREGATE^{(k)}({h_u^{(k-1)} : u \in N(v)})$,   $h_v^{(k)} = COMBINE^{(k)}(h_v^{(k-1)}, a_v^{(k)}),$ <br/><br/>
where $h_v^{(k)}$ is the feature vector of node $v$ at the $k$-th iteration/layer.
We initialize $h_v^{(0)}=X_v,$ and $N(v)$ is a set of nodes adjacent to $v$. 
The choice of $AGGREGATE^{(k)}$ and $COMBINE^{(k)}$ in GNN is crucial.
A number of architectures for $AGGREGATE$ and $COMBINE$ have been proposed.  
GNNs are capable of combining the scalability of backpropagation-based optimization with flexible data representations.
Due to their generalizability and ambiguity in graph representations of data, they have increasing usage in most modeling tasks.
As long as a problem or dataset can be described as a set of nodes and edges, GNNs show their overall strength in modeling problems. <br/><br/>


#### 1.2. Applications
&nbsp; Image classification, object detection, and node classification tasks. Additionally, modeling protein folding, predicting the emergent properties of social networks and modeling differential equations.

   

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
 we don't need to compute $\frac{dx}{dm}$. <br/><br/>
 Then,
 $$
 \frac{df}{dm} = \lambda^T\frac{\partial g}{\partial m}
 $$
<br/><br/>
,which can be calculated using $\lambda$ from the above equation. <br>

In summary, by utilizing Lagrangian and setting the adjoint variable properly, we can compute a gradient of the objective function with regard to parameters.

### 2. GNN for CFD applications
#### 2.1. Purpose
&nbsp; Turbulence modeling with machine learning techniques usually has taken _a-priori_ learning. While _a-priori_ approach in modeling turbulence is easy to implement, it can't guarantee accurate prediction for unseen data. This is because when a learned model is embedded into the PDE solver, numerical artifacts, such as numerical errors and temporal errors, are produced by the PDE solver. These artifacts are not included in the training process, thus deteriorating prediction capability for unseen data.

&nbsp; However, differentiable PDE solver enables end-to-end training of machine learning models with backpropagation, enhancing _a-posteriori_ prediction capability. A differentiable PDE solver can be realized by implementing a governing equation in an automatic differentiation (AD) framework, which is a necessary tool for training deep neural networks. <br>

#### 2.2. Examples
&nbsp; The AD framework's basic principle is that gradients of a composition of differentiable functions can be exactly computed using the chain rule. Deep neural networks are composed of straightforward operations, such as linear or pointwise nonlinear operations. Therefore, gradients can be easily computed in the training of deep neural networks using the AD framework. However, writing a PDE solver in an AD-enabled way can add enormous complexity, which makes it impractical. Instead, by implementing a vector-jacobian product using the discrete adjoint method, we can integrate a PDE solver within the AD framework.

&nbsp; The basic chain rule is as follows. <br>
$x_1=f_0(x_0)$ <br>
$x_2=f_0(x_1)$ <br>
.<br>
.<br>
.<br>
$y=f_n(x_n)$ <br/><br/>
$\frac{dy}{dx_i}=\frac{dy}{dx_{i+1}}\frac{dx_{i+1}}{dx_i}=\frac{dy}{dx_{i+1}}\frac{df_i(x_i)}{dx_i}$.  <br/><br/>

&nbsp;The iterative nature of the chain rule allows gradients of $y$ to be propagated backward starting from $\frac{dy}{dx_n}$. To enable backpropagation, the reverse function should be implemented in each forward operation. This reverse function is called a vector-jacobian product (VJP): <br>

$\frac{dy}{dx_i}={f_i}^{\prime}(\frac{dy}{dx_{i+1}},x_i).$ <br>

&nbsp; In the AD framework, VJPs are provided easily for the majority of functions, e.g., addition and multiplication. However, if the function $f_i$ represents an external routine not known by the AD framework, such as a PDE solver, a custom VJP should be implemented. <br>

&nbsp; Let's take an example of solving the Navier-Stokes equation using an implicit pressure correction scheme (IPCS). The overall flow of this algorithm is as follows: <br>


![algorithm 1](/assets/img/IPCS_Algorithm1.png)

&nbsp;In this algorithm, the GNN model is included to predict turbulence model parameters. And since the GNN model is written in the AD framework, no further work is required to enable gradient backpropagation. However, for the next three functions, *tentative_vel, pres_correct, vel_correct*, which are external routines and not known by the AD framework, a custom VJP for each of the PDE solution steps should be implemented.<br>

&nbsp; The PDE solver can be written generally as follows: <br>
$x=$*PDE_solve*$(m),$ <br>
where x is state variables, such as pressure and velocity, and m is learnable model parameters. <br/><br/>
Here we should implement a custom VJP:<br>
$y_m=$*PDE_solve_vjp*$(y_x,m),$ <br>
where subscript represents partial differentiation. <br/><br/>
The forward PDE-solving operations can be simplified into solving the linear system: <br>
$A(m)x=b(m),$ <br>
where A and b represent the discretized left hand side (LHS) and right hand side (RHS) of the PDE respectively. <br/><br/>
With this description, we can implement *PDE_solve_vjp* using the discrete adjoint method. Using the discrete adjoint method, the total derivative of y can be calculated as follows:<br>
$\frac{dy}{dm}=-\lambda^T(A_mx-b_m),$<br>
where $\lambda$ is the solution to the adjoint equation given by $A^T\lambda=y_x^T.$ <br/><br/>
The overall procedure for solving PDE and calculating custom VJP is shown below: <br>
![algorithm 1](/assets/img/Algorithm2.png)



[^1]: [Barwey, Shivam, et al. "Multiscale graph neural network autoencoders for interpretable scientific machine learning." Journal of Computational Physics 495 (2023): 112537.](https://doi.org/10.1016/j.jcp.2023.112537) 
[^2]: [Xu, Keyulu, et al. "How powerful are graph neural networks?." arXiv preprint arXiv:1810.00826 (2018).](https://doi.org/10.48550/arXiv.1810.00826)
