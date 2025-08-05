---
layout: post
title: Installing dolfinx_mpc for FeniCSx
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/Fenicsx-spack.png
share-img: /assets/img/Fenicsx-spack.png
tags: [HPC, GPU, FeniCSx]
---

_The thumbnail image is created by chatGPT-4o_
###### This content explains how to to install dolfinx_mpc library for FeniCSx to implement multipoint boundary conditions such as slip conditions or periodic conditions and is based on (<https://github.com/jorgensd/dolfinx_mpc>)
<br/>

<br/>

### 1. Cloning git directory of dolfinx_mpc and matching a version with dolfinx

```
git clone https://github.com/jorgensd/dolfinx_mpc.git
cd dolfinx_mpc
git checkout v0.9.0
```

### 2. Build libraries using cmake

#### 2.1. Setting cmake, ninja, c and c++ compilers

Since packages for FeniCSx including dolfinx have been installed using spack, it is important to have the same versions of cmake, c and c++ compilers as used when installing FeniCSx.

After activating your spack environment for FeniCSx, by using `which cmake` command, you can check which version of cmake is loaded. If it does not match with the one in spack environment, you can use `spack load cmake` to activate the same version of cmake used for installing FeniCSx.

Also, if ninja is not installed, you can install ninja using `spack install ninja` and `spack load ninja`.

Then, you can build dolfinx_mpc libraries using the following command.
```
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=./install-dir \
  -B build-dir cpp/
```

And you can install dolfinx_mpc libraries using the following command.
```
ninja -j3 install -C build-dir
```

Lastly, you can install python bind for dolfinx_mpc using the following command.

```
python3 -m pip -v install --config-settings=cmake.build-type="Release" --config-settings=cmake.args="-DCMAKE_PREFIX_PATH=/lus/grand/projects/NeuralDE/hjkim/dolfinx_mpc/install-dir" --no-build-isolation ./python -U
```

During installing process, you might need to install other necessary libraries using the following command:

```
python3 -m pip install nanobind
python3 -m pip install scikit-build-core
```




