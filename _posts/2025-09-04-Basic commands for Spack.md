---
layout: post
title: Basic commands for Spack
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/Fenicsx-spack.png
share-img: /assets/img/Fenicsx-spack.png
tags: [HPC, Spack]
---

_The thumbnail image is created by chatGPT-4o_
###### This content explains the basic commands for using Spakc in HPC environments.
<br/>

<br/>

### 1. `spack config blame packages`

Shows where package configuration entries come from (`packages.yaml`) across different scopes (system, site, user, environment).

Similar to `git blame` — helps track the origin of a particular package setting.


```
git clone https://github.com/jorgensd/dolfinx_mpc.git
cd dolfinx_mpc
git checkout v0.9.0
```

### 2. Build libraries using cmake

#### 2.1. Setting cmake, ninja, c and c++ compilers

Since packages for FeniCSx including dolfinx have been installed using spack, it is important to have the same versions of cmake, c & c++ compilers and other python-c binding packages as used when installing Fenicsx, which is called Application Binary Interface (ABI) compatibility.

After activating your spack environment for Fenicsx, by using `which cmake` command, you can check which version of cmake is loaded. If it does not match with the one in spack environment, you can use `spack load cmake` to activate the same version of cmake used for installing dolfinx.

Also, if ninja is not installed, you can install ninja using `spack install ninja` and `spack load ninja`.

You can check the version of c & c++ compilers used for installing dolfinx using `spack find`, and the current loaded compilers using `echo $CC` or `echo $CXX`. If they don't match, you can match them using the following command:

```
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
```

Also, if `nanobind` and `scikit-build-core` are not seen by `pip` (this happens because these libraries are installed as build-dependencies), you can load them using `spack load py-nanobind@2.5.0` and `spack load py-scikit-build-core@0.10.7`.

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




