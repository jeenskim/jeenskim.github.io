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


### 2. create anaconda environment and activate the environment

```
conda create --name jax python=3.9
conda activate jax
```

<br/>


### 3. install gpu-enabled jax 
(<https://docs.jax.dev/en/latest/installation.html>)

```
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

<br/>


### 3. test

```
python
import jax
jax.devices()
```

<br/>

