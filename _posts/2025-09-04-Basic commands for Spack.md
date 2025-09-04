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


### 2. `spack compilers`

Displays all compilers currently detected and registered by Spack.

Reads from `~/.spack/compilers.yaml` or environment-specific configs.


### 3. `spack compiler remove gcc@13.2.1`

Removes the registered `gcc@13.2.1` compiler from Spack’s config.

Does not uninstall `gcc` from the system — it only unregisters it from Spack.


### 4. `spack compiler add /opt/cray/pe/gcc-native/12/bin`

Registers a new compiler by scanning the given path for executables (`gcc`, `g++`, `gfortran`).

Commonly used in HPC systems for vendor-provided compilers (Cray, Intel, NVIDIA, AOCC, etc.).


### 5. `spack compiler info gcc@13.2.1`

Shows detailed information about the specified compiler.

Includes installation path, compiler executables, ABI support, environment variables, etc.

### 6. `spack compiler list`

Lists all compilers registered in Spack in a summarized format.

More concise than spack compilers.

### 7. `spack spec fenics-dolfinx`

Displays the dependency tree (specification) of the fenics-dolfinx package.

Shows compiler, package versions, build variants (+mpi, +cuda, etc.), and dependencies.

Very useful before installation to confirm the full spec.

### 8. `spack concretize`

Resolves and finalizes the package specification into a fully concrete form.

For example: `fenics-dolfinx` → `fenics-dolfinx@0.8.0 %gcc@12.3.0 ^petsc@3.20.0` ...

Ensures all versions and dependencies are pinned down before installation.

### 9. `spack concretize --fresh -f`

`--fresh`: Ignores cached concretizations and recomputes dependencies from scratch.

`-f` (force): Runs even if the environment is already concretized.

Useful after changing environment settings or resolving dependency conflicts.

### 10. `spack env st`

This is shorthand for `spack env status`.

Shows the currently active environment, or reports if no environment is active.

Useful for confirming whether you’re inside a particular Spack environment.

### 11. `spack env deactivate` (sometimes written as `spack deactivate`)

Deactivates the currently active Spack environment.

After running this, commands like spack install will no longer apply to that environment, but to the global Spack instance instead.

### 12. `spack find`

Lists all packages that are currently installed in your Spack installation.

By default, it shows unique specs (collapsed so that identical hashes/variants are grouped).

### 13. `spack find -c`

Stands for “concretized”.

Shows each package as it was fully concretized — including compiler, architecture, variants, and dependency hash.

Unlike plain spack find, this does not collapse identical specs, so you see the full details.




