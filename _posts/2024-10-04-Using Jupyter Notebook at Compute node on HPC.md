---
layout: post
title: Using Jupyter notebook at compute node on HPC
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/Jupyternotebook.png
share-img: /assets/img/Jupyternotebook.png
tags: [HPC]
---

_The thumbnail image is created by chatGPT-4o_
###### This content explains how to use Jupyter notebook at a compute node on HPC environment.
<br/>

Contents
1. Basic settings
2. Pytorch & torch-related libraries
3. Install Fenics
4. Enjoy



### 1. Enabling ssh between Polaris Compute Nodes 
(<https://docs.alcf.anl.gov/polaris/known-issues/>)

1. cd /home/<username>/.ssh
2. ssh-keygen -t rsa
3. create authorized_keys file and copy id_rsa.pub into authorized_keys
4. Try setting up your .ssh/config file like this (swapping out the username placeholder with your ALCF username)

```
polaris-login-03:~> cat .ssh/config
Host *
User <YOUR_USERNAME>
Compression yes
Protocol 2
ControlMaster auto
ControlPath ~/.ssh/master-%r@%h:%p
ForwardAgent yes
StrictHostKeyChecking no
UserKnownHostsFile=/dev/null
```

<br/>


### 2. Your /home/<username> directory permissions should be set to 700 (chmod 700 /home/<username>)


### 3. Confirm the following files exist in your .ssh directory and the permissions are set to the following: 1. -rw------- (600) authorized_keys 2. -rw-r--r-- (644) config 3. -rw------- (600) id_rsa 4. -rw-r--r-- (644) id_rsa.pub

```
chmod 600 ~/.ssh/authorized_keys
chmod 644 ~/.ssh/config
chmod 600 ~/.ssh/id_rsa
chmod 644 ~/.ssh/id_rsa.pub
```


### 4. Allocate debug node

### 5. ssh -L 8888:localhost:8888 <your_username>@<compute node>

### 6. jupyter notebook --no-browser --port=8888

### 7. Open in browser
