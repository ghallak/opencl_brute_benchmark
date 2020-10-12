Opencl Brute Benchmarking
=========================

Benchmarking of the OpenCL implementation of the scrypt algorithm from the
repository [opencl_brute](https://github.com/bkerler/opencl_brute/)
against the python [scrypt](https://pypi.org/project/scrypt/).

Listing Available Platforms
===========================
Running the program without any arguments will show a list of available platforms

```
$ python test.py
Implementation tests
-----------------------------------------------------------------
Platform 0 - Name Portable Computing Language, Vendor The pocl project

Please run as: python test.py [platform number] [number of passwords] [number of salts]
```

How to Run
==========
```
$ python test.py [platform number] [number of passwords] [number of salts]
```

Examples
========
Running on Linux Mint 20 with Intel(R) Core(TM) i5-3210M CPU @ 2.50GHz:
```
$ python test.py 0 4 2

============================================================
OpenCL Platforms and Devices
============================================================
Platform 0 - Name: Portable Computing Language
Platform 0 - Vendor: The pocl project
Platform 0 - Version: OpenCL 1.2 pocl 1.4, None+Asserts, LLVM 9.0.1, RELOC, SLEEF, DISTRO, POCL_DEBUG
Platform 0 - Profile: FULL_PROFILE
 --------------------------------------------------------
 Device - Name: pthread-Intel(R) Core(TM) i5-3210M CPU @ 2.50GHz
 Device - Type: ALL | CPU
 Device - Max Clock Speed: 3100 Mhz
 Device - Compute Units: 4
 Device - Local Memory: 2048 KB
 Device - Constant Memory: 2048 KB
 Device - Global Memory: 6 GB
 Device - Max Buffer/Image Size: 2048 MB
 Device - Max Work Group Size: 4096

6 random passwords and salts were generated in: 0.000092

Using salt: b"&hP5.d*dDp'k9BIK0@CNm-,XI\\)a)pG{%I_Wl~G6e"
Testing scrypt
CPU scrypt finished in: 1.673907
OpenCL scrypt finished in: 1.727857
Ok m11!

Using salt: b'R*x&C,Nkc?,nuUcQqjCZabCIa%,=R;m$U{<5biIJ,S|Yy9CtyO!-F##W'
Testing scrypt
CPU scrypt finished in: 1.688711
OpenCL scrypt finished in: 1.722424
Ok m11!

Tests have finished.
```

Running on Windows 10 with AMD Ryzen 7 3700X and NVIDIA GeForce GTX 1080:
```
> python test.py 0 100 2

============================================================
OpenCL Platforms and Devices
============================================================
Platform 0 - Name: NVIDIA CUDA
Platform 0 - Vendor: NVIDIA Corporation
Platform 0 - Version: OpenCL 1.2 CUDA 11.1.70
Platform 0 - Profile: FULL_PROFILE
 --------------------------------------------------------
 Device - Name: GeForce GTX 1080
 Device - Type: ALL | GPU
 Device - Max Clock Speed: 1860 Mhz
 Device - Compute Units: 20
 Device - Local Memory: 48 KB
 Device - Constant Memory: 64 KB
 Device - Global Memory: 8 GB
 Device - Max Buffer/Image Size: 2048 MB
 Device - Max Work Group Size: 1024


102 random passwords and salts were generated in: 0.001000

Using salt: b'.]3x("mlV~_wRHcxU'
Testing scrypt
CPU scrypt finished in: 24.682958
OpenCL scrypt finished in: 51.689940
Ok m11!

Using salt: b'Le2!^1V@[DMwH4>]AQ7'
Testing scrypt
CPU scrypt finished in: 24.756741
OpenCL scrypt finished in: 41.269631
Ok m11!
```
