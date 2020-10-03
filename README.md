Opencl Brute Benchmarking
=========================

Benchmarking of the OpenCL implementation of the scrypt algorithm from the
repository [opencl_brute](https://github.com/bkerler/opencl_brute/)
against the python [scrypt](https://pypi.org/project/scrypt/).

How to Run
==========
```
python test.py [platform number] [number of passwords] [number of salts]
```

Example
=======
```
$ python test.py 0 4 2
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
