#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
import scrypt
import functools, operator
from Library import opencl
from Library.opencl_information import opencl_information
from binascii import unhexlify

# ===================================== Test funcs =============================================

def scrypt_test(scrypt_opencl_algos, passwords, N_value=15, r_value=3, p_value=1, desired_key_length=32,
                hex_salt=unhexlify("DEADBEEFDEADBEEFDEADBEEFDEADBEEF")):
    print("Testing scrypt")
    correct_res = []
    for pwd in passwords:
        v = scrypt.hash(pwd, hex_salt, 1 << N_value, 1 << r_value, 1 << p_value, desired_key_length)
        correct_res.append(v)
    ctx=scrypt_opencl_algos.cl_scrypt_init(N_value)
    clResult = scrypt_opencl_algos.cl_scrypt(ctx,passwords,N_value,r_value,p_value,desired_key_length,hex_salt)

    # Determine success and print
    correct = [r == c for r, c in zip(clResult, correct_res)]
    succ = (len(passwords) == len(clResult)) and functools.reduce(operator.and_, correct, True)
    if succ:
        print("Ok m11!")
    else:
        print("Failed !")
        for i in range(len(passwords)):
            if clResult[i] == correct_res[i]:
                print("#{} succeeded".format(i))
            else:
                print(i)
                print(clResult[i])
                print(correct_res[i])

# ===========================================================================================

def main(argv):
    if (len(argv)<2):
        print("Implementation tests")
        print("-----------------------------------------------------------------")
        info=opencl_information()
        info.printplatforms()
        print("\nPlease run as: python test.py [platform number]")
        return

    # Input values to be hashed
    passwordlist = [b'password', b'hmm', b'trolololl', b'madness']
    salts = [b"salty123",b"salty12"]

    platform = int(argv[1])
    debug = 0
    write_combined_file = False
    opencl_algos = opencl.opencl_algos(platform, debug, write_combined_file,inv_memory_density=1)
    # Call the tests

    for salt in salts:
        print("Using salt: %s" % salt)
        scrypt_test(opencl_algos,passwordlist,15,3,1,0x20,salt)

    print("Tests have finished.")

if __name__ == '__main__':
  main(sys.argv)
