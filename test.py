#!/usr/bin/python3
# -*- coding: utf-8 -*-
import random
import time
import string
import sys
import scrypt
import functools, operator
from Library import opencl
from Library.opencl_information import opencl_information
from binascii import unhexlify

# ===========================================================================================

def random_string_generator(amount):
    for i in range(amount):
        yield str.encode(''.join(random.choices(string.ascii_lowercase + string.ascii_uppercase + string.digits + string.punctuation, k=random.randint(10,60))))

# ===================================== Test funcs =============================================

def scrypt_test(scrypt_opencl_algos, passwords, N_value=15, r_value=3, p_value=1, desired_key_length=32,
                hex_salt=unhexlify("DEADBEEFDEADBEEFDEADBEEFDEADBEEF")):
    print("Testing scrypt")
    correct_res = []

    t0 = time.time()
    for pwd in passwords:
        v = scrypt.hash(pwd, hex_salt, 1 << N_value, 1 << r_value, 1 << p_value, desired_key_length)
        correct_res.append(v)
    t1 = time.time()
    print("CPU scrypt finished in: %.6f" % (t1-t0))

    t0 = time.time()
    ctx = scrypt_opencl_algos.cl_scrypt_init(N_value)
    clResult = scrypt_opencl_algos.cl_scrypt(ctx,passwords,N_value,r_value,p_value,desired_key_length,hex_salt)
    t1 = time.time()
    print("OpenCL scrypt finished in: %.6f" % (t1-t0))

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
    info = opencl_information()
    if (len(argv)<4):
        print("Implementation tests")
        print("-----------------------------------------------------------------")
        info.printplatforms()
        print("\nPlease run as: python test.py [platform number] [number of passwords] [number of salts]")
        return

    info. printfullinfo()

    passwords_count = int(argv[2])
    salts_count = int(argv[3])

    # Input values to be hashed
    t0 = time.time()
    passwordlist = list(random_string_generator(passwords_count))
    salts = list(random_string_generator(salts_count))
    t1 = time.time()
    print("%d random passwords and salts were generated in: %.6f\n" % ((passwords_count + salts_count), (t1-t0)))

    platform = int(argv[1])
    debug = 0
    write_combined_file = False
    opencl_algos = opencl.opencl_algos(platform, debug, write_combined_file,inv_memory_density=1)
    # Call the tests

    for salt in salts:
        print("Using salt: %s" % salt)
        scrypt_test(opencl_algos, passwordlist, 14, 3, 3, 0x20, salt)
        print()

    print("Tests have finished.")

if __name__ == '__main__':
  main(sys.argv)
