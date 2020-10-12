#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
    Original copyright:
    Copyright by B.Kerler 2017, PBKDF1_SHA1 and SHA256 PyOpenCl implementation, max 32 chars for password + salt
    MIT License
    Implementation was confirmed to work with Intel OpenCL on Intel(R) HD Graphics 520 and Intel(R) Core(TM) i5-6200U CPU
'''
'''
    Adapted for generalising to more hash functions
    Allows any length input (efficiently, by declaring the max in advance)
     - salt ditched atm, but hoping to restore it
     - pbkdf2 forgotten about for now
'''

from Library.buffer_structs import buffer_structs
import pyopencl as cl
import numpy as np
from binascii import unhexlify, hexlify
import os
from itertools import chain, repeat, zip_longest
from collections import deque
from hashlib import pbkdf2_hmac

# Corresponding to opencl (CAN'T BE CHANGED):
r = 8
BLOCK_LEN_BYTES = 128 * r

# Little helper, (22,5) -> 5,5,5,5,2.  itertools is bae
def takeInChunks(n,d):
    assert d > 0 and n >= 0
    return chain(repeat(d, n // d), filter(lambda x:x!=0, [n % d]))

def printif(b, s):
    if b:
        print(s)

class opencl_interface:
    debug=False
    inv_memory_density=1
    # Initialiser for the key properties
    #   pbkdf related initialisation removed, will reappear somewhere else
    def __init__(self, platformNum, debug=0, write_combined_file=False, maxWorkgroupSize=60000, inv_memory_density=1, N_value=15, openclDevice = 0):
        printif(debug,"Using Platform %d:" % platformNum)
        devices = cl.get_platforms()[platformNum].get_devices()
        self.platform_number=platformNum
        # Show devices for the platform, and adjust workgroup size
        # Create the context for GPU/CPU
        self.workgroupsize = maxWorkgroupSize
        # Adjust workgroup size so that we don't run out of RAM:
        # As with bench_sCrypt.py, not really working!
        self.sworkgroupsize = self.determineWorkgroupsize(N_value)
        self.inv_memory_density=inv_memory_density
        self.ctx = cl.Context(devices)
        self.queue = cl.CommandQueue(self.ctx, devices[openclDevice])
        self.debug=debug

        for device in devices:
            printif(debug, '--------------------------------------------------------------------------')
            printif(debug, ' Device - Name: '+ device.name)
            printif(debug, ' Device - Type: '+ cl.device_type.to_string(device.type))
            printif(debug, ' Device - Compute Units: {0}'.format(device.max_compute_units))
            printif(debug, ' Device - Max Work Group Size: {0:.0f}'.format(device.max_work_group_size))
            printif(debug, ' Device - Global memory size: {}'.format(device.global_mem_size))
            printif(debug, ' Device - Local memory size:  {}'.format(device.local_mem_size))
            printif(debug, ' Device - Max clock frequency: {} MHz'.format(device.max_clock_frequency))

            assert device.endian_little == 1, "DEVICE is not little endian : pretty sure we rely on this!"
            if (device.max_work_group_size<self.workgroupsize):
                self.workgroupsize=device.max_work_group_size
        printif(debug, "\nUsing work group size of %d\n" % self.workgroupsize)

        # Set the debug flags
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = str(debug)
        self.write_combined_file = write_combined_file

    def compile(self, bufferStructsObj, library_file, footer_file=None, N=15, invMemoryDensity=2):
        assert type(N) == int
        assert N < 20, "N >= 20 won't fit in a single buffer, so is unsupported. Nothing sane should use 20, is this wickr?"
        self.N = N
        assert bufferStructsObj is not None, "need to supply a bufferStructsObj : set all to 0 if necessary"
        assert bufferStructsObj.code is not None, "bufferStructsObj should be initialised"
        self.bufStructs = bufferStructsObj
        self.wordSize = self.bufStructs.wordSize

        # set the np word type, for use in .run
        npType = {
            4 : np.uint32,
            8 : np.uint64,
        }
        self.wordType = npType[self.wordSize]

        src = self.bufStructs.code
        if library_file:
            with open(os.path.join("Library","worker","generic",library_file), "r") as rf:
                src += rf.read()

        if footer_file:
            with open(os.path.join("Library","worker","generic",footer_file), "r") as rf:
                src += rf.read()

        # Standardise to using no \r's, move to bytes to stop trickery
        src = src.encode("ascii")
        src = src.replace(b"\r\n",b"\n")

        # Debugging
        if self.write_combined_file:
            with open("combined_" + library_file, "wb") as wf:
                wf.write(src)

        # Convert back to text!
        src = src.decode("ascii")

        # Check that it starts with 2 newlines, for adding our defines
        if src.startswith("\n\n"):
            src = "\n\n" + src
            src = src[len("\n\n"):]
            # Prepend define N and invMemoryDensity
            defines = "#define N {}\n#define invMemoryDensity {}\n".format(N, invMemoryDensity)
            src = defines + src

        # Kernel function instantiation. Build returns self.
        self.prg = cl.Program(self.ctx, src).build()
        return self.prg

    # Forms the input buffer of derived keys
    # Returns the buffer and number in the buffer, <= n (iter may be exhausted)
    def makeInputBuffer(self, dkIter, n):
        inpArray = bytearray()
        numEaten = n

        for i in range(n):
            try:
                dk = dkIter.__next__()
            except StopIteration:
                # Correct the chunk size and break
                numEaten = i
                break

            assert len(dk) == BLOCK_LEN_BYTES
            #   , "Derived key input is length {}, when we expected {}".format(len(dk), BLOCK_LEN_BYTES)

            inpArray.extend(dk)

        # pyopencl doesn't like empty buffers, so just cheer it up
        #   (making the buffer larger isn't an issue)
        if len(inpArray) == 0:
            inpArray = b"\x00"

        inp_g = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=inpArray)

        return inp_g, numEaten

    def determineWorkgroupsize(self, N_value=15):
        devices = cl.get_platforms()[self.platform_number].get_devices()
        wgSize = 0
        for device in devices:
            ## Actually adjust based on invMemoryDensity!
            N_blocks_bytes = (1 << N_value) * BLOCK_LEN_BYTES // self.inv_memory_density
            memoryForOneCore = BLOCK_LEN_BYTES * 2 + N_blocks_bytes  # input, output & V

            ## ! Restrict to half the memory for now
            coresOnDevice = (int(0.5 * device.global_mem_size) // memoryForOneCore)
            percentUsage = 100 * memoryForOneCore * coresOnDevice / device.global_mem_size
            percentUsage = str(percentUsage)[:4]
            if self.debug == 1:
                print("Using {} cores on device with global memory {}, = {}%".format(
                    coresOnDevice, device.global_mem_size, percentUsage
                ))
            wgSize += coresOnDevice

        if self.debug == 1:
            print("Workgroup size determined as {}".format(wgSize))

        return wgSize

    def run_scrypt(self, sprg, kernelCall, dkIter):
        N_blocks_bytes = (1 << self.N) * BLOCK_LEN_BYTES

        # no. of cores' memory that we can fit into a single buffer
        #   (seemingly anyway, why isn't it 2^31?)
        # note: this is NOT the workgroupsize, nor does it bound it
        maxGangSize = (1 << 29) // N_blocks_bytes
        assert maxGangSize > 0, "Uh-oh we couldn't fit a single core's V in a buffer."


        #   A. Before the loop we produce our huge buffers, once only.
        #   B. Also make our output buffers & numpys now, just once, to save work
        #     Note these will be atleast big enough throughout the loop: sometimes they'll have extra room.
        largeBuffers = []
        outBuffers = []
        outNumpys = []
        outSizes = []
        for gangSize in takeInChunks(self.sworkgroupsize, maxGangSize):
            # Produce the large buffer for storing this gang's V arrays
            # No longer producing a big bytes object in Python

            ## arr = np.frombuffer(bytes(gangSize * N_blocks_bytes), dtype=np.uint32)
            ## Why is this read only?
            arr_g = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY, size = gangSize * N_blocks_bytes)
            largeBuffers.append(arr_g)

            # Produce the gang's output buffer and (small) numpy array to copy out to
            nBytes = BLOCK_LEN_BYTES * gangSize
            result = np.zeros(nBytes // 4, dtype=np.uint32)
            assert nBytes == result.nbytes
            result_g = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, nBytes)
            outBuffers.append(result_g)
            outNumpys.append(result)

            # No output from round 0!
            outSizes.append(0)



        # ! For minimal latency, we only block just before our next calls to the kernels:
        #     there is basically no work between the two.

        # Main loop is taking chunks of workgroup size,
        #   or less if we exhaust the input derived keys iter
        iterActive = True
        while iterActive:
            #   1. Make New SMALL input buffers (derived key buffer, was password & salt)
            #       if we exhaust dkIter, continue producing 'empty' input buffers, but mark to leave the main loop
            newInputs = []
            inCounts = []
            for gangSize in takeInChunks(self.sworkgroupsize, maxGangSize):
                input_g, numEaten = self.makeInputBuffer(dkIter, gangSize)
                iterActive = (numEaten == gangSize) # note gangSize > 0, so once False this will persist
                newInputs.append(input_g)
                inCounts.append(numEaten)


            #   2. (BLOCKING) wait for all our workers to finish (should be at similar times),
            #       and copy output buffers out to numpy (minimal time loss here, could use 2 sets of output buffers instead)
            #   Note we may well have copied too much: this is dealt with in 4. below
            for outSize, outNumpy, outBuf in zip_longest(outSizes, outNumpys, outBuffers):
                if outSize > 0:
                    cl.enqueue_copy(self.queue, outNumpy, outBuf)   # is_blocking defaults to true :)

            ##print("Calling kernels..")
            #   3. (NON-BLOCKING) queue the kernel calls
            for input_g, arr_g, result_g, inCount in zip_longest(newInputs, largeBuffers, outBuffers, inCounts):
                if inCount > 0:
                    dim = (inCount,)
                    # print("inCount = {}".format(inCount))
                    # print("All sizes in bytes (hopefully):")
                    # print("input_g.size = {}".format(input_g.size))
                    # print("arr_g.size = {}".format(arr_g.size))
                    # print("result_g.size = {}".format(result_g.size))
                    # print("\nOpenCL code now:\n")
                    kernelCall(sprg, (self.queue, dim, None, input_g, arr_g, result_g))
            ##print("Kernels running..")

            #   4. Process the outputs from the last round, yielding now (while the GPUs are busy)
            #       Also copy the input counts across to output sizes, for the next loop / final processing below
            for i, (outNumpy, inCount) in enumerate(zip_longest(outNumpys, inCounts)):
                outSize = outSizes[i]

                assert outSize % BLOCK_LEN_BYTES == 0
                outBytes = outNumpy.tobytes()
                for j in range(0, outSize, BLOCK_LEN_BYTES):
                    yield outBytes[j:j+BLOCK_LEN_BYTES]

                outSizes[i] = inCount * BLOCK_LEN_BYTES

            # Note that if exiting here then we've updated the outSizes & called the functions
            # Just remains to capture & process the output..

        ##print("Dropped out of loop")
        # Do a final loop of processing output (3 & 2)
        for outBuf, outNumpy, outSize in zip_longest(outBuffers, outNumpys, outSizes):
            # (BLOCKING) Copy!
            cl.enqueue_copy(self.queue, outNumpy, outBuf)

            assert outSize % BLOCK_LEN_BYTES == 0
            outBytes = outNumpy.tobytes()
            for i in range(0, outSize, BLOCK_LEN_BYTES):
                yield outBytes[i:i+BLOCK_LEN_BYTES]


class opencl_algos:
    def __init__(self, platform, debug, write_combined_file, inv_memory_density=1, openclDevice = 0):
        if debug==False:
            debug=0
        self.opencl_ctx= opencl_interface(platform, debug, write_combined_file, openclDevice = openclDevice)
        self.platform_number=platform
        self.inv_memory_density=inv_memory_density

    def cl_scrypt_init(self, N_value=15):
        # Initialise the openCL context & compile, with both debugging settings off
        debug = 0
        bufStructs = buffer_structs()
        sprg=self.opencl_ctx.compile(bufStructs, "sCrypt.cl", None, N=N_value, invMemoryDensity=self.inv_memory_density)
        return [sprg,bufStructs]

    def cl_scrypt(self, ctx, passwords, N_value=15, r_value=3, p_value=1, desired_key_length=32,
                    hex_salt=unhexlify("DEADBEEFDEADBEEFDEADBEEFDEADBEEF")):

        def getDkIter(p, salt, pwdIter, rtnPwds=None):
            # r fixed as 8 in the OpenCL
            r = 8
            blockSize = 128 * r
            for pwd in pwdIter:
                if rtnPwds is not None:
                    rtnPwds.append(pwd)
                # Get derived key, then split into p chunks and yield
                dk = pbkdf2_hmac("sha256", pwd, salt, 1, dklen=blockSize * p)

                # Yield
                for i in range(p):
                    yield dk[i * blockSize: (i + 1) * blockSize]

        sprg=ctx[0]

        # Our callback with the kernel name
        # Debugging: calls Salsa20
        def kernelCall(sprg, params):
            return sprg.ROMix(*params)  # prg.ROMix(*params)

        # Derived key iter: yields p keys for each password
        passwordList = deque()
        dkIter = getDkIter(1 << p_value, hex_salt, passwords, passwordList)

        result = []
        # Main call.
        group = []
        for singleOutput in self.opencl_ctx.run_scrypt(sprg, kernelCall, dkIter):
            group.append(singleOutput)
            if len(group) == 1 << p_value:
                expensiveSalt = b"".join(group)
                commonPwd = passwordList.popleft()
                sCryptResult = pbkdf2_hmac("sha256", commonPwd, expensiveSalt, 1, desired_key_length)

                # For now print out for debugging
                # print("Password={}".format(commonPwd))
                # print("sCrypt={}".format(hexlify(sCryptResult).decode().upper()))
                #result.append("{}".format(hexlify(sCryptResult).decode().lower()))
                result.append(sCryptResult)
                group = []
        return result
