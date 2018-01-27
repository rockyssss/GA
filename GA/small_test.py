# # # for j in range(2):
# # #     print('a:', j)
# # #     for i in range(3):
# # #         print(j, 'and:', i)
# # #! /usr/bin/env python
# # from __future__ import print_function
# #
# # from timeit import default_timer as time
# #
# # import numpy as np
# #
# # from numba import cuda
# #
# #
# # @cuda.jit('(f4[:], f4[:], f4[:])')
# # def cuda_sum(a, b, c):
# #     i = cuda.grid(1)
# #     c[i] = a[i] + b[i]
# #
# #
# # griddim = 50, 1
# # blockdim = 32, 1, 1
# # N = griddim[0] * blockdim[0]
# # print("N", N)
# # cuda_sum_configured = cuda_sum.configure(griddim, blockdim)
# # a = np.array(np.random.random(N), dtype=np.float32)
# # b = np.array(np.random.random(N), dtype=np.float32)
# # c = np.empty_like(a)
# #
# # ts = time()
# # cuda_sum_configured(a, b, c)
# # te = time()
# # print(te - ts)
# # assert (a + b == c).all()
# # #print c
# #! /usr/bin/env python
# # from __future__ import print_function
#
# from timeit import default_timer as time
#
# import numpy as np
#
# from numba import cuda
#
#
# bpg = 50
# tpb = 32
# n = bpg * tpb
#
#
# @cuda.jit('(float32[:,:], float32[:,:], float32[:,:])')
# def cu_square_matrix_mul(A, B, C):
#     tx = cuda.threadIdx.x
#     ty = cuda.threadIdx.y
#     bx = cuda.blockIdx.x
#     by = cuda.blockIdx.y
#     bw = cuda.blockDim.x
#     bh = cuda.blockDim.y
#
#     x = tx + bx * bw
#     y = ty + by * bh
#
#     if x >= n or y >= n:
#         return
#
#     C[y, x] = 0
#     for i in range(n):
#         C[y, x] += A[y, i] * B[i, x]
#
#
# A = np.array(np.random.random((n, n)), dtype=np.float32)
# B = np.array(np.random.random((n, n)), dtype=np.float32)
# C = np.empty_like(A)
#
# print("N = %d x %d" % (n, n))
#
# s = time()
# # for i in range(100):
# stream = cuda.stream()
# with stream.auto_synchronize():
#     dA = cuda.to_device(A, stream)
#     dB = cuda.to_device(B, stream)
#     dC = cuda.to_device(C, stream)
#     cu_square_matrix_mul[(bpg, bpg), (tpb, tpb), stream](dA, dB, dC)
#     dC.to_host(stream)
#
# e = time()
# tcuda = e - s
#
# # Host compute
# Amat = np.matrix(A)
# Bmat = np.matrix(B)
#
# s = time()
# # for i in range(100):
# Cans = Amat * Bmat
# e = time()
# tcpu = e - s
#
# # Check result
# assert np.allclose(C, Cans)
# #relerr = lambda got, gold: abs(got - gold)/gold
# #for y in range(n):
# #    for x in range(n):
# #        err = relerr(C[y, x], Cans[y, x])
# #        assert err < 1e-5, (x, y, err)
#
# print('cpu:  %f' % tcpu)
# print('cuda: %f' % tcuda)
# print('cuda speedup: %.2fx' % (tcpu / tcuda))
#

# x = 1111111111111111
# def f1():
#     x=1
#     y = 2
#     print('------>f1 ',x)
#     def f2():
#         x = 2
#         print('---->f2 ',x)
#         def f3():
#             x= 3
#             print('-->f3 ',x)
#         f3()
#     f2()
# f1()
import multiprocessing as mp
import shelve
import threading as td
import time


# def multi_run_wrapper(args):
#    return add(*args)
#
#
# def add(x,y):
#     return x+y
#
#
# if __name__ == "__main__":
#     from multiprocessing import Pool
#     pool = Pool(4)
#     results = pool.map(multi_run_wrapper,[(1,2),(2,3),(3,4)])
#     print(results)
import os
save_path2 = os.getcwd() + '\\res\\' + str(4712) + '.txt'
save_path3 = os.getcwd() + '\\res\\' + str(4713) + '.txt'

with shelve.open(save_path2) as f:
    all_species2 = f['all_species']
    best_idx2 = f['best_idx']
    all_species_cost2 = f['all_species_cost']

with shelve.open(save_path3) as f:
    all_species3 = f['all_species']
    best_idx3 = f['best_idx']
    all_species_cost3 = f['all_species_cost']
print(12)
