#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2021 Emil Zak <emil.zak@cfel.de>
#
import numpy as np
import sys
import time
from scipy import linalg
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.io import loadmat
import matplotlib.pyplot as plt
""" In this module we are testing several eigensolvers for wavefunciton propagation """

def gen_matrices():
    """ this routine generates random sparse matrices for benchmarking """
    N = 1000 #size of matrix
    neval = 20 #number of requested eigenvalues
    A = sparse.random(N,N, density=0.01)
    AT = A
    AT = np.transpose(AT)
    A = 0.5 * (A + AT) 
    print(type(A))
    sparse.save_npz("mat_1k_dens1e-2.npz", A, compressed=False) 
    eval, eigvec = eigsh(A, neval, M=None, sigma=None, which='LM')
    Aloaded = sparse.load_npz("mat_1k_dens1e-2.npz")
    evalloaded, eigvec = eigsh(Aloaded, neval, M=None, sigma=None, which='LM')
    print(eval - evalloaded)
    """ WARNING: you must use consistent saving and loading of npz files. Compression, if not handled consistently, can disturb the eigenvalues up to several %. """ 



def test_direct(filename):
    """this method compares the performance of numpy's eigh function with scipy's eigh and eigsh(sparse.) """
    Asparse = sparse.load_npz(filename)
    print(type(Asparse))

    A = sparse.csc_matrix.toarray(Asparse)
    print(type(A))
    
    t_start_process = time.process_time()  
    t_start_time = time.time()
    t_start_perf = time.perf_counter()
    evalscipysparse, eigvecscipysparse = sparse.linalg.eigsh(A,k=10,which='LM')
    t_stop_process = time.process_time()  
    t_stop_time = time.time()
    t_stop_perf = time.perf_counter()

    print(" ========= Scipy.sparse eigsh =========")
    print("time.time():" +  str("%10.3f"%(t_stop_time-t_start_time)) + "s")
    print("time.process_time():" +  str("%10.3f"%(t_stop_process-t_start_process)) + "s")
    print("time.perf_counter():" +  str("%10.3f"%(t_stop_perf-t_start_perf)) + "s")




    t_start_process = time.process_time()  
    t_start_time = time.time()
    t_start_perf = time.perf_counter()
    evalnp, eigvecnp = np.linalg.eigh(A, UPLO='L')
    t_stop_process = time.process_time()  
    t_stop_time = time.time()
    t_stop_perf = time.perf_counter()

    print(" ========= Numpy eigh =========")
    print("time.time():" +  str("%10.3f"%(t_stop_time-t_start_time)) + "s")
    print("time.process_time():" +  str("%10.3f"%(t_stop_process-t_start_process)) + "s")
    print("time.perf_counter():" +  str("%10.3f"%(t_stop_perf-t_start_perf)) + "s")

    np.abs(evalnp[::-1]).sort()
    print(np.abs(evalscipysparse[:5])-evalnp[:5])
    exit()

    t_start_process = time.process_time()  
    t_start_time = time.time()
    t_start_perf = time.perf_counter()
    evalscipy, eigvecscipy = linalg.eigh(A)
    t_stop_process = time.process_time()  
    t_stop_time = time.time()
    t_stop_perf = time.perf_counter()

    print(" ========= Scipy eigh =========")
    print("time.time():" +  str("%10.3f"%(t_stop_time-t_start_time)) + "s")
    print("time.process_time():" +  str("%10.3f"%(t_stop_process-t_start_process)) + "s")
    print("time.perf_counter():" +  str("%10.3f"%(t_stop_perf-t_start_perf)) + "s")
    print(evalnp - evalscipy)

def read_matlab():

    mat = loadmat('cars_train_annos.mat')

filename = "mat_5k_dens1e-4.npz"



def Lanczos_stackoverflow( A, v, m=100 ):
    n = len(v)
    if m>n: m = n;
    # from here https://en.wikipedia.org/wiki/Lanczos_algorithm
    V = np.zeros( (m,n) )
    T = np.zeros( (m,m) )
    vo   = np.zeros(n)
    beta = 0
    for j in range( m-1 ):
        w    = np.dot( A, v )
        alfa = np.dot( w, v )
        w    = w - alfa * v - beta * vo
        beta = np.sqrt( np.dot( w, w ) ) 
        vo   = v
        v    = w / beta 
        T[j,j  ] = alfa 
        T[j,j+1] = beta
        T[j+1,j] = beta
        V[j,:]   = v
    w    = np.dot( A,  v )
    alfa = np.dot( w, v )
    w    = w - alfa * v - beta * vo
    T[m-1,m-1] = np.dot( w, v )
    V[m-1]     = w / np.sqrt( np.dot( w, w ) ) 
    return T, V

# ---- generate matrix A
n = 100; m=40
sqrtA = np.random.rand( n,n ) - 0.5
A = np.dot( sqrtA, np.transpose(sqrtA) )

# ---- full solve for eigenvalues for reference
esA, vsA = np.linalg.eig( A )

# ---- approximate solution by Lanczos
v0   = np.random.rand( n )
v0 /= np.sqrt( np.dot( v0, v0 ) )
T, V = Lanczos_stackoverflow( A, v0, m=m )
esT, vsT = np.linalg.eig( T )
VV = np.dot( V, np.transpose( V ) ) # check orthogonality

#print "A : "; print A
print("T : " + str(T))
print("VV :" + str(VV))
print("esA :"+ str(esA))
print("esT : "+str(np.sort(-esT)))
print("differences : "+str(esA[:m]+np.sort(-esT)))

plt.plot( esA, np.ones(n)*0.2,  '+' )
plt.plot( esT, np.ones(m)*0.1,  '+' )
plt.ylim(0,1)
plt.show( m )

exit()

#read_matlab()

#gen_matrices()

test_direct(filename)

exit()




N = 5000 #size of matrix 
neval = 20 #number of requested eigenvalues
""" time.time() is most basis wall clock time with resolution dictated by linux clock. It returns absolute time.
    time.perf_counter() is like time.time() but with higest possilbe resolution. They both contain sleep time , which means that also CPU load is reflected in it. This is relative time.
    time.process_time() does not include sleep time, it includes both CPU and system time. This is relative time. It does not containt time used for instance to plot and see an image during run.
    """

t_start_process = time.process_time()  
t_start_time = time.time()
t_start_perf = time.perf_counter()


# create a sparse matrix w0ith specific density
A = sparse.random(N,N, density=0.0001)
AT = A
AT = np.transpose(AT)
A = 0.5 * (A + AT)
# visualize the sparse matrix with Spy
plt.spy(A,color='r',markersize=0.1)
plt.show()
print(A)
eval, eigvec = linalg.eigsh(A, neval, M=None, sigma=None, which='LM')
print(eval)
print("eigenvectors:")
print(eigvec)
t_stop_process = time.process_time()  
t_stop_time = time.time()
t_stop_perf = time.perf_counter()


print("time.time():" +  str("%10.3f"%(t_stop_time-t_start_time)) + "s")
print("time.process_time():" +  str("%10.3f"%(t_stop_process-t_start_process)) + "s")
print("time.perf_counter():" +  str("%10.3f"%(t_stop_perf-t_start_perf)) + "s")

