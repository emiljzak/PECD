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
from scipy.sparse.linalg import expm
from scipy.io import loadmat
from pyexpokit import expmv


import matplotlib.pyplot as plt
""" In this module we are testing several eigensolvers for wavefunciton propagation """

def gen_matrices():
    """ this routine generates random sparse matrices for benchmarking """
    N = 500 #size of matrix
    neval = 10 #number of requested eigenvalues
    A = sparse.random(N,N, density=0.001, format='coo', dtype=float, random_state=0)
    AT = A.copy()
    AT = AT.transpose()
    print(sparse.isspmatrix_coo(A))

    A = A.tocsr()
    print(type(A))
    AT = AT.tocsr()
    
    print(type(AT))
    print(sparse.isspmatrix_csr(A))
    print(sparse.isspmatrix_csr(AT))
    B = 0.5e0 * (A+AT) 
    #print(B.toarray())
    print(type(B))
    print(B.nnz)
    #print(B.row)
    print(B.data)


    sparse.save_npz("mat_500_dens1e-3.npz", B, compressed=False) 
    eval, eigvec = eigsh(B, neval, M=None, sigma=None, which='LM')
    Aloaded = sparse.load_npz("mat_500_dens1e-3.npz")
    evalloaded, eigvec = eigsh(Aloaded, neval, M=None, sigma=None, which='LM')
    print(eval - evalloaded)
    """ WARNING: you must use consistent saving and loading of npz files. Compression, if not handled consistently, can disturb the eigenvalues up to several %. """ 



def test_direct(filename):
    """this method compares the performance of numpy's eigh function with scipy's eigh and eigsh(sparse.) """
    Asparse = sparse.load_npz(filename)
    print(type(Asparse))

    A = sparse.csr_matrix.toarray(Asparse)
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

def run_lanczos(read_npz):
    # ---- generate matrix A
    m=500
    #sqrtA = np.random.rand( n,n ) - 0.5
    #A = np.dot( sqrtA, np.transpose(sqrtA) )
    #plt.spy(A, color='r',markersize=0.1)

    if read_npz == True:
        H = sparse.load_npz("mat_500_dens1e-3.npz")
        print(sparse.issparse(H))
        H = H.toarray()
        HT = np.copy(H)
        HT = np.transpose(HT)
        H = 0.5e00 * (H + HT)
    else:
        H = np.loadtxt("hmat1k.dat",dtype=float)
        #HT = np.copy(H)
        #H = H + np.transpose(HT)
        n = np.size(H,axis=0)
        #for i in range(n):
        #    H[i,i] -= HT[i,i]
        #plt.spy(H, color='r',markersize=0.5)
        #plt.show()

    print(sparse.issparse(H))
    htest = np.copy(H)
    # ---- full solve for eigenvalues for reference
    evalH, vecH = np.linalg.eigh( H )

    # ---- approximate solution by Lanczos
    v0   = np.random.rand( n )
    v0 /= np.sqrt( np.dot( v0, v0 ) )
    T, V = Lanczos_stackoverflow( H, v0, m=m )
    #plt.spy(T)
    #plt.show()
    evalT, vecT = np.linalg.eigh( T )
    VV = np.dot( V, np.transpose( V ) ) # check orthogonality

    #print "A : "; print A
    evalHs = np.sort(evalH)


    evals = np.column_stack((evalHs[:m],np.sort(evalT)))

    with np.printoptions(precision=4, suppress=True, formatter={'float': '{:12.5f}'.format}, linewidth=40):
        #print(eval-eval[0],evalhydrogen - evalhydrogen[0])
        print(evals)
    vtest   = np.random.rand( n )
    print(vtest)
    A = sparse.random(n,n, density=0.001, format='csr', dtype=float, random_state=0)
    print(expmv(0.01, A, vtest, tol=1e-7,krylov_dim=20))

    #print("Exact eigenvalues :"+ str(evalHs))
    #print("Lanczos eigenvalues : "+str(np.sort(evalT)))
    #print("differences : "+str(evalHs[:m]-np.sort(evalT)))

    #plt.plot( esA, np.ones(n)*0.2,  '+' )
    #plt.plot( esT, np.ones(m)*0.1,  '+' )
    #plt.ylim(0,1)
    #plt.show( m )


def test_expms(filename):
    dt = 0.01
    m = 1097
    nprint = 20
    H = np.loadtxt(filename,dtype=float)
    v0   = np.random.rand( np.size(H,axis=0) )
    v0 /= np.sqrt( np.dot( v0, v0 ) )
    
    # 1 PyExpokit
    #print(expmv(dt, H, v0, tol=1e-7,krylov_dim=m))

    # 3 Lanczos - own implementation
    T, V = Lanczos_stackoverflow( H, v0, m=m )
    evalsLan, vecLan = np.linalg.eigh(T)

    print("T: " + str(np.shape(T)))
    print("V: " + str(np.shape(V)))
    print("vecLan: " + str(np.shape(vecLan)))    

    expkrylov = np.dot(vecLan, np.dot( np.diag( np.exp(-1j*evalsLan * dt) ) , vecLan.T ))
    print("expkrylov: " + str(np.shape(expkrylov )))    

    Vexpkrylov = np.dot(V.T,expkrylov)
    print("Vexpkrylov: " + str(np.shape(Vexpkrylov))) 
    myexp =    np.dot(Vexpkrylov, V)
    print("exp(-i*H*t): "+str(np.shape(myexp)) )
    v1_lan = np.dot( myexp , v0)
    print("v1: " + str(np.shape(v1_lan)))    
    print("v0:"+str(v0))
    print("v1:"+str(v1_lan))
    print("v1-v0:" + str(v1_lan-v0)  )


    # 2 Scipy
    print("sparse.Scipy expm:")
    v1_sparsescipy = np.dot(expm(-1j*H*dt),v0)
    print(v1_sparsescipy)

    # 2 Scipy
    print("Scipy expm:")
    v1_scipy = np.dot(linalg.expm(-1j*H*dt),v0)
    print(v1_scipy)
    print("v1: scipy, sparse.scipy, lanczos")
    for i in range(nprint):
         print('%10.5f %10.5fi' % (v1_scipy[i].real, v1_scipy[i].imag)+'%10.5f %10.5fi' % (v1_sparsescipy[i].real, v1_sparsescipy[i].imag)+'%10.5f %10.5fi' % (v1_lan[i].real, v1_lan[i].imag))

filename = "hmat1k.dat"


#read_matlab()

#gen_matrices()

#run_lanczos(read_npz = False)
#test_direct(filename)


test_expms(filename)
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

