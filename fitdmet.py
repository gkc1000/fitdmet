#! /usr/bin/env python

import itertools as it
import numpy as np
import numpy.random
import scipy
import scipy.linalg as la
import scipy.optimize
np.set_printoptions(3, linewidth=1000, suppress=True)

def make_inter_h(unit_L, t):
    inter_h = np.zeros([unit_L, unit_L])
    inter_h[0,unit_L-1] = t
    return inter_h

def make_lattice_h(unit_cell_h, inter_h, L):
    h = np.zeros([L,L])
    unit_L = unit_cell_h.shape[0]
    nunits = L // unit_L
    for i in range(L//unit_L):
        icell = slice(i*unit_L,(i+1)*unit_L)
        h[icell,icell] = unit_cell_h[:,:]
        for j in range(L//unit_L):
            jcell = slice(j*unit_L,(j+1)*unit_L)
            if abs(i-j) == 1 and j > i:
                h[icell,jcell] = inter_h.T
            if abs(i-j) == 1 and j < i:
                h[icell,jcell] = inter_h
                
    h[0:unit_L, (nunits-1)*unit_L:] = inter_h[:,:]
    h[(nunits-1)*unit_L:,0:unit_L] = inter_h[:,:].T
    return h

def make_unit_cell_h(t):
    unit_cell_h = np.array([[0.,t],
                          [t,0.]])
    return unit_cell_h

def make_dm(h, N):
    eigs, vecs = la.eigh(h)
    dm = np.zeros_like(h)
    for i in range(N):
        dm += np.outer(vecs[:,i], vecs[:,i])
    return dm

def make_dm_mu(h, mu):
    eigs, vecs = la.eigh(h)
    dm = np.zeros_like(h)
    for i in range(h.shape[0]):
        if eigs[i] < mu:
            dm += np.outer(vecs[:,i], vecs[:,i])
    return dm

def make_imp_bath_h(h, unit_L, dm):
    L = h.shape[0]
    trans = np.zeros([L, 2*unit_L])
    trans[:unit_L,:unit_L] = np.eye(unit_L)
    u, s, vt = la.svd(dm[:unit_L, unit_L:], full_matrices=False)
    trans[unit_L:,unit_L:] = vt.T
    return np.dot(trans.T, np.dot(h, trans))

def unpack_tri(u, L):
    umat = np.zeros([L,L])
    ptr = 0
    for i in range(L):
        for j in range(i+1):
            #if ptr!= 0:
            umat[i,j] = u[ptr]
            ptr+=1
    for i in range(L):
        for j in range(i):            
            umat[j,i] = umat[i,j]
    return umat

def pack_tri(umat):
    L = umat.shape[0]
    u = np.zeros([L*(L+1)//2])
    ptr = 0
    for i in range(L):
        for j in range(i+1):
            #if ptr != 0:
            u[ptr] = umat[i,j]
            ptr +=1
    return u

def add_u(lattice_h, umat):
    L = lattice_h.shape[0]
    unit_L = umat.shape[0]
    new_h = lattice_h.copy()
    for i in range(L//unit_L):
        icell = slice(i*unit_L,(i+1)*unit_L)
        new_h[icell,icell] += umat
    return new_h

def get_dV_latt_du(u, L):
    nparam = len(u)
    nao = int(np.sqrt(nparam * 2))
    nblocks = L // nao
    g = np.zeros((nparam, nblocks, nao, nblocks, nao))
    for idx, (i, j) in enumerate(it.combinations_with_replacement(range(nao), 2)):
        for R in range(nblocks):
            g[idx, R, i, R, j] = g[idx, R, j, R, i] = 1
    g = g.reshape((-1, nblocks * nao, nblocks * nao))
    return g

def test_grad(u, errfunc, gradfunc, dx=1e-5):
    """
    Test analytic gradient and compare with numerical one.
    """
    param0 = u.copy()
    grad_num = np.zeros_like(param0)
    grad_ana = gradfunc(param0)
    f0 = errfunc(param0)
    for i in range(len(grad_num)):
        param_tmp = param0.copy()
        param_tmp[i] += dx
        grad_num[i] = (errfunc(param_tmp) - f0) / dx
    
    print ("Test gradients in fitting, finite difference dx = %s" % dx)
    print("Analytical gradient:\n%s" % grad_ana)
    print("Numerical gradient:\n%s" %  grad_num)
    print("grad_ana / grad_num:\n%s" % (grad_ana / grad_num))
    print("Norm difference: %s" % la.norm(grad_ana - grad_num))    

def fit_dm(lattice_h, N, target_dm, use_scipy=True, num_grad=False):
    L = lattice_h.shape[0]
    unit_L = target_dm.shape[0]
    nL = L // unit_L
    u = np.random.random([unit_L*(unit_L+1)//2])
    dV_latt_du = get_dV_latt_du(u, L)
    #target_dm_latt = la.block_diag(*((target_dm,) * nL))

    def metric(u):
        umat = unpack_tri(u, unit_L)
        new_h = add_u(lattice_h, umat)                
        new_dm = make_dm(new_h, N)[:unit_L,:unit_L]
        ret = la.norm(new_dm-target_dm)
        return ret
    
    def grad(u):
        umat = unpack_tri(u, unit_L)
        new_h = add_u(lattice_h, umat)
        ew, ev = la.eigh(new_h)
        ewocc, ewvirt = ew[:N], ew[N:]
        evocc, evvirt = ev[:, :N], ev[:, N:]
        
        dm = np.dot(ev[:, :N], ev[:, :N].T)
        new_dm = dm[:unit_L,:unit_L]
        
        drho = new_dm - target_dm
        val = la.norm(drho)

        e_mn = 1. / (-ewvirt.reshape(-1, 1) + ewocc)
        temp_mn = np.dot(evvirt[:unit_L].T, np.dot(drho, evocc[:unit_L])) * e_mn / val
        dnorm_dV = np.dot(evvirt, np.dot(temp_mn, evocc.T))
        dnorm_dV += dnorm_dV.T
        # dnorm_dV shape ((nAO, nAO))
        # dV_du shape (nu, nAO, nAO)
        return np.tensordot(dV_latt_du, dnorm_dV, axes=((1, 2), (0, 1)))
    
    #test_grad(u, metric, grad, dx=1e-4)
    #test_grad(u, metric, grad, dx=1e-5)
    if num_grad:
        grad = None

    if use_scipy:
        res = scipy.optimize.minimize(metric, u, jac=grad, method="cg", \
                options={"maxiter": 500, 'disp': True})
        #print res
        #print "minimum", res.fun, res.x
        return unpack_tri(res.x, unit_L)
    else:
        from libdmet_solid.routine import fit
        from libdmet_solid.utils import logger as log
        log.verbose = 'DEBUG1'
        param, val, pattern, grad = fit.minimize(metric, u, MaxIter=500, gradfunc=grad)
        print ("param:", param)
        print ("err:", val)
        print ("grad:", grad)
        return unpack_tri(param, unit_L)

def main():
    np.set_printoptions(precision=4,linewidth=120)
    unit_L = 2
    t = -0.5
    tinter = 1.
    mu = -1
    N = 14
    L = 34
    #N = 2
    #L = 6
    lattice_h = make_lattice_h(make_unit_cell_h(t),
                               make_inter_h(unit_L, tinter),L)
    
    dm = make_dm(lattice_h, N)

    himp = make_imp_bath_h(lattice_h, unit_L, make_dm(lattice_h, N))

    target_dm = dm[:unit_L,:unit_L]
    target_dm[0,0]+=0.1
    target_dm[1,1]-=0.1

    target_dm[0,1]+=0.1
    target_dm[1,0]+=0.1

    umat = fit_dm(lattice_h, N, target_dm)

    new_h = add_u(lattice_h, umat)

    new_dm = make_dm(new_h, N)[:unit_L,:unit_L]
    print ("fitted dm\n", new_dm)
    print ("target dm\n", target_dm)
    print ("error:", la.norm(new_dm - target_dm))
    print ("umat\n", umat)
    old_eigs = scipy.linalg.eigvalsh(lattice_h)
    new_eigs = scipy.linalg.eigvalsh(new_h)
    print (old_eigs)
    print (new_eigs)
    print ("old gap:", old_eigs[N] - old_eigs[N-1])
    print ("new gap:", new_eigs[N] - new_eigs[N-1])

if __name__ == "__main__":
    main()
