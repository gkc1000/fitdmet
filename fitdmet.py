import numpy as np
import numpy.random
import scipy
import scipy.linalg
import scipy.optimize

def make_inter_h(unit_L, t):
    inter_h = np.zeros([unit_L, unit_L])
    inter_h[0,unit_L-1] = t
    return inter_h

def make_lattice_h(unit_cell_h, inter_h, L):
    h = np.zeros([L,L])
    unit_L = unit_cell_h.shape[0]
    nunits = L / unit_L
    for i in range(L/unit_L):
        icell = slice(i*unit_L,(i+1)*unit_L)
        h[icell,icell] = unit_cell_h[:,:]
        for j in range(L/unit_L):
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
    eigs, vecs = np.linalg.eigh(h)
    dm = np.zeros_like(h)
    for i in range(N):
        dm += np.outer(vecs[:,i], vecs[:,i])
    return dm

def make_dm_mu(h, mu):
    eigs, vecs = np.linalg.eigh(h)
    dm = np.zeros_like(h)
    for i in range(h.shape[0]):
        if eigs[i] < mu:
            dm += np.outer(vecs[:,i], vecs[:,i])
    return dm


def make_imp_bath_h(h, unit_L, dm):
    L = h.shape[0]
    trans = np.zeros([L, 2*unit_L])
    trans[:unit_L,:unit_L] = np.eye(unit_L)
    u, s, vt = np.linalg.svd(dm[:unit_L, unit_L:], full_matrices=False)
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
    u = np.zeros([L*(L+1)/2])
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
    for i in range(L/unit_L):
        icell = slice(i*unit_L,(i+1)*unit_L)
        new_h[icell,icell] += umat
    return new_h

def fit_dm(lattice_h, N, target_dm):
    L = lattice_h.shape[0]
    unit_L = target_dm.shape[0]
    def metric(u):
        umat = unpack_tri(u, unit_L)
        new_h = add_u(lattice_h, umat)                
        new_dm = make_dm(new_h, N)[:unit_L,:unit_L]
        ret = np.linalg.norm(new_dm-target_dm)
        return ret

    u = np.random.random([unit_L*(unit_L+1)/2])
    res = scipy.optimize.minimize(metric, u, method="cg")
    #print res
    #print "minimum", res.fun, res.x
    return unpack_tri(res.x, unit_L)

def main():
    np.set_printoptions(precision=4,linewidth=120)
    unit_L = 2
    t = -0.5
    tinter = 1.
    mu = -1
    N = 14
    L = 34
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
    print "fitted dm\n", new_dm
    print "target dm\n", target_dm
    print "error:", np.linalg.norm(new_dm - target_dm)
    print "umat\n", umat
    old_eigs = scipy.linalg.eigvalsh(lattice_h)
    new_eigs = scipy.linalg.eigvalsh(new_h)
    print old_eigs
    print new_eigs
    print "old gap:", old_eigs[N] - old_eigs[N-1]
    print "new gap:", new_eigs[N] - new_eigs[N-1]

