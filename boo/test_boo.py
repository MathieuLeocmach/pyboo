from math import sqrt, floor
import numpy as np
import pytest
from scipy.spatial import cKDTree as KDTree
from numba import jit
from . import boo


def bonds2ngbs(bonds, nb_particles=None):
    """convert from bonds to ngb"""
    if nb_particles is None:
        nb_particles = bonds.max()+1
    nngbs = np.zeros(nb_particles, int)
    np.add.at(nngbs, bonds.ravel(), 1)
    ngbs = np.full((nb_particles, nngbs.max()), -1, int)
    for a, b in bonds:
        ngbs[a, np.where(ngbs[a] == -1)[0][0]] = b
        ngbs[b, np.where(ngbs[b] == -1)[0][0]] = a
    return ngbs


@jit(nopython=True) #numba to speed up our brute force approach
def periodic_neighbours(pos, maxdist, L):
    maxdistsq = maxdist**2
    rL = 1./L
    bonds = []
    dists = []
    for i in range(len(pos)-1):
        for j in range(i+1, len(pos)):
            distsq = 0
            for d in range(pos.shape[1]):
                diff = pos[i,d] - pos[j,d]
                diff -= L * floor(diff * rL + 0.5)
                distsq += diff*diff
            if distsq < maxdistsq:
                bonds.append(i)
                bonds.append(j)
                dists.append(distsq)
    return np.array(bonds, np.int64).reshape((len(dists), 2)), np.sqrt(np.array(dists, np.float64))



@jit(nopython=True) #numba to speed up our brute force approach
def shortest_bonds2ngbs(bonds, dists, N, Nngb=12):
    """Convert from bonds to neighbour array (at most Nngb neighbours)"""
    ngbs = np.full((N, Nngb), -1, np.int64)
    nngbs = np.zeros(N, np.int64)
    #sort bonds by increasing length to be able to use first-in-first-served
    sbonds = bonds[np.argsort(dists)]
    for i in range(len(sbonds)):
        a = sbonds[i,0]
        b = sbonds[i,1]
        if nngbs[a]< ngbs.shape[1]:
            ngbs[a, nngbs[a]] = b
            nngbs[a] += 1
        if nngbs[b]< ngbs.shape[1]:
            ngbs[b, nngbs[b]] = a
            nngbs[b] += 1
    return ngbs




def known_structure_test(pos, bonds, q6, w6, q4):
    """Generic test for known structures"""
    q6m = boo.bonds2qlm(pos, bonds)
    assert boo.ql(q6m)[0] == pytest.approx(q6)
    assert boo.wl(q6m)[0] == pytest.approx(w6)
    assert boo.ql(q6m)[0] == pytest.approx(sqrt(boo.product(q6m, q6m)[0]))
    q4m = boo.bonds2qlm(pos, bonds, l=4)
    assert boo.ql(q4m)[0] == pytest.approx(q4)
    q6m_b = boo.ngbs2qlm(pos, bonds2ngbs(bonds))
    assert np.all(q6m_b[0] == q6m[0])
    assert np.all(boo.ql(q6m_b) <= boo.ql(q6m))




def test_ico():
    """13 particles icosahedron"""
    golden = (1 + sqrt(5))/2
    pos = np.array([
        [0, 0, 0],
        [0, 1, golden], [0, 1, -golden], [0, -1, golden], [0, -1, -golden],
        [1, golden, 0], [1, -golden, 0], [-1, golden, 0], [-1, -golden, 0],
        [golden, 0, 1], [golden, 0, -1], [-golden, 0, 1], [-golden, 0, -1],
    ])
    bonds = np.column_stack((np.zeros(12, int), np.arange(12)+1))
    known_structure_test(
        pos, bonds,
        0.66332495807107972,
        -0.052131352806275739,
        0)



def test_fcc():
    """Face centered cubic"""
    pos = np.array([
        [0, 0, 0],
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1]
    ])
    bonds = np.column_stack((np.zeros(12, int), np.arange(12)+1))
    known_structure_test(
        pos, bonds,
        0.57452425971406984,
        -0.0026260383340077592,
        0.19094065395649326)

def test_hcp():
    """Hexagonal compact"""
    pos = np.array([
        [0, 0, 0],
        [1, 0, 0], [0.5, sqrt(3)/2, 0], [-0.5, sqrt(3)/2, 0],
        [-1, 0, 0], [0.5, -sqrt(3)/2, 0], [-0.5, -sqrt(3)/2, 0],
        [0.5, 1/(2*sqrt(3)), sqrt(2/3)],
        [-0.5, 1/(2*sqrt(3)), sqrt(2/3)],
        [0, -1/sqrt(3), sqrt(2/3)],
        [0.5, 1/(2*sqrt(3)), -sqrt(2/3)],
        [-0.5, 1/(2*sqrt(3)), -sqrt(2/3)],
        [0, -1/sqrt(3), -sqrt(2/3)]
    ])
    bonds = np.column_stack((np.zeros(12, int), np.arange(12)+1))
    known_structure_test(
        pos, bonds,
        0.48476168522368324,
        -0.0014913304119056434,
        0.097222222222222085)



def test_bcc9():
    """Body centered cubic (the 8 closest particles)"""
    pos = np.array([
        [0, 0, 0],
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
    ])
    bonds = np.column_stack((np.zeros(8, int), np.arange(8)+1))
    known_structure_test(
        pos, bonds,
        0.62853936105470865,
        0.0034385344925653301,
        0.50917507721731547)
        
def test_crystal_gel():
    """Experimental data from a crystallizing gel."""
    pos = np.loadtxt('examples/AR-Res06A_scan2_t890.xyz', skiprows=1)
    maxbondlength = 12.5
    #spatial indexing
    tree = KDTree(pos, 12)
    #query
    bonds = tree.query_pairs(maxbondlength, output_type='ndarray')
    inside = np.all((pos - pos.min(0) > maxbondlength) & (pos.max() - pos > maxbondlength), -1)
    #number of neighbours per particle
    Nngb = np.zeros(len(pos), int)
    np.add.at(Nngb, bonds.ravel(), 1)
    inside[Nngb<4] = False
    #tensorial boo
    q6m = boo.bonds2qlm(pos, bonds, l=6)
    q4m = boo.bonds2qlm(pos, bonds, l=4)
    #coarse-graining
    Q6m, inside2 = boo.coarsegrain_qlm(q6m, bonds, inside)
    Q4m, inside3 = boo.coarsegrain_qlm(q4m, bonds, inside)
    assert np.all(inside2 == inside3)
    #crystals
    xpos = boo.x_particles(q6m, bonds)
    assert xpos.sum() == 14188
    #surface particles
    surf = boo.x_particles(q6m, bonds, nb_thr=2) & np.bitwise_not(xpos)
    assert surf.sum() == 9288

def test_periodic_glass():
    """Monte-Carlo simulation data of a polydisperse hard sphere glass."""
    #prepare input
    L = 203.
    xyzr = np.loadtxt('examples/3d_6_0.54_0.dat')
    radius = xyzr[:, -1]
    pos = xyzr[:, :-1]
    #bond network (oversized bond length)
    maxbondlength = 15.
    bonds, dists = periodic_neighbours(pos, maxbondlength, L)
    #12 closest (surface distance) neighbours
    drij = dists - (radius[bonds[:, 0]] + radius[bonds[:, 1]])
    ngbs = shortest_bonds2ngbs(bonds, drij, len(pos))
    #tensorial boo
    q6m = boo.ngbs2qlm(pos, ngbs, l=6, periods=L)
    q4m = boo.ngbs2qlm(pos, ngbs, l=4, periods=L)
    #crystals
    xpos = boo.x_ngbs(q6m, ngbs).sum(-1) > 6
    assert xpos.sum() == 22
    #surface particles
    surf = (boo.x_ngbs(q6m, ngbs).sum(-1) > 2) & np.bitwise_not(xpos)
    assert surf.sum() == 748
    #icosahedra
    w6 = boo.wl(q6m)
    ico = w6 < -0.028
    assert ico.sum() == 111
