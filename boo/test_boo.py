from math import sqrt
import numpy as np
import pytest
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
