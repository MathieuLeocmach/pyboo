import pytest
from .boo import *
from math import sqrt

def bonds2ngbs(bonds, N=None):
    """convert from bonds to ngb"""
    if N is None:
        N = bonds.max()+1
    nngbs = np.zeros(N, int)
    np.add.at(nngbs, bonds.ravel(), 1)
    ngbs = np.full((N, nngbs.max()), -1, int)
    for a,b in bonds:
            ngbs[a, np.where(ngbs[a]==-1)[0][0]] = b
            ngbs[b, np.where(ngbs[b]==-1)[0][0]] = a
    return ngbs
        

golden = (1 + sqrt(5))/2

pos_ico = np.array([ [0,0,0],
    [0, 1, golden], [0, 1, -golden], [0, -1, golden], [0, -1, -golden],
    [1, golden, 0], [1, -golden, 0], [-1, golden, 0], [-1, -golden, 0],
    [golden, 0, 1], [golden, 0, -1], [-golden, 0, 1], [-golden, 0, -1],
])
bonds_ico = np.column_stack((np.zeros(12, int), np.arange(12)+1))

def test_ico():
    q6m = bonds2qlm(pos_ico, bonds_ico)
    assert ql(q6m)[0] == pytest.approx(0.66332495807107972)
    assert wl(q6m)[0] == pytest.approx(-0.052131352806275739)
    assert ql(q6m)[0] == pytest.approx(sqrt(product(q6m,q6m)[0]))
    q4m = bonds2qlm(pos_ico, bonds_ico, l=4)
    assert ql(q4m)[0] == pytest.approx(0)
    q6m_b = ngbs2qlm(pos_ico, bonds2ngbs(bonds_ico))
    assert np.all(q6m_b[0] == q6m[0])
    assert np.all(ql(q6m_b) <= ql(q6m))
    
pos_fcc = np.array([[0,0,0],
    [1,1,0], [1,-1,0], [-1,1,0], [-1,-1,0],
    [0,1,1], [0,-1,1], [0,1,-1], [0,-1,-1],
    [1,0,1], [1,0,-1], [-1,0,1], [-1,0,-1]
])

def test_fcc():
    q6m = bonds2qlm(pos_fcc, bonds_ico)
    assert ql(q6m)[0] == pytest.approx(0.57452425971406984)
    assert wl(q6m)[0] == pytest.approx(-0.0026260383340077592)
    assert ql(q6m)[0] == pytest.approx(sqrt(product(q6m,q6m)[0]))
    q4m = bonds2qlm(pos_fcc, bonds_ico, l=4)
    assert ql(q4m)[0] == pytest.approx(0.19094065395649326)
    q6m_b = ngbs2qlm(pos_fcc, bonds2ngbs(bonds_ico))
    assert np.all(q6m_b[0] == q6m[0])
    assert np.all(ql(q6m_b) <= ql(q6m))
    
pos_hcp = np.array([[0,0,0],
    [1, 0, 0], [0.5, sqrt(3)/2, 0], [-0.5, sqrt(3)/2, 0],
    [-1, 0, 0], [0.5, -sqrt(3)/2, 0], [-0.5, -sqrt(3)/2, 0],  
    [0.5, 1/(2*sqrt(3)), sqrt(2/3)], [-0.5, 1/(2*sqrt(3)), sqrt(2/3)], [0, -1/sqrt(3), sqrt(2/3)],
    [0.5, 1/(2*sqrt(3)), -sqrt(2/3)], [-0.5, 1/(2*sqrt(3)), -sqrt(2/3)], [0, -1/sqrt(3), -sqrt(2/3)]
])

def test_hcp():
    q6m = bonds2qlm(pos_hcp, bonds_ico)
    assert ql(q6m)[0] == pytest.approx(0.48476168522368324)
    assert wl(q6m)[0] == pytest.approx(-0.0014913304119056434)
    assert ql(q6m)[0] == pytest.approx(sqrt(product(q6m,q6m)[0]))
    q4m = bonds2qlm(pos_hcp, bonds_ico, l=4)
    assert ql(q4m)[0] == pytest.approx(0.097222222222222085)
    q6m_b = ngbs2qlm(pos_hcp, bonds2ngbs(bonds_ico))
    assert np.all(q6m_b[0] == q6m[0])
    assert np.all(ql(q6m_b) <= ql(q6m))
    
pos_bcc9 = np.array([[0,0,0],
    [1, 1, 1], [1, 1, -1], [1, -1, 1], [1,-1,-1],
    [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1,-1,-1],
])
bonds_bcc9 = np.column_stack((np.zeros(8, int), np.arange(8)+1))

def test_bcc():
    q6m = bonds2qlm(pos_bcc9, bonds_bcc9)
    assert ql(q6m)[0] == pytest.approx(0.62853936105470865)
    assert wl(q6m)[0] == pytest.approx(0.0034385344925653301)
    assert ql(q6m)[0] == pytest.approx(sqrt(product(q6m,q6m)[0]))
    q4m = bonds2qlm(pos_bcc9, bonds_bcc9, l=4)
    assert ql(q4m)[0] == pytest.approx(0.50917507721731547)
    q6m_b = ngbs2qlm(pos_bcc9, bonds2ngbs(bonds_bcc9))
    assert np.all(q6m_b[0] == q6m[0])
    assert np.all(ql(q6m_b) <= ql(q6m))
