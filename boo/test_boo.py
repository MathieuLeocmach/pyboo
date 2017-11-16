import pytest
from .boo import *
from math import sqrt

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
    
pos_fcc = np.array([[0,0,0],
    [1,1,0], [1,-1,0], [-1,1,0], [-1,-1,0],
    [0,1,1], [0,-1,1], [0,1,-1], [0,-1,-1],
    [1,0,1], [1,0,-1], [-1,0,1], [-1,0,-1]
])

def test_fcc():
    qlm = bonds2qlm(pos_fcc, bonds_ico)
    assert ql(qlm)[0] == pytest.approx(0.57452425971406984)
    assert wl(qlm)[0] == pytest.approx(-0.0026260383340077592)
    assert ql(qlm)[0] == pytest.approx(sqrt(product(qlm,qlm)[0]))
    q4m = bonds2qlm(pos_fcc, bonds_ico, l=4)
    assert ql(q4m)[0] == pytest.approx(0.19094065395649326)
