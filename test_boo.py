import pytest
from boo import *

from math import sqrt

golden = (1 + sqrt(5))/2

pos_ico = np.array([ [0,0,0],
    [0, 1, golden], [0, 1, -golden], [0, -1, golden], [0, -1, -golden],
    [1, golden, 0], [1, -golden, 0], [-1, golden, 0], [-1, -golden, 0],
    [golden, 0, 1], [golden, 0, -1], [-golden, 0, 1], [-golden, 0, -1],
])
bonds_ico = np.column_stack((np.zeros(12, int), np.arange(12)+1))

def test_ico():
    qlm = bonds2qlm(pos_ico, bonds_ico)
    assert ql(qlm)[0] == pytest.approx(0.66332495807107972)
    assert wl(qlm)[0] == pytest.approx(-0.052131352806275739)
