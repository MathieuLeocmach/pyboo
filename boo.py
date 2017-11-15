#
#    Copyright 2011 Mathieu Leocmach
#
#    This file is part of Colloids.
#
#    Colloids is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Colloids is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Colloids.  If not, see <http://www.gnu.org/licenses/>.
#
import numpy as np
from scipy.special import sph_harm
from scipy.spatial import cKDTree as KDTree
from numba import jit, vectorize, guvectorize
from colloids import periodic

@vectorize#(['float64(complex128)', 'float32(complex64)'])
def abs2(x):
    return x.real**2 + x.imag**2
        

def cart2sph(cartesian):
    """Convert Cartesian coordinates [[x,y,z],] to spherical coordinates [[r,phi,theta],]
phi is cologitudinal and theta azimutal"""
    spherical = np.zeros_like(cartesian)
    #distances
    c2 = cartesian**2
    r2 = c2.sum(-1)
    spherical[:,0] = np.sqrt(r2)
    #work only on non-zero, non purely z vectors
    sel = (r2 > c2[:,0]) | (r2+1.0 > 1.0)
    x, y, z = cartesian[sel].T
    r = spherical[sel,0]
    #colatitudinal phi [0, pi[
    spherical[sel,1] = np.arccos(z/r)
    #azimutal (longitudinal) theta [0, 2pi[
    theta = np.arctan2(y, x)
    theta[theta<0] += 2*np.pi
    spherical[sel,2] = theta
    return spherical
    
def vect2Ylm(v, l):
    """Projects vectors v on the base of spherical harmonics of degree l."""
    spherical = cart2sph(v)
    return sph_harm(
        np.arange(l+1)[:,None], l, 
        spherical[:,2][None,:], 
        spherical[:,1][None,:]
        )
        
def single_pos2qlm(pos, i, ngb_indices, l=6):
    """Returns the qlm for a single position"""
    #vectors to neighbours
    vectors = pos[ngb_indices]-pos[i]
    return vect2Ylm(vectors, l).mean(-1)
    
def bonds2qlm(pos, bonds, l=6, periods=-1):
    """Returns the qlm for every particle"""
    qlm = np.zeros((len(pos), l+1), np.complex128)
    #spherical harmonic coefficients for each bond
    Ylm = vect2Ylm(
        periodic.periodify(
            pos[bonds[:,0]],
            pos[bonds[:,1]],
            periods
        ),
        l
    ).T
    #bin bond into each particle belonging to it
    np.add.at(qlm, bonds[:,0], Ylm)
    np.add.at(qlm, bonds[:,1], Ylm)
    #divide by the number of bonds each particle belongs to
    Nngb = np.zeros(len(pos), int)
    np.add.at(Nngb, bonds.ravel(), 1)
    return qlm / np.maximum(1, Nngb)[:,None]
    
def coarsegrain_qlm(qlm, bonds, inside):
    """Coarse grain the bond orientational order on the neighbourhood of a particle
    $$Q_{\ell m}(i) = \frac{1}{N_i+1}\left( q_{\ell m}(i) +  \sum_{j=0}^{N_i} q_{\ell m}(j)\right)$$
    Returns Qlm and the mask of the valid particles
    """
    #Valid particles must be valid themselves have only valid neighbours
    inside2 = np.copy(inside)
    np.bitwise_and.at(inside2, bonds[:,0], inside[bonds[:,1]])
    np.bitwise_and.at(inside2, bonds[:,1], inside[bonds[:,0]])
    #number of neighbours
    Nngb = np.zeros(len(qlm), int)
    np.add.at(Nngb, bonds.ravel(), 1)
    #sum the boo coefficients of all the neighbours
    Qlm = np.zeros_like(qlm)
    np.add.at(Qlm, bonds[:,0], qlm[bonds[:,1]])
    np.add.at(Qlm, bonds[:,1], qlm[bonds[:,0]])
    Qlm[np.bitwise_not(inside2)] = 0
    return Qlm / np.maximum(1, Nngb)[:,None], inside2
    

@guvectorize(
    ['void(complex128[:], complex128[:], float64[:])'], 
    '(n),(n)->()', 
    #nopython=True
)
def boo_product(qlm1, qlm2, prod):
    """Product between two qlm"""
    l = qlm1.shape[0]-1
    prod[0] = (qlm1[0] * qlm2[0].conjugate()).real
    for i in range(1, len(qlm1)):
        prod[0] += 2 * (qlm1[i] * qlm2[i].conjugate()).real
    prod[0] *= 4*np.pi/(2*l+1)
    
@jit(nopython=True)
def ql(qlm):
    """Second order rotational invariant of the bond orientational order of l-fold symmetry
    $$ q_\ell = \sqrt{\frac{4\pi}{2l+1} \sum_{m=-\ell}^{\ell} |q_{\ell m}|^2 } $$"""
    q = abs2(qlm[...,0])
    q += 2*abs2(qlm[...,1:]).sum(-1)
    l = qlm.shape[-1]-1
    return np.sqrt(4*np.pi / (2*l+1) * q)
    
@jit#(['complex128[:](complex128[:,:], int64)'], nopython=True)
def get_qlm(qlms, m):
    """qlm coefficients are redundant, negative m are obtained from positive m"""
    if m>=0:
        return qlms[...,m]
    elif (-m)%2 == 0:
        return np.conj(qlms[...,-m])
    else:
        return -np.conj(qlms[...,-m])


@jit#(['float64(int64, int64[:])'], nopython=True)
def get_w3j(l, ms):
    """Wigner 3j coefficients"""
    sm = np.sort(np.abs(ms))
    return _w3j[l//2, _w3j_m1_offset[sm[-1]] + sm[0]]
     
@jit
def wl(qlm):
    """Third order rotational invariant of the bond orientational order of l-fold symmetry
    $$ w_\ell = \sum_{m_1+m_2+m_3=0} 
			\left( \begin{array}{ccc}
				\ell & \ell & \ell \\
				m_1 & m_2 & m_3 
			\end{array} \right)
			q_{\ell m_1} q_{\ell m_2} q_{\ell m_3}
			$$"""
    l = qlm.shape[-1]-1
    w = np.zeros(qlm.shape[:-1])
    for m1 in range(-l, l+1):
        for m2 in range(-l, l+1):
            m3 = -m1-m2
            if -l<=m3 and m3<=l:
                w += get_w3j(l, np.array([m1, m2, m3])) * (get_qlm(qlm, m1) * get_qlm(qlm, m2) * get_qlm(qlm, m3)).real
    return w
    


def gG_l(pos, qlms, is_center, Nbins, maxdist):
    """Spatial correlation of the qlms (non normalized).
    For each particle i tagged as is_center, for each particle j closer than maxdist, do the cross product between their qlm and count, 
    then bin each quantity with respect to distance. 
    The two first sums need to be normalised by the last one.
    
     - pos is a Nxd array of coordinates, with d the dimension of space
     - qlms is a list of Nx(2l+1) arrays of boo coordinates for l-fold symmetry. l can be different for each item.
     - is_center is a N array of booleans. For example all particles further away than maxdist from any edge of the box.
     - Nbins is the number of bins along r
     - maxdist is the maximum distance considered"""
    for qlm in qlms:
        assert len(pos) == len(qlm)
    assert len(is_center) == len(pos)
    #conversion factor between indices and bins
    l2r = Nbins/maxdist
    #result containers
    hqQ = np.zeros((Nbins, len(qlms)))
    g = np.zeros(Nbins, int)
    #spatial indexing
    tree = KDTree(pos, 12)
    centertree = KDTree(pos[is_center], 12)
    #all pairs of points closer than maxdist with their distances in a record array
    query = centertree.sparse_distance_matrix(tree, maxdist, output_type='ndarray')
    #keep only pairs where the points are distinct
    centerindex = np.where(is_center)[0]
    query['i'] = centerindex[query['i']]
    good = query['i'] != query['j']
    query = query[good]
    #binning of distances
    rs = (query['v'] * l2r).astype(int)
    np.add.at(g, rs, 1)
    #binning of boo cross products
    pqQs = np.empty((len(rs), len(qlms)))
    for it, qlm in enumerate(qlms):
        pqQs[:,it] = boo_product(qlm[query['i']], qlm[query['j']])
    np.add.at(hqQ, rs, pqQs)
    return hqQ, g


    
# Define constants for Wigner 3j symbols
            
_w3j = np.zeros([6,36])
_w3j[0,0] = 1
_w3j[1,:4] = np.sqrt([2/35., 1/70., 2/35., 3/35.])*[-1,1,1,-1]
_w3j[2,:9] = np.sqrt([
        2/1001., 1/2002., 11/182., 5/1001.,
        7/286., 5/143., 14/143., 35/143., 5/143.,
        ])*[3, -3, -1/3.0, 2, 1, -1/3.0, 1/3.0, -1/3.0, 1]
_w3j[3,:16] = np.sqrt([
        1/46189., 1/46189.,
        11/4199., 105/46189.,
        1/46189., 21/92378.,
        1/46189., 35/46189., 14/46189.,
        11/4199., 21/4199., 7/4199.,
        11/4199., 77/8398., 70/4199., 21/4199.
        ])*[-20, 10, 1, -2, -43/2.0, 3, 4, 2.5, -6, 2.5, -1.5, 1, 1, -1, 1, -2]
_w3j[4,:25] = np.sqrt([
        10/96577., 5/193154.,
        1/965770., 14/96577.,
        1/965770., 66/482885.,
        5/193154., 3/96577., 77/482885.,
        65/14858., 5/7429., 42/37145.,
        65/14858., 0.0, 3/7429., 66/37145.,
        13/74290., 78/7429., 26/37145., 33/37145.,
        26/37145., 13/37145., 273/37145., 429/37145., 11/7429.,
        ])*[
            7, -7, -37, 6, 73, -3,
            -5, -8, 6, -1, 3, -1,
            1, 0, -3, 2, 7, -1, 3, -1,
            1, -3, 1, -1, 3]
_w3j[5,:36] = np.sqrt([
        7/33393355., 7/33393355.,
        7/33393355., 462/6678671.,
        7/33393355., 1001/6678671.,
        1/233753485., 77/6678671., 6006/6678671.,
        1/233753485., 55/46750697., 1155/13357342.,
        1/233753485., 2926/1757545., 33/46750697., 3003/6678671.,
        119/1964315., 22/2750041., 1914/474145., 429/5500082.,
        17/13750205., 561/2750041., 77/392863., 143/27500410., 2002/392863.,
        323/723695., 1309/20677., 374/144739., 143/144739., 1001/206770.,
        323/723695., 7106/723695., 561/723695., 2431/723695., 2002/103385., 1001/103385.
        ])*[
            -126, 63, 196/3.0, -7, -259/2.0, 7/3.0,
            1097/3.0, 59/6.0, -2,
            4021/6.0, -113/2.0, 3,
            -914, 1/3.0, 48, -3,
            -7/3.0, 65/3.0, -1, 3,
            214/3.0, -3, -2/3.0, 71/3.0, -1,
            3, -1/3.0, 5/3.0, -2, 1/3.0,
            2/3.0, -1/3.0, 2, -4/3.0, 2/3.0, -1]
    
_w3j_m1_offset = np.array([0,1,2,4,6,9,12,16,20,25,30], int)
