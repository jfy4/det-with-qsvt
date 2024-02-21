#!/home/judah/miniconda3/bin/python

import numpy as np
# import math as mt
from scipy.optimize import minimize
from functools import reduce
from scipy.special import erf


def min_func(params, targ_func, d):
    djtil = int(np.ceil((d+1) / 2))
    assert len(params) == djtil
    cheby_zeros = np.cos([(2*j-1)*np.pi / (4 * djtil)
                          for j in range(1, djtil+1)])
    if (d % 2) == 1:
        phis = np.array(list(params) + list(params)[::-1])
    else:
        phis = np.array(list(params) + list(params)[::-1][1:])
    Upis = [np.array([[np.exp(1j * phis[a]), 0],
                      [0, np.exp(-1j * phis[a])]])
            for a in range(len(phis))]
    want = 0
    start = Upis[0]
    for i in range(djtil):
        W = np.array([[cheby_zeros[i], 1j * np.sqrt(1 - cheby_zeros[i]**2)],
                      [1j * np.sqrt(1 - cheby_zeros[i]**2), cheby_zeros[i]]])
        prod_list = [W.dot(Upis[a]) for a in range(1, d)]
        one_mat = reduce(np.dot, prod_list)
        one_mat = start.dot(one_mat)
        want += np.abs(np.real(one_mat[0, 0]) - targ_func(cheby_zeros[i]))**2
    want /= djtil
    return want


def output_mat(x, params, d):
    if (d % 2) == 1:
        phis = np.array(list(params) + list(params)[::-1])
    else:
        phis = np.array(list(params) + list(params)[::-1][1:])
    Upis = [np.array([[np.exp(1j * phis[a]), 0],
                      [0, np.exp(-1j * phis[a])]])
            for a in range(len(phis))]
    start = Upis[0]
    W = np.array([[x, 1j * np.sqrt(1 - x**2)],
                  [1j * np.sqrt(1 - x**2), x]])
    prod_list = [W.dot(Upis[a]) for a in range(1, d)]
    one_mat = reduce(np.dot, prod_list)
    one_mat = start.dot(one_mat)
    return one_mat


def test(x):
    return erf(5 * x)


if __name__ == "__main__":
    d = 40
    params0 = np.zeros((int(np.ceil((d+1) / 2)),))
    params0[0] = np.pi / 4
    print(params0)
    out = minimize(min_func, params0, args=(test, d))
    print(out)
    print(output_mat(-0.1, out.x, d))
    print(test(-0.1))
          
    
