#!/home/judah/miniconda3/bin/python

from qsvt_funcs import *
import numpy as np

if __name__ == "__main__":
    # test = StaggeredPhase(1, 1)
    # print(test)
    # test = OA(2, 5)
    # test = BlockEncode(2, 5)
    # print(test)
    # print(np.real(np.round(Operator(test).data[:16, :16], 8)))
    # print(Operator(test).data)
    # print(cheby_coeff(np.cos, 4))
    d = 20               # The degree of the polynomial
    # arr = cheby_coeff(sym_log, d)
    arr = log_coeffs(d)
    # arr = approx_coeffs(sym_log, d)
    print(len(arr))

    # xx = np.linspace(-1, 1, 100000)
    # arr = chebyshev.chebfit(xx, np.cos(1 * xx), d)  # using the errorfunction to approximate sign
    print(arr)
    # assert False
    # assert False
    test_val = -0.1

    # def test(x):
    #     return chebyshev.chebval(x, arr)
    def test(x):
        return taylor_eval(x, arr)

    print(sym_log(test_val))
    print(test(test_val))
    # assert False
    djtil = int(np.ceil((d+1) / 2))
    print("dtil = ", djtil)
    params0 = np.zeros((djtil,))
    params0[0] = np.pi / 4
    print(params0)
    # print(grad(params0, test, d))
    # assert False
    # out = minimize(min_func, params0, args=(test, d),
    #                method='BFGS', jac='3-point',
    #                options={'gtol':1e-15})

    if (d % 2) == 1:
        out = minimize(min_func, params0, args=(test, d),
                       method='BFGS', jac=grad,
                       options={'gtol':1e-24})
        print(out)
        phis = np.array(list(out.x) + list(out.x)[::-1])
    else:
        out = minimize(min_func, params0, args=(test, d),
                       method='BFGS', jac=grad,
                       options={'gtol':1e-24})
        print(out)
        phis = np.array(list(out.x) + list(out.x)[::-1][1:])        
    print(np.real(output_mat(test_val, out.x, d))[0,0])
    print(sym_log(test_val))
    print(test(test_val))
    assert False
    
    # qsp = RealPart(phis, 2, 5)
    # print(qsp)
    # # qsp = QuantumSignalProcess(phis, 2, 5)
    # # print(qsp)
    # print(Operator(qsp).data[:16, :16])
    # print(BlockEncode(2, 5))
    # print(Operator(BlockEncode(2, 5)).data[:16, :16])
    # lap = Operator(BlockEncode(2, 5)).data[:16, :16]
    # print(erf(lap))
    # U, s, Vh = np.linalg.svd(lap)
    # sign_lap = U.dot(np.diag(erf(s)).dot(Vh))
    # print(sign_lap)
    
