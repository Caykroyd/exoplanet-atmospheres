import numpy as np
import numpy.ma as ma # masked arrays

def radtransf(tau, I0, S):
    assert len(tau) == len(S) + 1

    delta_tau = tau[-1] - tau # optical depth from current position to end

    dI = S * (np.exp(-delta_tau[1:]) - np.exp(-delta_tau[:-1]))

    return I0 * np.exp(-delta_tau[0]) #+ np.sum(dI, axis=0)

def radtransf_ma(I0, S, tau, cells_in_ray):
    '''
    Inputs:
        I0  [M, N, FREQ]
        S   [M, N, CELLS, FREQ]
        tau [M, N, POS  , FREQ]
        cells_in_ray [M, N]
    Output:
        I    [M, N, FREQ]
    '''
    M, N, CELLS, FREQ = S.shape
    assert I0.shape == (M, N, FREQ),f"Shapes do not correspond! {I0.shape}, {(M, N, FREQ)}"
    assert tau.shape == (M, N, CELLS+1, FREQ),f"Shapes do not correspond! {tau.shape}, {(M, N, CELLS+1, FREQ)}"
    assert cells_in_ray.shape == (M, N),f"Shapes do not correspond! {cells_in_ray.shape}, {(M, N)}"

    dI = S * np.diff(np.exp(-tau), axis = -2)

    # take the elements (cells_in_ray-1) from tau.
    # delta_tau = tau[(*np.ix_(np.arange(M), np.arange(N)), cells_in_ray, None)].squeeze(axis=-2)
    i0, i1 = ma.notmasked_edges(tau, axis = -2)
    delta_tau = (tau[i1] - tau[i0]).reshape(M, N, FREQ)

    assert np.all(delta_tau >= 0)
    print(f'Radiative transfer magnitudes:')
    print('\t', 'I_0', f'{np.abs(I0).max()}')
    print('\t', 'exp(-tau)', f'{np.exp(-delta_tau).min()}')
    print('\t', 'I_tau', f'{np.abs(I0 * np.exp(-delta_tau)).max()}')
    print('\t', 'dtau', f'{np.abs(np.diff(np.exp(-tau), axis = -2)).max()}')
    print('\t', 'S', f'{np.abs(S).max()}')
    print('\t', 'dI', f'{np.abs(dI).max()}')
    return I0 * np.exp(-delta_tau) + np.sum(dI, axis=-2)
