import numpy as np

def radtransf(tau, I0, S):
    assert len(tau) == len(S) + 1

    delta_tau = tau[-1] - tau # optical depth from current position to end

    dI = S * (np.exp(-delta_tau[1:]) - np.exp(-delta_tau[:-1]))

    return I0 * np.exp(-delta_tau[0]) #+ np.sum(dI, axis=0)

def radtransf_block(I0, S, delta_tau, cell_blocks):
    '''
    Inputs:
        I0   [M, N, FREQ]
        S    [M, N, CELLS, FREQ]
        dtau [M, N, CELLS, FREQ]
        cell_blocks [M, N]
    Output:
        I    [M, N, FREQ]
    '''
    assert delta_tau.shape == S.shape,f"Shapes do not correspond! {delta_tau.shape}, {S.shape}"
    M, N, CELLS, FREQ = S.shape

    dI = np.zeros((M, N, CELLS, ))
    dI = S * np.diff(np.exp(-delta_tau), axis = -2)
    # We are only allowed to fill the blocks in dI up to their cell_count!!

    return I0 * np.exp(-delta_tau[...,-1,:]) + np.sum(dI, axis=-2)
