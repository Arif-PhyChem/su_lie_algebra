import numpy as np

def generate_su_basis(N: int):

    basis_matrices = []

    # Generate the state vectors |m> and |n> for 1 <= m, n <= N
    states = [np.zeros((N, 1), dtype=complex) for _ in range(N)]
    for i in range(N):
        states[i][i] = 1  # Creating the state vectors |i>

    # 1. Symmetric matrices (S_{mn}^+)
    for m in range(N):
        for n in range(m + 1, N):
            mat = np.zeros((N, N), dtype=complex)
            mat = 0.5 * (np.outer(states[m], states[n].conj().T) + np.outer(states[n], states[m].conj().T))
            basis_matrices.append(mat)

    # 2. Asymmetric matrices (S_{mn}^-)
    for m in range(N):
        for n in range(m + 1, N):
            mat = np.zeros((N, N), dtype=complex)
            mat = -1j * 0.5 * (np.outer(states[m], states[n].conj().T) - np.outer(states[n], states[m].conj().T))
            basis_matrices.append(mat)

    # 3. Diagonal matrices (S_n) -- starting from n = 2 to N
    for n in range(1, N):  # Start from 2 to N (1 is added inside equations for this sake)
        mat = np.zeros((N, N), dtype=complex)
        
        # Sum for k=1 to n-1
        for k in range(n):  # 0 <= k <= n-1, corresponding to |k><k|
            mat += np.outer(states[k], states[k].conj().T)  # |k><k|

        # Add (1-n)|n><n|
        mat += (1 - (n + 1)) * np.outer(states[n], states[n].conj().T)
        
        # Normalization factor
        norm_factor = np.sqrt(1 / (2 * (n + 1) * n))
        mat *= norm_factor
        
        basis_matrices.append(mat)


    return basis_matrices

