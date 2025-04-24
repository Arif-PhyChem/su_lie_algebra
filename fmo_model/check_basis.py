import numpy as np

def check(su_basis: np.ndarray):
    # Confirmation checks
    traceless = True
    normalized = True
    orthogonal = True
    
    # Check traceless and normalization
    for i, B_i in enumerate(su_basis):
        trace_Bi = np.trace(B_i)
        if not np.isclose(trace_Bi, 0):
            print(f"Matrix B_{i} is not traceless: Tr(B_{i}) = {trace_Bi}")
            traceless = False
    
    # Check orthogonality
    for i, B_i in enumerate(su_basis):
        for j, B_j in enumerate(su_basis):
            if i != j:
                inner_product = np.trace(B_i @ B_j.conj().T)
                if not np.isclose(inner_product, 0):
                    print(f"Matrices B_{i} and B_{j} are not orthogonal: Tr(B_{i} B_{j}) = {inner_product}")
                    orthogonal = False
    
    # Print results of the checks
    if traceless:
        print("All matrices are traceless.")
    else:
        print("Some matrices are not traceless.")
    
    if orthogonal:
        print("All matrices are orthogonal under the trace inner product.")
    else:
        print("Some matrices are not orthogonal.")


    #Checks if the basis matrices satisfy the Casimir operator condition.
    
    N = su_basis[0].shape[0]  # Dimension of the matrices

    sum_of_squares = sum(np.dot(mat, mat.conj().T) for mat in su_basis)

    casimir_value = (N**2 - 1) / (2 * N) * np.eye(N)
    print(sum_of_squares, casimir_value)
    # Check if the sum of squares is approximately equal to the Casimir value
    if np.allclose(sum_of_squares, casimir_value):
        print("The basis matrices satisfy the Casimir operator condition.")
    else:
        print("The basis matrices do not satisfy the Casimir operator condition.")
