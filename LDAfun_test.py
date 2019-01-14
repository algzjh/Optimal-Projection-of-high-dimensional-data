import numpy as np


def lda(c0, c1):
    """
    pass two class data to the function
    return the optimal projection direction using LDA
    """
    # get the basic information
    dim = c0.shape[1]  # the dim is the number of features, for instance, the dim of iris is 4
    c = np.concatenate((c0, c1))  # vertical stacking, default axis is 0
    overall_mean = np.mean(c, axis=0)
    print("overall_mean: ", overall_mean)
    num0 = c0.shape[0]  # the number of class-0
    num1 = c1.shape[1]  # the number of class-1

    # Computing the mean vectors
    mean0_vectors = []
    mean1_vectors = []
    mean0_vectors = np.mean(c0, axis=0)
    mean1_vectors = np.mean(c1, axis=0)
    print("mean0_vectors: ", mean0_vectors)
    print("mean1_vectors: ", mean1_vectors)

    # Computing the Scatter Matrices
    # Within-class scatter matrix S_W

    S_W = np.zeros((dim, dim))
    class0_sc_mat = np.zeros((dim, dim))
    mv = mean0_vectors.reshape(dim, 1)
    print("mv: ", mv)
    for row in c0:
        row = row.reshape(dim, 1)
        # print("row-mv: \n", row-mv)
        class0_sc_mat += (row-mv).dot((row-mv).T)
        # print("class0_sc_mat: ", class0_sc_mat)
    S_W += class0_sc_mat
    print("within-class Scatter Matrix: \n", S_W)

    class1_sc_mat = np.zeros((dim, dim))
    mv = mean1_vectors.reshape((dim, 1))
    print("mv: ", mv)
    for row in c1:
        row = row.reshape(dim, 1)
        class1_sc_mat += (row-mv).dot((row-mv).T)
    S_W += class1_sc_mat
    print("within-class Scatter Matrix: \n", S_W)

    # Between-class scatter matrix S_B

    S_B = np.zeros((dim, dim))
    print("overall_mean: ", overall_mean)
    overall_mean = overall_mean.reshape(dim, 1)
    mean_vec = mean0_vectors.reshape(dim, 1)
    S_B += num0*(mean_vec-overall_mean).dot((mean_vec-overall_mean).T)
    mean_vec = mean1_vectors.reshape(dim, 1)
    S_B += num1*(mean_vec-overall_mean).dot((mean_vec-overall_mean).T)
    print("between-class Scatter Matrix: \n", S_B)

    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    # for i in range(len(eig_vals)):
    #     eigvec_sc = eig_vecs[:, i].reshape(dim, 1)
    #     print("\nEigenvector {}: \n{}".format(i+1, eigvec_sc.real))
    #     print("Eigenvalue {:}: {:.2e}".format(i+1, eig_vals[i].real))

    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

    # print("Eigenvalues in decreasing order:\n")
    # for i in eig_pairs:
    #     print(i[0], i[1])
    #
    # print("Variance explained:\n")
    # eigv_sum = sum(eig_vals)
    # for i,j in enumerate(eig_pairs):
    #     print("eigenvalue {0}: {1:.6%}".format(i+1, (j[0]/eigv_sum).real))
    #
    # print("eig: ", eig_pairs[0][1])
    return np.real(eig_pairs[0][1])
