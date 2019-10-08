import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def lda(c0, c1):
    """
    type is ndarray
    shape like (200,2) and 200 is the number of samples and 2 is the number of features
    pass two class data to the function
    return the Eigenvectors using LDA
    the solve method is Eigenvalue Decomposition
    """
    # get the basic information
    dim = c0.shape[1]  # the dim is the number of features, for instance, the dim of iris is 4
    c = np.concatenate((c0, c1))  # Vertical stacking, default axis is 0
    overall_mean = np.mean(c, axis=0)  # the overall_mean of the samples
    # print("overall_mean: ", overall_mean)
    num0 = c0.shape[0]  # the number of class-0
    num1 = c1.shape[0]  # the number of class-1

    # Computing the mean vectors
    mean0_vectors = np.mean(c0, axis=0)
    mean1_vectors = np.mean(c1, axis=0)
    # print("mean0_vectors: ", mean0_vectors)
    # print("mean1_vectors: ", mean1_vectors)

    # Computing the Scatter Matrices

    # Within-class scatter matrix S_W
    S_W = np.zeros((dim, dim))
    class0_sc_mat = np.zeros((dim, dim))
    mv = mean0_vectors.reshape(dim, 1)
    # print("mv: ", mv)
    for row in c0:
        row = row.reshape(dim, 1)
        # print("row-mv: \n", row-mv)
        class0_sc_mat += (row-mv).dot((row-mv).T)
        # print("class0_sc_mat: ", class0_sc_mat)
    S_W += class0_sc_mat
    # print("within-class Scatter Matrix: \n", S_W)

    class1_sc_mat = np.zeros((dim, dim))
    mv = mean1_vectors.reshape((dim, 1))
    # print("mv: ", mv)
    for row in c1:
        row = row.reshape(dim, 1)
        class1_sc_mat += (row-mv).dot((row-mv).T)
    S_W += class1_sc_mat
    # print("within-class Scatter Matrix: \n", S_W)

    # Between-class scatter matrix S_B
    S_B = np.zeros((dim, dim))
    # print("overall_mean: ", overall_mean)
    overall_mean = overall_mean.reshape(dim, 1)
    mean_vec = mean0_vectors.reshape(dim, 1)
    S_B += num0*(mean_vec-overall_mean).dot((mean_vec-overall_mean).T)
    mean_vec = mean1_vectors.reshape(dim, 1)
    S_B += num1*(mean_vec-overall_mean).dot((mean_vec-overall_mean).T)
    # print("between-class Scatter Matrix: \n", S_B)

    eig_vals, eig_vecs = np.linalg.eig(np.linalg.pinv(S_W).dot(S_B))
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
    # return np.real(eig_pairs[:][1])

    return [i[1] for i in eig_pairs]


def genGaussianData():
    """
    this function is to generate Gaussian Data
    :return: class_sample which contains 2 class
    each has shape num[i]*2
    """
    num = [200, 200]
    mu_vec = np.array([[5.0, 11.0], [16.0, 2.0]])
    # print("mu_vec: \n", mu_vec)
    cov_mat = []
    for i in range(2):
        cov_mat.append(np.array([[1, 0], [0, 9]]))
        # print("i=%d cov_mat: \n"%i, cov_mat)
    class_sample = []
    for i in range(2):
        class_sample.append(np.random.multivariate_normal(mu_vec[i], cov_mat[i], num[i]))
        # print("i: \n", class_sample[i])
    return class_sample, num


if __name__ == "__main__":
    class_sample, num = genGaussianData()
    c0 = class_sample[0]
    c1 = class_sample[1]
    # plt.plot(c0[:, 0], c0[:, 1], 'o', markersize=8, color='b', alpha=0.3, label='class0')
    # plt.plot(c1[:, 0], c1[:, 1], '^', markersize=8, color='g', alpha=0.3, label='class1')
    mix = np.min([np.min(c0[:, 0]), np.min(c1[:, 0])])
    max = np.max([np.max(c0[:, 1]), np.max(c1[:, 1])])
    print("mix: ", mix, "max: ", max)

    discriminating_vectors = lda(c0, c1)
    print("discriminating_vectors: \n", discriminating_vectors)

    # Comparing with the lda in sklearn
    tc = np.concatenate((c0, c1))
    label = np.array([0 for i in range(200)] + [1 for i in range(200)])
    lda = LDA(solver='eigen')
    clf = lda.fit(tc, label)
    print("scalings: \n", clf.scalings_)

    # print("line: \n", discriminating_vectors[0][0], discriminating_vectors[0][1])
    x = np.linspace(mix, max)
    k1 = discriminating_vectors[0][1] / discriminating_vectors[0][0]
    # plt.plot(x, k1*x, color='r')
    k2 = discriminating_vectors[1][1] / discriminating_vectors[1][0]
    # plt.plot(x, k2*x, color='y')
    # plt.show()
    """
    because Sw^-1*Sb is not a symmetric matrix, so the eigenvectors are not necessarily orthogonal
    由于Sw-1Sb不一定是对称阵，因此得到的k个特征向量不一定正交
    """
