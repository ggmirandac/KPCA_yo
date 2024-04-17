
import pandas as pd
import numpy as np

import scipy as sp
import seaborn as sns

from scipy.spatial.distance import pdist, squareform


class KernelPCAHand:
    """
    Class to perform Kernel PCA decompostion

    """
    def __init__(self, kernel = 'RBF', kernel_params = None):
        self.X = None
        self.kernel = kernel
        self.kernel_args = kernel_params


        self.K = None
        self.K_tlde = None
        self.eigen_pairs = None

        self.fitted = False

    def fit(self, X):
        '''
        Fit the KPCA to the X input getting the eigen values that are needed
        together with the eigen values
        '''
        self.X = X


        if type(self.X) is pd.core.frame.DataFrame:
            self.X = self.X.values


        K = self.K_matrix(self.X, self.kernel, self.kernel_args)

        self.K_tilde = self.genKtilde(self.X, K)
        self.eigen_pairs = self.gen_eigen(self.K_tilde)
        self.fitted = True

    def transform(self, n_of_pc = 3):
        '''
        Output the kernel princial components and the eigen values
        '''
        if self.fitted == False:
            raise ValueError('Need to fit the KPCA to data X')

        self.kpca, self.eigen_vals = self.calc_kpca(n_pc = n_of_pc)
        return self.kpca, self.eigen_vals

    def calc_kpca(self, n_pc = 3):
        '''
        Calculate the KPCA of the X matrix by using the fitted objects
        ------
        Reference:
        Bishop, C. M., & Nasrabadi, N. M. (2006). 
            Pattern recognition and machine learning (Vol. 4, No. 4, p. 738). 
            New York: springer.
        
        Wang, Q. (2012). 
            Kernel principal component analysis and its applications in face recognition and active shape models. 
            arXiv preprint arXiv:1207.3538.
        '''
        transformed = pd.DataFrame()
        eigen_vect_to_use = self.eigen_pairs[:n_pc]
        proj_df = pd.DataFrame(columns = [i for i in range(n_pc)])

        for id in range(self.X.shape[0]):
            sample = self.X[id,:]

            for pc in range(n_pc):
                a_k = eigen_vect_to_use[pc][1]
                ykx = 0
                for sample in range(self.X.shape[0]):
                    a_ki = a_k[sample]
                    Kxxi = self.K[id, sample]
                    ykx += a_ki * Kxxi
                proj_df.at[id, pc] = ykx
        eigen_val = {n: eigen_vect_to_use[n][0] for n in range(n_pc)}
        return proj_df, eigen_val

    def K_matrix(self, X, kernel, kernel_args):
        '''
        Function that generates the K matrix from the input matrix and 
        the selected kernel, with the respected arguments
        '''
        if self.kernel == 'RBF':
            if kernel_args is None:
                kernel_args = 1/X.shape[1] # 1/n_features
                print(kernel_args)
            else:
                gamma = kernel_args
            K = self.rbf_kernel(X, gamma)



        elif self.kernel == 'Linear':
            K = self.linear_kernel(X)

        elif self.kernel == 'Cosine':
            K = self.cosine_kernel(X)

        elif self.kernel == 'Poly':
            if kernel_args is None:
                gamma = 1/X.shape[1]
                c0 = 1
                degree = 2
            else:
                gamma,  degree, c0 = kernel_args

            K = self.poly_kernel(X, gamma, degree, c0)

        elif self.kernel == 'Sigmoid':
            if kernel_args is None:
                gamma = 1/X.shape[1]
                c0 = 1
            else:
                gamma, c0 = kernel_args
            K = self.sigmoid_kernel(X, gamma, c0)
        
        elif self.kernel == 'Laplacian':
            if kernel_args is None:
                gamma = 1/X.shape[1]
            else:
                gamma = kernel_args

            K = self.laplacian_kernel(X, gamma)
        
        else:
            raise ValueError(f'There is no implementation of the kernel: {kernel}')

        self.K = K


        return self.K


    def genKtilde(self, X, K):
        oneN = np.ones((len(X),len(X)))
        oneN *= 1/len(X)

        self.Ktilde = K - oneN @ K - K @ oneN + oneN @ K @ oneN

        return self.Ktilde


    def gen_eigen(self, Ktilde):
        '''
        Get the eigen values and vectors of the K matrix
        '''
        eigen_val, eigen_vect = np.linalg.eigh(Ktilde)

        self.eigen_pairs = [(eigen_val[i],eigen_vect[:,i]) for i in range(len(eigen_val))]

        self.eigen_pairs.sort(key = lambda x:x[0], reverse = True)
        return self.eigen_pairs

    def rbf_kernel(self,X, gamma):
        '''
        Defines the Radial basis function kernel
        '''
    # Error Control
        if gamma < 0:
            raise ValueError(f'gamma must be > 0: given gamma = {gamma}')
        # calculate the pairwise distance between elements of X
        # and output a square matrix
        distance = squareform(pdist(X, 'euclidean'))
        
        # generate the argument of the exponential in the RBF kernel
        inside = - gamma * distance ** 2
        # output the application of the RBF Kernel
        return np.exp(inside)


    def linear_kernel(self, X):
        '''
        Applies the Linear Kernel to X. In practise, applying this kernels
        results in the linear PCA
        '''
        return X @ X.T

    def cosine_kernel(self, X):
        '''
        Applies the cosine similarity Kernel to the input matrix
        '''
        # calculates the Frobinious norm of X
        norm_X = np.sqrt((X*X).sum(axis = 1))
        # normalizes the X matrix
        X_norm = X/norm_X[:,None]
        # returns the application of the Cosine kernel to X
        return X_norm @ X_norm.T

    def poly_kernel(self, X, gamma, degree, c0):
        '''
        Applies the polynomial kernel to the input matrix
        '''
        # calculates the kernel transformation of the matrix X
        K = (gamma * X @ X.T + c0) ** degree

        return K
    
    def sigmoid_kernel(self, X, gamma, c0):
        '''
        Applies the sigmoid kernel to the X matrix
        '''
        
        # Applies the sigmoid kernel to X and outputs it
        return np.tanh(gamma * X@X.T + c0)

    def laplacian_kernel(self, X, gamma):
        '''
        Applies the laplacian kernel to the X matrix
        '''
        # calculates the pair-wise distance using manhattan distance
        distance = squareform(pdist(X, 'cityblock'))
        # returns the kernel transformation
        return np.exp(gamma * distance)
