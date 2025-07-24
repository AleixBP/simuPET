from muPET import array_lib as np
from plt import plt


class matrix_weighting:
    def __init__(self, A, diag=True):
        self.A = A.ravel()
        # multiplication the "diagonal" is a matrix that multiplies elementwise 2D sino instead of a tensor
        self.diag = diag
        self.lip = np.max(A)
        self.eigen_min = np.min(A)
        self.dim = (self.A).size

    def __call__(self, sino):
        if self.diag:
            return self.A * sino

    def adjoint(self, sino):
        if self.diag:
            return self.A * sino  # transpose?


class LinOpGrad:

    def __init__(self, shp, dtype=np.float64):
        import pylops as pl  # pylops supports both numpy and cupy, pylops_gpu for pytorch

        self.shp = shp
        self.ndim = len(shp)
        self.Grad = pl.Gradient(
            shp, kind="forward", dtype=dtype
        )  # required flattened inputs for evaluations and returns flattened outputs
        # This class does nothing but change rmatvec for adjoint

    def __call__(self, x):
        return self.Grad.matvec(x)  # self.Grad(x)

    def adjoint(self, x):
        return self.Grad.rmatvec(x)  # self.Grad.adjoint(x) to create hermitian adjoint
