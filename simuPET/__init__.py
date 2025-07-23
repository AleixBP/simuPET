if True:
    import cupy as array_lib
    import cupyx.scipy as scipy_lib
    import cupyx.scipy.ndimage as scipy_lib_ndimage
    import cupyx.scipy.signal as scipy_lib_sg
    import cupyx.scipy.linalg as scipy_lib_linalg

else:
    import numpy as array_lib
    import scipy as scipy_lib
    import scipy.ndimage as scipy_lib_ndimage
    import scipy.signal as scipy_lib_sg
    import scipy.linalg as scipy_lib_linalg