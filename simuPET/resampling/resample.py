from simuPET import array_lib as np
from plt import plt


def rescale(x):
    minim = np.min(x, axis=0)
    maxim = np.max(x, axis=0)
    return (x-minim)/(maxim-minim), minim, maxim


def unscale(rescaled, minim=0, maxim=1, original=None):
    if original is not None:
        minim = np.min(original)
        maxim = np.max(original)
    return rescaled*(maxim-minim)+minim


class rescaler:
    def __init__(self, x, a=0, b=1, minim=None, maxim=None):
        self.minim = np.min(x, axis=0) if minim is None else minim
        self.maxim = np.max(x, axis=0) if maxim is None else maxim
        self.maxmin = self.maxim-self.minim

        self.a = a
        self.b = b
        self.ba = b-a

    def __call__(self, x):
        return self.a + self.ba*(x-self.minim)/self.maxmin

    def unscale(self, x):
        return self.minim + (x-self.a)*self.maxmin/self.ba


def func_estim(xys, zs, rescale=False):
    from scipy.interpolate import LSQBivariateSpline, NearestNDInterpolator, LinearNDInterpolator, interp2d
    if not rescale:
        return LinearNDInterpolator(xys, zs, fill_value=0)
    else:
        rescale = rescaler(xys)
        interp = LinearNDInterpolator(rescale(xys), zs, fill_value=0)
        return lambda coords: rescale.unscale(interp(rescale(coords)))


def hist_dens_estim(sino_data, bins=100, rg=None, weights=None):

    hist_values, *sp = np.histogram2d(*sino_data.T, bins=bins, range=rg, weights=weights)
    sp = np.array(sp)
    sp_diff = sp[:,1]-sp[:,0]
    sp_off = np.min(sp, axis=1)#[:, np.newaxis]
    #plt.imshow(hist_values)

    def rebinning(sino):
        #sino-sino%sp_diff #change sino to sino.T to comply with dens_estim
        indexes = (( (sino.T-sp_off) // sp_diff).astype(int) + np.array([0,0], dtype=int)).T
        indexes = tuple(np.vsplit(indexes, indexes.ndim))
        return hist_values[indexes].squeeze() #* not supported

    return rebinning


def kde_dens_estim(sino_data, resize=100000, bw=None, weights=None, plot=False):
    from scipy.stats import gaussian_kde
    import numpy

    if not isinstance(type(sino_data), numpy.ndarray):
        sino_data = sino_data.get()
        if weights is not None:
            weights = weights.get()

    # Make periodic
    band = numpy.pi/8
    ss = numpy.concatenate((sino_data, sino_data[sino_data[:,0]<band]+numpy.array([numpy.pi,0])), axis=0)
    ss = numpy.concatenate((ss,        sino_data[sino_data[:,0]>(numpy.pi-band)]+numpy.array([-numpy.pi,0])), axis=0)
    if plot: plt.hist2d(*ss.T, bins=100); plt.show()
    if weights is None:
        ww=weights
    else:
        ww = numpy.concatenate((weights, weights[sino_data[:,0]<band]), axis=0)
        ww = numpy.concatenate((ww,      weights[sino_data[:,0]>(numpy.pi-band)]), axis=0)
        if plot: plt.hist2d(*ss.T, bins=100, weights=ww); plt.show()

    # Reduce size
    sl = numpy.random.randint(0,ss.shape[0], size=int(resize))
    ss=ss[sl,:]
    if plot: plt.hist2d(*ss.T, bins=20); plt.show()
    if weights is not None: 
        ww=ww[sl]

    if isinstance(type(sino_data), numpy.ndarray):
        return gaussian_kde(ss.T, bw_method=bw, weights=ww)
    else:
        return lambda x: np.array(gaussian_kde(ss.T, bw_method=bw, weights=ww)(x.get())) #bw_method=0.1,


def func_resample(sino_data, sampled_phi, sampled_s, plot=False):
    ## FUNCTION ESTIMATION
    # For function estimation: gather all repetitions and count them
    unique_sino_data, sino_counts = np.unique(sino_data, return_counts=True, axis=0)
    if plot: plt.hist2d(*unique_sino_data.T, bins=100, weights=sino_counts)
    # unique_sino_data = np.random.rand(140000,2)*np.array([np.pi, 2*np.max(s)])-np.array([0., np.max(s)]); sino_counts = np.linalg.norm(unique_sino_data, axis=1)

    # Fit function from sinogram data
    interp = func_estim(unique_sino_data, sino_counts, rescale=False)

    # Make parallel grid
    parallel_grid = np.meshgrid(sampled_phi, sampled_s) # size (2, s, phi)
    # xx, yy = parallel_grid
    
    # Sample function at parallel grid (thereby resampling on a parallel grid)
    f_parallel_sino_data = interp(*parallel_grid).T

    if plot: 
        pl=plt.imshow(f_parallel_sino_data.T, aspect="auto")
        plt.colorbar(pl)

    return f_parallel_sino_data, interp
    

def dens_resample(sino_data, sampled_phi, sampled_s, dens_estim=hist_dens_estim, args=[], plot=False):
    parallel_grid = np.meshgrid(sampled_phi, sampled_s) # size (2, s, phi)
    xx, yy = parallel_grid
    parallel_pos = np.vstack([xx.ravel(), yy.ravel()]) # size (2, s*phi)
    sino_shp = (len(sampled_phi), len(sampled_s))

    density = dens_estim(sino_data, *args)
    parallel_sino_data = density(parallel_pos).reshape(sino_shp[::-1]).T
    if plot:
        plt.imshow(parallel_sino_data.T,aspect="auto",
                   extent=[0, np.pi, np.min(sampled_s).item(), np.max(sampled_s).item()])

    return parallel_sino_data


if __name__ == '__main__':

    # Visualize simulated data
    sino_data = np.stack((phi,s), axis=1)
    pl = plt.hist2d(*sino_data.T, bins=100)

    # Make parallel grid on sinogram domain
    number_of_phi_samples = 180
    sampled_phi = np.linspace(0, np.pi, number_of_phi_samples) # no need for uniformity
    number_of_s_samples = 800 # detectors
    sampled_s = np.linspace(*np.array([np.min(s), np.max(s)]), number_of_s_samples)

    parallel_grid = np.meshgrid(sampled_phi, sampled_s) # size (2, s, phi)
    xx, yy = parallel_grid
    parallel_pos = np.vstack([xx.ravel(), yy.ravel()]) # size (2, s*phi)


    ## DENSITY ESTIMATION
    # For density estimation: sino data directly
    density = dens_estim(sino_data)
    parallel_sino_data = density(parallel_pos).reshape(sino_shp[::-1]).T
    plt.imshow(parallel_sino_data.T,aspect="auto")

    from scanner_sensitivity import xray_scanner
    weights = xray_scanner(boxes, *parallel_grid)
    w_parallel_sino_data = (parallel_sino_data.T/weights).T
    pl=plt.imshow(w_parallel_sino_data.T,aspect="auto"); plt.colorbar(pl); plt.show()

    # Weighted KDE
    w2 = xray_scanner(boxes, *sino_data.T)
    density2 = dens_estim(sino_data, resize=1e6, weights=1./w2)
    parallel_sino_data2 = density2(parallel_pos).reshape(sino_shp[::-1]).T
    pl=plt.imshow(parallel_sino_data2.T,aspect="auto"); plt.colorbar(pl); plt.show()

    # Resampling: check if rescaling works
    #sino_rescale = rescaler(sino_data)
    #np.allclose(sino_data, sino_rescale.unscale(sino_rescale(sino_data)))

    # Resampling: otherwise send uniform data
    #unif = np.zeros((number_of_phi_samples, number_of_s_samples))
    #radius = 100; center = 400 # remember singoram sampling size (proj_geom) is different from real volume geom
    #unif[:, (center-radius):(center+radius) ]=1
    #parallel_sino_data = 5*unif