from muPET import array_lib as np
from plt import plt


def mse(f, f_ref, normalize=False, db=False):
    ms = np.sum((f-f_ref)**2)

    ms /= np.sum(f_ref**2) if normalize else f_ref.size
    if db: ms = 10.*np.log10(ms)

    return ms


def rmse(f, f_ref, normalize=False, db=False):
    ms = mse(f, f_ref, normalize=normalize)
    return 5.*np.log10(ms) if db else np.sqrt(ms)


def psnr(f, f_ref):
    rmin = np.min(f_ref)
    range = np.max(f_ref)-rmin*(rmin<0)

    return 10.*np.log10( mse(f, f_ref)**2 / range)


def mutual_info(f, f_ref, bins=100):
    from scipy.stats import entropy #cupy-compatible but not?

    hist_values, _ = np.histogramdd([f.ravel(), f_ref.ravel()],
                                    bins=bins, density=True)

    return ( entropy(np.sum(hist_values, axis=0).get()) + entropy(np.sum(hist_values, axis=1).get()) ) \
           / entropy(hist_values.ravel().get())


def rescale_and_compare(f, f_ref, compare, rescale=None, **kwargs):
    if rescale is None: rescale = lambda x: x/np.sum(x)

    nf = rescale(f)
    nf_ref = rescale(f_ref)

    return compare(nf, nf_ref, **kwargs)


def compare_plots(f, f_ref, compare=None, rescale=None, single_cb=True):

    if rescale is None: rescale = lambda x: x
    if compare is None: compare = lambda x,y : np.abs(x-y)

    if not isinstance(f, list): f = list(f)

    ncols = 1+len(f)
    nrows = 2
    magni = 5
    fig = plt.figure(figsize=(magni*ncols, magni*nrows), constrained_layout=True)
    

    images = [rescale(im) for im in ([f_ref] + f)]
    diffs = [compare(im, images[0]) for im in images]

    images = np.array(images)
    vmin = np.min(images).item(); vmax = np.max(images).item()

    diffs = np.array(diffs)
    vmind, vmaxd = (vmin, vmax) if single_cb else (np.min(diffs).item(), np.max(diffs).item())
    
    for index in range(1, ncols+1):
        fig.add_subplot(nrows, ncols, index)
        im = plt.imshow(images[index-1], vmin=vmin, vmax=vmax)

        fig.add_subplot(nrows, ncols, index+ncols)
        df = plt.imshow(diffs[index-1], vmin=vmind, vmax=vmaxd)
    
    axs = fig.axes

    sz = nrows*ncols
    if single_cb:
        fig.colorbar(im, ax=[axs[sz-2], axs[sz-1]], location="right", fraction=0.05)# pad=0.04)
    else:
        fig.colorbar(im, ax=axs[sz-2], fraction=0.05) #[axs[0], axs[1]]
        fig.colorbar(df, ax=axs[sz-1], fraction=0.05)

    for ax in axs: ax.axis('off')
    plt.show()

    return fig

## Measuring PSFs
def moving_average(x, w, cond="same"): #"valid"
    return np.convolve(x, np.ones(w), cond) / w


def slice_psf(psf):
    max_idxs = np.unravel_index(np.argmax(psf), psf.shape)
    return np.abs( psf[max_idxs[-2], :].copy() ), psf[max_idxs], max_idxs


def fwhm_from_psf(psf, smooth=None, interp=False, plot=False):

    if psf.ndim==1:
        psf_x = psf
        max_idx = np.argmax(psf_x)
        max_val = psf_x[max_idx]
    else:
        psf_x, max_val, max_idxs = slice_psf(psf)
        max_idx = max_idxs[-1]

    if smooth not in (None, 0):
        psf_x = moving_average(psf_x, smooth)
        max_idx = np.argmax(psf_x)
        max_val = psf_x[max_idx]

    psf_x_half = np.abs(psf_x-max_val/2.)
    #halfmaxs = np.argpartition(psf_x_half, 1)[:2] #not robust to noise
    #fwhm = np.abs(np.diff(halfmaxs))
    if not interp:
        fwhm = 2.*np.abs(max_idx-np.argmin(psf_x_half))
    else:
        mi = np.argmin(psf_x_half)
        # cand = mi-1 if psf_x_half[mi-1]<psf_x_half[mi+1] else mi+1
        # signs should be opposite to move in the direction that compensates
        cand = mi-1 if (psf_x[mi]-max_val/2.)*(psf_x[mi+1]-psf_x[mi])>0 else mi+1
        if cand>mi: cand, mi = mi, cand 
        hm = cand + (max_val/2.-psf_x[cand])/(psf_x[mi] - psf_x[cand])
        fwhm = 2.*np.abs(max_idx-hm)

    if plot:
        plt.plot(psf_x); plt.show()
        plt.plot(np.abs(psf_x-max_val/2.)); plt.show()

    return float(fwhm)


def histo_fwhm(psf, pixel_res, offs=None):
    #fwhm from variance

    psf_x, _, max_idxs = slice_psf(psf)

    psf_x = np.abs( psf[max_idxs[-2], :].copy() )

    if offs is None: offs = round(1./(3*pixel_res))
    x = psf_x[max_idxs[-1]-offs : max_idxs[-1]+offs]
    mids = np.arange(len(x))-max_idxs[-1]
    mean = np.average(mids, weights=x)
    var = np.average((mids - mean)**2, weights=x)
    fwhm = np.sqrt(8*np.log(2)*var)

    return pixel_res*fwhm #or scale mids


def image_to_mm(y, x, img, boxes):
    
    N, M = img.shape if hasattr(img, "shape") else img
    xnew =  x*2*boxes.Rx/M - boxes.Rx
    ynew = boxes.Ry - y*2*boxes.Ry/N 

    return np.array([xnew, ynew])