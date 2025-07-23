#### Histogram
import sys
sys.path.append('../')
sys.path.append('../../')
#from muPET import array_lib as np
import cupy as np
from plt import plt
#nvidia-smi
np.cuda.Device(0).use()

def positron(x, levin=False):
    if levin:
        k1=37.9 
        k2=3.1
        c = 0.516
        nrm = (2*c/k1 + 2*(1-c)/k2) #0.339488
        return (c*np.exp(-k1*np.abs(x))+(1-c)*np.exp(-k2*np.abs(x)))/nrm
    elif False:
        fwhm = 0.102
        loc = 0
        scale = fwhm/float(2*np.log(2))
        return np.exp(-np.abs(x-loc)/scale)/(2*scale)
    elif False:
        fwhm = 0.102
        mu = 0
        sigma = fwhm / np.sqrt(8*np.log(2))
        return np.exp(-((x-mu)**2)/(2*sigma**2)) / (sigma*np.sqrt(2*np.pi))
    elif False: #soft tissue in carter 2020
        k1 = 36.0
        k2 = 3.597
        c = 0.56
        nrm = (2*c/k1 + 2*(1-c)/k2)
        return (c*np.exp(-k1*np.abs(x))+(1-c)*np.exp(-k2*np.abs(x)))/nrm
    else: #JD water
        k1 = 23.5554
        k2 = 3.7588
        c = 0.5875
        nrm = (2*c/k1 + 2*(1-c)/k2)
        return (c*np.exp(-k1*np.abs(x))+(1-c)*np.exp(-k2*np.abs(x)))/nrm

def acollinearity(x, radius=17, gauss=True):
    if gauss: 
        nr = float(np.sqrt(2*np.pi))
        siga = 0.0044*radius/2.355
        return np.exp((-x**2)/(2*siga**2))/(siga*nr)
    else:
        xx = 1000*0.124*x/radius
        B = (346.5*(xx**2 + 1)**2 + 5330*(xx**2 + 1)- 4264)/(xx**2 + 1)**5
        M = 0
        b = 1
        m = 0
        return (b*B + m*M)
        
    
def unit_triangle(x):
    xa = np.abs(x)
    return (xa<1)*(1-xa)

def detector(x, pitch=0.1, parallax=0, radius=17, height=0.27):
    if parallax==0:
        hp = pitch/2
    else:
        hp = float((height*parallax + pitch*np.sqrt(radius**2-parallax**2))/(2*radius))
        
    return hp*unit_triangle(x/hp)

#xs = np.linspace(-50,50,100000)
step = 0.001
xs = np.arange(-10,10,step)
ranges = positron(xs)
plt.plot(xs, ranges/np.max(ranges))

pa = np.convolve(positron(xs), acollinearity(xs, gauss=True), mode="same")#, mode="valid")
plt.plot(xs, pa/np.max(pa)); plt.xlim((-0.5,0.5))
fwhm_pixels, _ = fw(pa)
fwhm_x = fwhm_pixels*step
fwhm_x

pad = np.convolve(pa, detector(xs), mode="same")
plt.plot(xs, pad/np.max(pad)); plt.xlim((-0.5,0.5))
fwhm_pixels, _ = fw(pad)
fwhm_x = fwhm_pixels*step
fwhm_x
#tikzplotlib.save("resolution_psf_theory_rev.tex")


def fwhm_vs_pitch(pitches, parallax=0, radius=17, height=0.27, step = 0.001):
    xs = np.arange(-10,10,step)
    
    fwhms = []
    aux = np.convolve(positron(xs), acollinearity(xs, radius=radius), mode="same")
    for pitch in pitches:
        func = np.convolve(aux, detector(xs, pitch, parallax, radius, height), mode="same")
        fwhm_pixels, _ = fw(func)
        fwhms.append(fwhm_pixels*step)
        
    return fwhms

pitches = np.arange(0.01, 0.5, 0.01)
fwhms = fwhm_vs_pitch(pitches, step=0.00001)
plt.plot(pitches, fwhms)
fwhms
fwhms[9]
fwhms[14]
fwhms[19]

# diminishing returns for pitch
#pitches_dif = np.arange(0.075, 0.5, 0.01)
pitches_dif = np.arange(0.035, 0.5, 0.01)
fwhms_dif = np.diff(fwhm_vs_pitch(pitches_dif, step=0.000001))
plt.plot(pitches_dif[:-1], fwhms_dif)

nmi = np.min(np.array(fwhms))
nma = np.max(np.array(fwhms))
import tikzplotlib
plt.plot(pitches, fwhms); plt.plot(pitches_dif[:-1], (fwhms_dif-np.min(fwhms_dif))*(nma-nmi)/(np.max(fwhms_dif)-np.min(fwhms_dif)) + nmi)
#tikzplotlib.save("resolution_tradeoff.tex")
#tikzplotlib.save("resolution_tradeoff2.tex")

#third derivative
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
pitches_dif = np.arange(0.01, 0.5, 0.01)
fwhms_dif = np.diff(np.diff(np.diff(fwhm_vs_pitch(pitches_dif, step=0.0000003))))
ma = 5
fwhms_dif = moving_average(fwhms_dif, ma)
plt.plot(pitches_dif[:(-3-ma+1)], fwhms_dif)
plt.plot(pitches_dif[-(-3-ma+1):], fwhms_dif)
plt.plot((pitches_dif[-(-3-ma+1):]+pitches_dif[:(-3-ma+1)])/2, fwhms_dif)
nmi = np.min(np.array(fwhms))
nma = np.max(np.array(fwhms))
plt.plot(pitches, fwhms); plt.plot(pitches_dif[:(-3-ma+1)], (fwhms_dif-np.min(fwhms_dif))*(nma-nmi)/(np.max(fwhms_dif)-np.min(fwhms_dif)) + nmi)
plt.plot(pitches, fwhms); plt.plot((pitches_dif[-(-3-ma+1):]+pitches_dif[:(-3-ma+1)])/2, (fwhms_dif-np.min(fwhms_dif))*(nma-nmi)/(np.max(fwhms_dif)-np.min(fwhms_dif)) + nmi)
#tikzplotlib.save("resolution_tradeoff_thirdderiv.tex")

def fwhm_vs_radius(radiuses, pitch=0.1, parallax=0, height=0.27):
    step = 0.001
    xs = np.arange(-10,10,step)
    
    fwhms = []
    for radius in radiuses:
        aux = np.convolve(positron(xs), detector(xs, pitch, parallax, radius, height), mode="same")
        func = np.convolve(aux, acollinearity(xs, radius=radius), mode="same")
        fwhm_pixels, _ = fw(func)
        fwhms.append(fwhm_pixels*step)
        
    return fwhms

radiuses = np.arange(1, 100, 1)
fwhms = fwhm_vs_radius(radiuses, pitch=0.01)
plt.plot(radiuses, fwhms)
fwhms


def fwhm_vs_parallax(parallaxes, pitch=0.1, radius=17, height=0.27/2, step=0.001):
    xs = np.arange(-10,10,step)
    
    fwhms = []
    aux = np.convolve(positron(xs), acollinearity(xs, radius=radius), mode="same")
    for parallax in parallaxes:
        func = np.convolve(aux, detector(xs, pitch, parallax, radius=17, height=height), mode="same")
        fwhm_pixels, _ = fw(func)
        fwhms.append(fwhm_pixels*step)
        
    return fwhms

import tikzplotlib
parallaxes = np.arange(0, 17, 1)
fwhms = fwhm_vs_parallax(parallaxes, pitch=0.1, step=0.00001)
plt.plot(parallaxes, fwhms)
#tikzplotlib.save("resolution_parallax.tex")
fwhms
print(fwhms[0], fwhms[5], fwhms[10], fwhms[15])


def fw(psf, ht=2., smooth=None, interp=False, plot=False):
    def slice_psf(psf):
        max_idxs = np.unravel_index(np.argmax(psf), psf.shape)
        return np.abs( psf[max_idxs[-2], :].copy() ), psf[max_idxs], max_idxs

    if psf.ndim==1:
        psf_x = psf
        max_idx = np.argmax(psf_x)
        max_val = psf_x[max_idx]
    else:
        max_idxs = np.unravel_index(np.argmax(psf), psf.shape)
        psf_x = (np.abs(psf[max_idxs[-2], :].copy())+np.abs(psf[max_idxs[-2]+1, :].copy())+np.abs(psf[max_idxs[-2]-1, :].copy()))/3.
        max_val = psf[max_idxs]
        #psf_x, max_val, max_idxs = slice_psf(psf)
        max_idx = max_idxs[-1]

    if smooth not in (None, 0):
        psf_x = moving_average(psf_x, smooth)
        max_idx = np.argmax(psf_x)
        max_val = psf_x[max_idx]

    psf_x_half = np.abs(psf_x-max_val/ht)
    #halfmaxs = np.argpartition(psf_x_half, 1)[:2] #not robust to noise
    #fwhm = np.abs(np.diff(halfmaxs))
    if not interp:
        fwhm = 2.*np.abs(max_idx-np.argmin(psf_x_half))
    else:
        mi = np.argmin(psf_x_half)
        # cand = mi-1 if psf_x_half[mi-1]<psf_x_half[mi+1] else mi+1
        # signs should be opposite to move in the direction that compensates
        cand = mi-1 if (psf_x[mi]-max_val/ht)*(psf_x[mi+1]-psf_x[mi])>0 else mi+1
        if cand>mi: cand, mi = mi, cand 
        hm = cand + (max_val/ht-psf_x[cand])/(psf_x[mi] - psf_x[cand])
        fwhm = 2.*np.abs(max_idx-hm)

    if plot:
        plt.plot(psf_x); plt.xlim(max_idx-20,max_idx+20); plt.show()
        #plt.plot(np.abs(psf_x-max_val/2.)); plt.show()

    return float(fwhm), psf_x