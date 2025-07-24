# https://github.com/NVlabs/instant-ngp/issues/747
# https://hub.docker.com/r/jjleewustledu/niftypetr-image
# https://github.com/NiftyPET/NiftyPET/issues/4 #hardware attenuation map
# niftypet

# check also parproj for simulations (additive and poisson)
# https://github.com/gschramm/2023-MIC-ImageRecon-Shortcourse/blob/88f521c3792e18c3424acc76eaf5473a90a097c7/data.py#L220
# https://github.com/gschramm/2023-MIC-ImageRecon-Shortcourse/blob/88f521c3792e18c3424acc76eaf5473a90a097c7/02_torch_em_update.py#L94

# imports & helper functions
from __future__ import print_function, division
from collections import OrderedDict
from glob import glob
from os import path
import functools
import logging
import os

# if os.getenv("OMP_NUM_THREADS", None) != "1":
#    raise EnvironmentError("should run `export OMP_NUM_THREADS=1` before notebook launch")

from miutil.plot import apply_cmap, imscroll
from niftypet import nipet
from niftypet.nimpa import getnii
from scipy.ndimage.filters import gaussian_filter
from tqdm.auto import trange
import matplotlib.pyplot as plt
import numpy as np
import pydicom


logging.basicConfig(level=logging.INFO)
print(nipet.gpuinfo())
# get all the scanner parameters
# /home/boquet/logsplines/try_nifty/amyloidPET_FBP_TP0/umap
mMRpars = nipet.get_mmrparams()
# conversion for Gaussian sigma/[voxel] to FWHM/[mm]
SIGMA2FWHMmm = (
    (8 * np.log(2)) ** 0.5 * np.array([mMRpars["Cnt"]["SO_VX" + i] for i in "ZYX"]) * 10
)


def div_nzer(x, y):
    return np.divide(x, y, out=np.zeros_like(y), where=y != 0)


def trimVol(x):
    return x[:, 100:-100, 100:-100]


def mapVol(vol, cmap=None, vmin=0, vmax=None):
    msk = ~np.isnan(vol)
    if vmin is None:
        vmin = vol[msk].min()
    if vmax is None:
        vmax = vol[msk].max()
    vol = (np.clip(vol, vmin, vmax) - vmin) / (vmax - vmin)
    return apply_cmap(**{cmap: vol}) if cmap is not None else vol


def register_spm(ref_file, mov_file, opth):
    """
    ref_file  : e.g. recon['fpet']
    mov_file  : e.g. datain['T1nii']
    """
    from spm12.regseg import coreg_spm, resample_spm

    reg = coreg_spm(ref_file, mov_file, outpath=opth, save_arr=False, save_txt=False)
    return getnii(
        resample_spm(
            ref_file,
            mov_file,
            reg["affine"],
            outpath=opth,
            del_ref_uncmpr=True,
            del_flo_uncmpr=True,
            del_out_uncmpr=True,
        )
    )


def register_dipy(ref_file, mov_file, ROI=None):
    """
    ref_file  : e.g. recon['fpet']
    mov_file  : e.g. datain['T1nii']
    """
    from brainweb import register

    return register(
        getnii(mov_file),
        getnii(ref_file),
        ROI=ROI or ((0, None), (100, -100), (100, -100)),
    )


folderin = "/home/boquet/logsplines/try_nifty/amyloidPET_FBP_TP0"
folderout = "."  # realtive to `{folderin}/niftyout`
itr = 7  # number of iterations (will be multiplied by 14 for MLEM)
fwhm = 2.5  # mm (for resolution modelling)
totCnt = None  # bootstrap sample (e.g. `300e6`) counts

# datain
folderin = path.expanduser(folderin)

# automatically categorise the input data
datain = nipet.classify_input(folderin, mMRpars, recurse=-1)

# output path
opth = path.join(datain["corepath"], "niftyout")

datain

# hardware mu-map (bed, head/neck coils)
# mu_h = nipet.hdw_mumap(datain, [1,2,4], mMRpars, outpath=opth, use_stored=True)

# MR-based human mu-map

# UTE-based object mu-map aligned (need UTE sequence or T1 for pseudo-CT)
# mu_o = nipet.align_mumap(
#    datain,
#    scanner_params=mMRpars,
#    outpath=opth,
#    t0=0, t1=0, # when both times are 0, will use full data
#    itr=2,      # number of iterations used for recon to which registering MR/UTE
#    petopt='ac',# what PET image to use (ac-just attenuation corrected)
#    musrc='ute',# source of mu-map (ute/pct)
#    ute_name='UTE2', # which UTE to use (UTE1/2 shorter/faster)
#    verbose=True,
# )

# the same as above without any faff though (no alignment)
mu_o = nipet.obj_mumap(datain, mMRpars, outpath=opth, store=True)

# create histogram
mMRpars["Cnt"]["BTP"] = 0
mMRpars["Cnt"]["SPN"] = 11
m = nipet.mmrhist(datain, mMRpars, outpath=opth, store=True, use_stored=True)
if totCnt:
    mMRpars["Cnt"]["BTP"] = 2  # enable parametric bootstrap
    mMRpars["Cnt"]["BTPRT"] = (
        totCnt / m["psino"].sum()
    )  # ratio count level relative to the original
    m = nipet.mmrhist(
        datain, mMRpars, outpath=path.join(opth, "BTP", "%.3g" % totCnt), store=True
    )

mu_o["im"].shape
imscroll(mu_o["im"], cmap="bone")
plt.imshow(mu_o["im"][70, ...])
"sinos" in datain

# sinogram index (<127 for direct sinograms, >=127 for oblique sinograms)
imscroll(
    [m["psino"], m["dsino"]],
    titles=[
        "Prompt sinogram (%.3gM)" % (m["psino"].sum() / 1e6),
        "Delayed sinogram (%.3gM)" % (m["dsino"].sum() / 1e6),
    ],
    cmap="inferno",
    # colorbars=[1]*2,
    fig=plt.figure(figsize=(9.5, 3.5), tight_layout=True, frameon=False),
)
axs = plt.gcf().axes
axs[-1].set_xlabel("bins")
[i.set_ylabel("angles") for i in axs]

# constants: https://github.com/NiftyPET/NIPET/blob/e39dcaa2040ab3ed3b8cb85b7baf788f956bfcf2/niftypet/nipet/include/def.h#L41
yo = np.fromfile(
    "amyloidPET_FBP_TP0/LM/17598013_1946_20150604155500.000000.bf", dtype=np.uint16
)
yo = np.fromfile(
    "amyloidPET_FBP_TP0/LM/17598013_1946_20150604155500.000000.bf", dtype=np.uintc
)
yo = np.fromfile(
    "amyloidPET_FBP_TP0/LM/17598013_1946_20150604155500.000000.bf", dtype=np.uint32
)
# yo = np.fromfile('amyloidPET_FBP_TP0/LM/17598013_1946_20150604155500.000000.bf', dtype=np.intc)
np.max(yo)
(8 * 56 * 8 * 8) ** 2
ahaa = np.unique(yo, return_counts=True)
hu = np.sort(ahaa[1])
plt.hist(hu[:-4], bins=1000)
import nibabel

phantom = nibabel.loadsave.load(
    "amyloidPET_FBP_TP0/LM/17598013_1946_20150604155500.000000.dcm"
)

m["psino"].shape
plt.imshow(m["psino"][:, 120, :])
plt.imshow(m["psino"][:, :, 120])
plt.imshow(m["psino"][100, :, :])
plt.imshow(m["psino"][100, 100:200, 100:200])
m.keys()
m["pssr"].shape
plt.imshow(m["pssr"][100, :, :])
# Number of slices accounted in sino
mMRpars["Cnt"]["SPN"]
# Bootstrapping
mMRpars["Cnt"]["BTP"]
# Data
nipet.lm.mmr_lmproc.lminfo(datain["lm_bf"])
# Angles
mMRpars["Cnt"]["NSANGLES"]
# Distances
mMRpars["Cnt"]["NSBINS"]
# NSBINANGS is NSBINS*NSANGLES
mMRpars["Cnt"]["NSANGLES"] * mMRpars["Cnt"]["NSBINS"]
# Slices
mMRpars["Cnt"]["NSEG0"]
mMRpars["Cnt"]["NSN1"]  # 4084
mMRpars["Cnt"]["NSN11"]
#
# BTHREADS 10
# NTHREADS 256
mMRpars["Cnt"]["BPE"]

#
8 * 56 * 8 * 8
(8 * 56 * 8 * 8) ** 2
127 * mMRpars["Cnt"]["NSANGLES"] * mMRpars["Cnt"]["NSBINS"]
# eight rings with 56 detectors blocks per ring, each consisting of 8×8 arrays of 4×4×20 mm3crystals read out by a 3×3 array of APDs
2 * 344 * 0.4

if True:
    ma = nipet.mmrhist(
        datain, mMRpars, t0=3000, t1=3601, outpath=opth, store=True, use_stored=True
    )
    plt.imshow(ma["pssr"][100, :, :])
    ma["pssr"][100, :, :]
    np.sum(ma["pssr"][100, :, :])
    np.max(ma["pssr"][100, :, :])
    ma["psino"][100, :, :]
    plt.imshow(ma["psino"][100, :, :])
    np.sum(ma["psino"][100, :, :])
    np.max(ma["psino"][100, :, :])
#### MLEM
## Attenuation, Normalisation & Sensitivity

# https://github.com/NiftyPET/NiftyPET/issues/4
# A = nipet.frwd_prj(mu_h['im'] + mu_o['im'], mMRpars, attenuation=True)
A = nipet.frwd_prj(mu_o["im"], mMRpars, attenuation=True)
N = nipet.mmrnorm.get_norm_sino(datain, mMRpars, m)
AN = A * N
sim = nipet.back_prj(AN, mMRpars)
msk = (
    nipet.img.mmrimg.get_cylinder(
        mMRpars["Cnt"], rad=29.0, xo=0.0, yo=0.0, unival=1, gpu_dim=False
    )
    <= 0.9
)


A = nipet.frwd_prj(mu_o["im"], mMRpars, attenuation=True)
N = nipet.mmrnorm.get_norm_sino(datain, mMRpars, m)
AN = A * N
N.shape
A.shape
sim.shape
plt.imshow((N)[90, :, :], cmap="inferno")
plt.imshow((A)[90, :, :], cmap="inferno")
plt.imshow((N * A)[90, :, :], cmap="inferno")
plt.imshow((sim)[63, :, :])
plt.colorbar()

plt.plot((N)[50, 120, :])
plt.plot((N)[50, 120, 100:200])
np.min(N)
plt.plot((A)[50, 120, :])
plt.plot((A)[50, 120, 100:200])
np.min(A)
plt.plot((A * N)[50, 120, :])
plt.plot((A * N)[50, 120, 100:200])
np.min(A * N)

plt.imshow(m["psino"][70, :, :])
plt.imshow((N)[70, :, :])
plt.imshow((A)[70, :, :])
plt.imshow(m["psino"][90, :, :])
plt.imshow((N)[90, :, :])
plt.imshow((A)[90, :, :])

t0 = 3500
t1 = 3600
m = nipet.mmrhist(
    datain, mMRpars, t0=t0, t1=t1, outpath=opth, store=True, use_stored=True
)
## Randoms
r = nipet.randoms(m, mMRpars)[0]
print("Randoms: %.3g%%" % (r.sum() / m["psino"].sum() * 100))

## Scatter

# One OSEM iteration estimate (implicitly using voxel-driven scatter model)
# eim = nipet.mmrchain(datain, mMRpars, mu_h=mu_h, mu_o=mu_o, histo=m, itr=1, outpath=opth)['im']
eim = nipet.mmrchain(
    datain, mMRpars, mu_h=None, mu_o=mu_o, histo=m, itr=1, outpath=opth
)["im"]

# Recalculate scatter
# s = nipet.vsm(datain, (mu_h['im'], mu_o['im']), eim, mMRpars, m, r)
s = nipet.vsm(datain, (np.zeros_like(mu_o["im"]), mu_o["im"]), eim, mMRpars, m, r)
print("Scatter: %.3g%%" % (s.sum() / m["psino"].sum() * 100))

imscroll(
    OrderedDict([("Prompts", m["psino"]), ("Delayed", m["dsino"]), ("Attenuation", A)]),
    cmap="inferno",
    fig=plt.figure(figsize=(9.5, 3), tight_layout=True, frameon=False),
)
imscroll(
    OrderedDict([("Scatter", s), ("Randoms", r), ("Normalisation", N)]),
    cmap="inferno",
    fig=plt.figure(figsize=(9.5, 3), tight_layout=True, frameon=False),
)


## MLEM with RM
SIGMA2FWHMmm = np.array([4.78322822, 4.91276687, 4.91276687])
SIGMA2FWHMmm *= 10
psf = functools.partial(gaussian_filter, sigma=fwhm / SIGMA2FWHMmm)
sim_inv = div_nzer(1, psf(sim))
sim_inv[msk] = 0
rs_AN = div_nzer(r + s, AN)
recon_mlem = [np.ones_like(sim)]
for k in trange(itr * 14, desc="MLEM"):
    fprj = nipet.frwd_prj(psf(recon_mlem[-1]), mMRpars) + rs_AN
    recon_mlem.append(
        recon_mlem[-1]
        * sim_inv
        * psf(nipet.back_prj(div_nzer(m["psino"], fprj), mMRpars))
    )


# central slice across iterations
imscroll(
    np.asanyarray([trimVol(i) for i in recon_mlem[1::14]]),
    titles=["iter %d" % i for i in range(1, len(recon_mlem), 14)],
    cmap="magma",
    fig=plt.figure(figsize=(9.5, 2), tight_layout=True, frameon=False),
)


len(recon_mlem)
recon_mlem[85].shape
trimVol(recon_mlem[85])[63, :, :].shape
plt.imshow(recon_mlem[85][60, :, :])
plt.imshow(recon_mlem[85][63, :, :], cmap="magma")
plt.imshow(trimVol(recon_mlem[85])[63, :, :], cmap="magma")
# np.save("recon_mlem_zenodo_data.npy", np.array(recon_mlem))
recon = np.load("recon_mlem_zenodo_data.npy")
plt.imshow(trimVol(recon[85])[63, :, :], cmap="magma")


# The code below provides full image reconstruction for the last 10 minutes of the acquisition to get an estimate of the amyloid load through the ratio image (SUVr).

# recon = nipet.mmrchain(
#     datain, mMRpars,
#     frames = ['timings', [3000, 3600]],
#     mu_h = muhdct,
#     mu_o = muodct,
#     itr=4,
#     fwhm=0.0,
#     outpath = opth,
#     fcomment = 'niftypet-recon',
#     store_img = True)


fcomment = f"_fwhm-{fwhm}_recon"
outpath = path.join(opth, folderout)
recon = glob(f"{outpath}/PET/single-frame/a_t-*-*sec_itr-{itr}{fcomment}.nii.gz")
t0 = 3500
t1 = 3600
m = nipet.mmrhist(
    datain, mMRpars, t0=t0, t1=t1, outpath=opth, store=True, use_stored=True
)
recon = nipet.mmrchain(
    datain,
    mMRpars,
    frames=["timings", [3000, 3600]],
    itr=itr,
    # store_itr=range(itr),
    histo=m,
    mu_h=None,
    mu_o=mu_o,
    psf=fwhm,
    # recmod=1,  # no scatter & rand
    outpath=outpath,
    fcomment=fcomment,
    store_img=True,
)

vol = recon["tuple"][0]
# for k in range(32, 122):
for k in [60, 64, 70, 89, 93, 98, 107]:
    print(k)
    plt.imshow(vol[k, :, :], cmap="magma")
    plt.show()
plt.imshow(m["psino"][93, 100:200, 100:200])
plt.imshow(m["psino"][93, ...])

mMRpars["txLUT"].keys()
mMRpars["txLUT"]["cij"].shape

mMRpars["axLUT"].keys()
mMRpars["axLUT"]["sn1_ssrb"]

#####OTHER EXAMPLE


logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.DEBUG)
print(nipet.gpuinfo())
# get all the scanner parameters
# /home/boquet/logsplines/try_nifty/amyloidPET_FBP_TP0/umap
mMRpars = nipet.get_mmrparams()
# conversion for Gaussian sigma/[voxel] to FWHM/[mm]
SIGMA2FWHMmm = (
    (8 * np.log(2)) ** 0.5 * np.array([mMRpars["Cnt"]["SO_VX" + i] for i in "ZYX"]) * 10
)


folderin = "amyloidPET_FBP_TP0"
folderout = "."  # realtive to `{folderin}/niftyout`
itr = 7  # number of iterations (will be multiplied by 14 for MLEM)
fwhm = 2.5  # mm (for resolution modelling)
totCnt = None  # bootstrap sample (e.g. `300e6`) counts

# datain
folderin = path.expanduser(folderin)

# automatically categorise the input data
datain = nipet.classify_input(folderin, mMRpars, recurse=-1)

# output path
opth = path.join(datain["corepath"], "niftyout")

datain

# obtain the hardware mu-map (the bed and the head&neck coil)
# muhdct = nipet.hdw_mumap(datain, [1,2,4], mMRpars, outpath=opth, use_stored=True)

# obtain the MR-based human mu-map
muodct = nipet.obj_mumap(datain, mMRpars, outpath=opth, store=True)

recon = nipet.mmrchain(
    datain,
    mMRpars,
    frames=["timings", [1000, 3600]],
    mu_h=None,
    mu_o=muodct,
    itr=7,
    fwhm=0.0,
    recmod=1,
    outpath=opth,
    fcomment="niftypet-recon",
    store_img=True,
)


vol = recon["tuple"][0]
# for k in range(32, 122):
for k in [60, 64, 70, 89, 93, 98, 107]:
    print(k)
    plt.imshow(vol[k, :, :], cmap="magma")
    plt.show()
plt.imshow(m["psino"][93, 100:200, 100:200])
plt.imshow(m["psino"][93, ...])

hh = nipet.frwd_prj(vol, mMRpars)  # , attenuation=True
plt.imshow(hh[93, :, :])
mMRpars
hh = nipet.frwd_prj(
    np.ones_like(vol), mMRpars
)  # , attenuation=True #with attenuation it does the integral
plt.imshow(hh[93, :, :])
N = nipet.mmrnorm.get_norm_sino(datain, mMRpars, m)
plt.imshow(N[93, :, :])
sho = 1.0 / hh[93, :, :]
sho[sho == np.inf] = 0
plt.imshow(sho)

normcomp, _ = nipet.mmrnorm.get_components(datain, mMRpars["Cnt"])
normcomp["geo"].shape
normcomp["cinf"].shape
normcomp.keys()
m["buckets"]


# m = nipet.mmrhist(datain, mMRpars, outpath=opth, store=True, use_stored=True)
if False:
    t0 = 3000
    t1 = 3601
    ma = nipet.mmrhist(datain, mMRpars, t0=t0, t1=t1, outpath=opth)
    A = nipet.frwd_prj(mu_o["im"], mMRpars, attenuation=True)
    N = nipet.mmrnorm.get_norm_sino(datain, mMRpars, ma)
    np.save("normalization_3000_3601.npy", N)
    np.save("attenuation.npy", A)
    np.save("sinogram_3000_3601.npy", ma["psino"])
    np.save("ssr_sinogram_3000_3601.npy", ma["pssr"])
    recon = nipet.mmrchain(
        datain,
        mMRpars,
        frames=["timings", [t0, t1]],
        itr=4,
        mu_h=None,
        mu_o=mu_o,
        fwhm=0,
        psf=None,
        recmod=1,  # no scatter & rand
        outpath=opth,
    )
    np.save("reconstruction_3000_3601.npy", recon["im"])

    t0 = 0
    t1 = 3601
    ma = nipet.mmrhist(datain, mMRpars, t0=t0, t1=t1, outpath=opth)
    A = nipet.frwd_prj(mu_o["im"], mMRpars, attenuation=True)
    N = nipet.mmrnorm.get_norm_sino(datain, mMRpars, ma)
    np.save("normalization_0000_3601.npy", N)
    np.save("sinogram_0000_3601.npy", ma["psino"])
    np.save("ssr_sinogram_0000_3601.npy", ma["pssr"])
    recon = nipet.mmrchain(
        datain,
        mMRpars,
        frames=["timings", [t0, t1]],
        itr=7,
        mu_h=None,
        mu_o=mu_o,
        fwhm=0,
        psf=None,
        recmod=1,  # no scatter & rand
        outpath=opth,
    )
    np.save("reconstruction_0000_3601.npy", recon["im"])

    vol = recon["im"]
    # for k in range(32, 122):
    for k in [60, 64, 70, 89, 93, 98, 107]:
        print(k)
        plt.imshow(vol[k, :, :], cmap="magma")
        plt.show()
