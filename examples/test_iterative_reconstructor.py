import sys

sys.path.append("../")
from simuPET import array_lib as np
from plt import plt

# RELEVANT PARAMS
# nsim, radioactivity, number_of_phi_samples, number_of_s_samples, bins, algo params
if False:
    nsim = 1
    radioactivity = 0.7 * 1e7
    number_of_phi_samples = 1000
    number_of_s_samples = 840
    bins = 400
    account_for_weights = True
    image_path = None
    alpha = 10.0
    lam = 1000
    niter = 45
    n_prox_iter = 100
elif False:
    nsim = 40
    radioactivity = 0.7 * 1e7
    number_of_phi_samples = 1649
    number_of_s_samples = 1050
    bins = 1000
    account_for_weights = True
    image_path = "../simuPET/input_data/derenzo_200mu.npy"
    alpha = 100
    lam = 500
    niter = 120
    n_prox_iter = 100
elif False:
    # https://pet-mr.github.io/brain_phantom/
    # https://zenodo.org/record/1190598
    # https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.14218
    image_path = "../simuPET/input_data/brain_phantom_y270.npy"

    diag = 1.0
    nsim = 20
    radioactivity = 0.7 * 1e7
    number_of_s_samples = 473
    number_of_phi_samples = int(number_of_s_samples * np.pi / 2)
    bins = 500
    account_for_weights = True

    alpha = 100
    lam = 1000  # 1500 higher-level smoothing #500 # amb 1500 -7.18 dB
    niter = 120
    n_prox_iter = 100
else:
    image_path = "/home/boquet/simuPET/geant/brain_sagittal700b.npy"

    diag = np.sqrt(2)
    nsim = 8
    radioactivity = 0.7 * 1e7
    number_of_s_samples = int(340 * diag)
    number_of_phi_samples = int(number_of_s_samples * np.pi / 2)
    bins = 350
    account_for_weights = False

    lam = 520
    niter = 120
    n_prox_iter = 100


## SIMULATE
from simuPET.simulations import simulator

boxes, img, detections, phi, s = simulator.simulate_simuPET(
    radioactivity=radioactivity, plot=False, image_path=((840, 840), 200, None, None)
)
np.max(detections, axis=1)
boxes, img, *_, phi, s, data_weights = simulator.iterate_simulate_simuPET(
    nsim=nsim, weights=True, radioactivity=radioactivity, image_path=image_path
)
sino_data = np.stack((phi, s), axis=1)
# sino_data = np.load("./sino_datas.npy")
pl = plt.hist2d(*sino_data.T, bins=100)


## WEIGHT
from simuPET.simulations.scanner_sensitivity import xray_scanner

data_weights = xray_scanner(boxes, *sino_data.T)
plt.show()
## Check photons and normalization
# 30*60*nsim*radioactivity/np.sum(img) #should be 1.
# area = np.prod(np.diff(boxes.domain))
# nsim*area*np.max(img) #prospective number for simulations
# nsim*area*np.sum(img)/img.size #should be the number of photons emitted (mean of the Poisson distro at least)
# (1./data_weights[data_weights!=0]).sum() #estimate of number of photons emitted
# sino_data.shape[0]/np.mean(data_weights) #more naive estimate
# sino_data.shape[0]/((1-np.exp(-0.06*boxes.layerDepth))**2) #even more naive


## RESAMPLE
from simuPET.resampling.resample import dens_resample, hist_dens_estim, kde_dens_estim

# Build grid first
number_of_phi_samples = number_of_phi_samples  ####
number_of_s_samples = number_of_s_samples  #### # detectors
domain_s = boxes.domain[0, :] * diag  # np.array([np.min(s), np.max(s)])

sampled_phi = np.linspace(
    0, np.pi, number_of_phi_samples, endpoint=False
)  # no need for uniformity
sampled_s = np.linspace(*domain_s, number_of_s_samples)  # need uniformity

# Take the chance to define vol too
spacing_of_s = 1.0  # uniform
sino_shp = (number_of_phi_samples, number_of_s_samples)
voxel_rows, voxel_cols = img.shape
vol_shp = (voxel_rows, voxel_cols)

# Compute weights on grid
from simuPET.simulations.scanner_sensitivity import xray_scanner

parallel_sino_weights = xray_scanner(boxes, *np.meshgrid(sampled_phi, sampled_s)).T
# data_weights = xray_scanner(boxes, *sino_data.T); plt.show()
if not account_for_weights:  # do not weight
    parallel_sino_weights = np.ones_like(parallel_sino_weights)
    data_weights = np.ones_like(data_weights)

# Actually resample
rg = np.array([np.array([0, np.pi]), domain_s])  # None
parallel_sino_data = dens_resample(
    sino_data,
    sampled_phi,
    sampled_s,
    hist_dens_estim,
    args=[bins, rg, 1.0 / data_weights],
)
plt.imshow(
    parallel_sino_data.T,
    aspect="auto",
    extent=[0, np.pi, np.min(sampled_s).item(), np.max(sampled_s).item()],
)
if True:
    a = np.argpartition(parallel_sino_data.ravel(), -4)[-4:]
    a = np.unravel_index(a, parallel_sino_data.shape)
    parallel_sino_data[a] = 0
    plt.imshow(
        parallel_sino_data.T,
        aspect="auto",
        extent=[0, np.pi, np.min(sampled_s).item(), np.max(sampled_s).item()],
    )
# plt.savefig("without_weighting.eps")

## RECONSTRUCT (INIT)
from simuPET.reconstruction.astra_operators import init_parallel_projector_CUPY_2D
from simuPET.reconstruction.linear_operators import matrix_weighting
from simuPET.reconstruction.functional_costs import (
    weighted_squared_l2_loss,
    squared_l2_norm,
    sum_differentiable,
    TV_cost,
    powerit_PTAP,
)

P = init_parallel_projector_CUPY_2D(
    spacing_of_s, number_of_s_samples, sampled_phi, vol_shp
)

A_weight = matrix_weighting(parallel_sino_weights)
# A_weight = matrix_weighting(np.ones_like(parallel_sino_weights))


ws_loss = weighted_squared_l2_loss(P, A=A_weight, data=parallel_sino_data.ravel())
max_eigenvalue = powerit_PTAP(P, A_weight, niter=10)
ws_loss.lip = max_eigenvalue

s_norm = squared_l2_norm(P.dim[0], alpha=10.0)


## RECONSTRUCT (RUN)
from simuPET.reconstruction.descent_algorithms import GD, APGD
from simuPET.reconstruction.descent_algorithms_backup import GD as GD2
from simuPET.reconstruction.descent_algorithms_backup import APGD as APGD2

# L2
data_plus_l2_cost = sum_differentiable(ws_loss, s_norm)
result_l22 = GD2(
    data_plus_l2_cost,
    niter=40,
    min_max=[0, np.infty],
    force_nostrong=True,
    verbose_iter=5,
).reshape(vol_shp)[::-1, :]
result_l2 = GD(data_plus_l2_cost, niter=40, min_max=[0, np.infty], verbose=5).reshape(
    vol_shp
)[::-1, :]
pl = plt.imshow(result_l2)
plt.colorbar(pl)
plt.show()

# TV
tv_cost = TV_cost(
    vol_shp, n_prox_iter=100, lam=750.01, proj=[0, np.infty], acc="None"
)  # 7000 for 100 bins, 1000 (-9.49db) or 1200 for 400
result_tv2 = APGD2(ws_loss, tv_cost, niter=40, verbose_iter=5, acc="BT").reshape(
    vol_shp
)[::-1, :]
result_tv = APGD(ws_loss, tv_cost, niter=120, verbose=5, acc_method="BT").reshape(
    vol_shp
)[::-1, :]

pl = plt.imshow(result_tv)
plt.colorbar(pl)
plt.show()
# plt.savefig("tv_brain.eps");
pl = plt.imshow(result_tv[100:200, 100:200])
pl = plt.imshow(img[100:200, 100:200])
pl = plt.imshow(result_tv[150:170, 140:160])
pl = plt.imshow(img[150:170, 140:160])
pl = plt.imshow(result_fbp[150:170, 140:160])


# FBP, BP
from simuPET.reconstruction.astra_operators import fbp_reconstruction, bp_reconstruction
import astra

proj_geom = astra.creators.create_proj_geom(
    "parallel", spacing_of_s, number_of_s_samples, sampled_phi.get()
)
vol_geom = astra.creators.create_vol_geom(vol_shp)
result_fbp = fbp_reconstruction(proj_geom, vol_geom, parallel_sino_data.get())
result_fbp = np.array(result_fbp)
pl = plt.imshow(result_fbp)
plt.colorbar(pl)
plt.show()

result_bp = bp_reconstruction(proj_geom, vol_geom, parallel_sino_data.get())
result_bp = np.array(result_bp)
pl = plt.imshow(result_bp)
plt.colorbar(pl)
plt.show()

## COMPARISON
from simuPET.reconstruction.evaluate_reconstruction import (
    compare_plots,
    rescale_and_compare,
    rmse,
    mse,
)

fig = compare_plots(
    [result_tv, result_fbp, result_bp],
    img,
    rescale=lambda x: x / np.sum(x),
    single_cb=False,
)
fig.savefig("gt_tv_fbp_bp.eps")
rescale_and_compare(result_tv, img, rmse, normalize=True)
rescale_and_compare(result_tv, img, mse, normalize=True, db=True)
rescale_and_compare(result_fbp, img, mse, normalize=True, db=True)
rescale_and_compare(
    result_tv, img, lambda x, y: np.max((x - y)[y > 0]) / np.max(y[y > 0])
)


## SCALING
avg_sens = (
    np.sum(parallel_sino_weights) / parallel_sino_weights.size
)  # average sensitivity
photon_pairs_emitted = np.sum(img)
photon_pairs_det = s.size  # photons detected
estim_photons_emit1 = photon_pairs_det / avg_sens
estim_photons_emit2 = np.sum(parallel_sino_data / parallel_sino_weights)
