import sys

sys.path.append("../")

from simuPET import array_lib as np
from plt import plt
from simuPET.simulations import simulator

boxes, img, *_, phi, s = simulator.simulate_simuPET()
from simuPET.simulations.scanner_sensitivity import xray_scanner
from simuPET.resampling.resample import dens_resample, kde_dens_estim, hist_dens_estim
from simuPET.reconstruction.astra_operators import fbp_reconstruction, bp_reconstruction
import astra

# Visualize simulated data
sino_data = np.stack((phi, s), axis=1)
pl = plt.hist2d(*sino_data.T, bins=100)

# Make parallel grid on sinogram domain
boxes.fov = [35.0, 35.0]
number_of_phi_samples = 180
number_of_s_samples = 800  # detectors
domain_s = np.array([np.min(s), np.max(s)])

sampled_phi = np.linspace(0, np.pi, number_of_phi_samples)  # no need for uniformity
sampled_s = np.linspace(*domain_s, number_of_s_samples)  # need uniformity

# Astra: fix parallel projection grid
spacing_of_s = 1  # because they need to be uniform

proj_geom = astra.creators.create_proj_geom(
    "parallel", spacing_of_s, number_of_s_samples, sampled_phi.get()
)
sino_shp = (number_of_phi_samples, number_of_s_samples)

# Astra: fix image grid
# voxel_rows, voxel_cols = 840, 840 # volume voxels
voxel_rows, voxel_cols = img.shape
shp = (voxel_rows, voxel_cols)
vol_geom = astra.creators.create_vol_geom(shp)

## WEIGHTED RESAMPLING
# Create weights on data points and parallel grid
data_weights = xray_scanner(boxes, *sino_data.T)
parallel_sino_weights = xray_scanner(boxes, *np.meshgrid(sampled_phi, sampled_s)).T

parallel_sino_data = dens_resample(
    sino_data,
    sampled_phi,
    sampled_s,
    hist_dens_estim,
    args=[100, 1.0 / data_weights],
    plot=True,
)
parallel_sino_data = np.load("hist_dense_weighted.npy")
parallel_sino_data = dens_resample(
    sino_data,
    sampled_phi,
    sampled_s,
    kde_dens_estim,
    args=[1e5, 0.025, 1.0 / data_weights, False],
    plot=True,
)
parallel_sino_data = np.load("kde_dense_weighted.npy")

plt.imshow(parallel_sino_data.T, aspect="auto")
plt.imshow(parallel_sino_weights.T, aspect="auto")
plt.imshow(1.0 / parallel_sino_weights.T, aspect="auto")

# FBP
fbp_rec = fbp_reconstruction(proj_geom, vol_geom, parallel_sino_data)
pl = plt.imshow(fbp_rec)
plt.colorbar(pl)
plt.show()  # ; plt.clim(0,10); plt.show()

# BP
bp_rec = bp_reconstruction(proj_geom, vol_geom, parallel_sino_data)
pl = plt.imshow(bp_rec)
plt.colorbar(pl)
plt.show()


## NON-WEIGHTED RESAMPLING
parallel_sino_data = dens_resample(
    sino_data, sampled_phi, sampled_s, hist_dens_estim, args=[100, None], plot=True
)
parallel_sino_data = np.load("hist_dense_notweighted.npy")
parallel_sino_data = dens_resample(
    sino_data,
    sampled_phi,
    sampled_s,
    kde_dens_estim,
    args=[1e5, 0.025, None, False],
    plot=True,
)
parallel_sino_data = np.load("kde_dense_notweighted.npy")

plt.imshow(parallel_sino_data.T, aspect="auto")

# FBP
fbp_rec = fbp_reconstruction(proj_geom, vol_geom, parallel_sino_data)
pl = plt.imshow(fbp_rec)
plt.colorbar(pl)
plt.show()  # ; plt.clim(0,10); plt.show()

# BP
bp_rec = bp_reconstruction(proj_geom, vol_geom, parallel_sino_data)
pl = plt.imshow(bp_rec)
plt.colorbar(pl)
plt.show()


## OTHER EXAMPLES
# Examples with applying FP and BP iteratively
P = astra.OpTomo(astra.creators.create_projector("line", proj_geom, vol_geom))
yuk = np.copy(parallel_sino_data)
for i in range(5):
    # plt.imshow(P(yuk).reshape(sino_shp).T.get(), aspect="auto");plt.show()
    yuk = P.adjoint(P(yuk))
    plt.imshow(yuk.reshape(shp).get())
    plt.show()

## OTHER EXAMPLES
# Examples with BP operator
P = astra.OpTomo(astra.creators.create_projector("line", proj_geom, vol_geom))
rec_unif = (P.T * parallel_sino_data).reshape(shp)
fp = P * rec_unif
plt.imshow(fp.reshape((number_of_phi_samples, number_of_s_samples)), aspect="auto")
plt.show()
bp2 = P.T * fp
plt.imshow(bp2.reshape(shp))
plt.show()
bp = P.T * parallel_sino_data
plt.imshow(bp.reshape(shp))
plt.show()

# More
radius = shp[0] / 4.0
norm_p = 2
data = (
    np.linalg.norm(
        np.mgrid[: shp[0], : shp[1]]
        - np.array([shp[0], shp[1]])[:, np.newaxis, np.newaxis] / 2.0,
        ord=norm_p,
        axis=0,
    )
    < radius
)
yo = (P * data).reshape(sino_shp)
plt.imshow(yo)

bp = P.T * yo
plt.imshow(bp.reshape(shp))
plt.show()

sino_data = (
    np.linalg.norm(
        np.mgrid[: sino_shp[0], : sino_shp[1]]
        - np.array([sino_shp[0], sino_shp[1]])[:, np.newaxis, np.newaxis] / 2.0,
        ord=norm_p,
        axis=0,
    )
    < radius
)
bp = P.T * sino_data  # lets assume instead the rombo is in sinogram domain
plt.imshow(bp.reshape(shp))
plt.show()

## If the sinogram is a uniform column, then the original object was probably a circle with more density at the edges: lets check!
unif = np.zeros_like(parallel_sino_data)
radius = 100
center = 400  # remember singoram sampling size (proj_geom) is different from real volume geom
unif[:, (center - radius) : (center + radius)] = 1
unif *= 5

# Normal BP
plt.imshow((P.T * unif).reshape(shp))
plt.show()

# FBP
rec_unif = fbp_reconstruction(proj_geom, vol_geom, unif)
plt.imshow(rec_unif)
plt.show()


plt.plot(rec_unif[(420 - radius) : (420 + radius), 420])
plt.show()
plt.imshow((P * rec_unif).reshape(180, 800))
