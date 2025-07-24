import sys

sys.path.append("../")
from simuPET import array_lib as np
from plt import plt


def bpf_2D(boxes, detections, Ns, factor=2, weights=None):

    import parallelproj

    Ns = int(Ns)
    vol_shp = (1, Ns, Ns)

    # Origin and voxels in physical units
    radius = boxes.Rx
    voxel_size = np.array(3 * [2 * radius / vol_shp[-1]], dtype=np.float32)
    img_origin = ((-np.array(vol_shp) / 2 + 0.5) * voxel_size).astype(np.float32)

    # LOR coordinates in physical units
    # dets = detections[:,:,[1,2]]
    dets = np.zeros(tuple((np.array(detections.shape) + np.array([0, 0, 1])).get()))
    dets[..., 1:3] = detections
    xstart = dets[0, ...].astype(np.float32)
    xend = dets[1, ...].astype(np.float32)

    # setup a "sinogram" full of ones
    sino = (
        np.ones(xstart.shape[0], dtype=np.float32)
        if weights is None
        else weights.astype(np.float32)
    )

    # allocate memory for the back projection array
    back_img = np.zeros(vol_shp, dtype=np.float32)

    parallelproj.joseph3d_back(
        xstart, xend, back_img, img_origin, voxel_size, sino, num_chunks=1
    )
    plt.imshow(back_img[0, ...])
    plt.show()

    closest2 = max(64, int(2 ** np.ceil(np.log2(2 * Ns))))

    rec = np.copy(back_img[0, ...])
    rec_f = np.fft.fftn(rec, tuple(2 * [closest2]))  # power of 2
    fy_1D = np.fft.fftfreq(rec_f.shape[0])
    fx_1D = np.fft.fftfreq(rec_f.shape[1])
    fy, fx = np.meshgrid(fy_1D, fx_1D)

    F_filter = np.sqrt(fx**2 + fy**2)
    max_freq = np.max(fx_1D) / factor
    Window = (0.5 + 0.5 * np.cos(np.pi * F_filter / max_freq)) * (F_filter < max_freq)
    filtered_rec_f = (np.fft.ifftn(rec_f * F_filter * Window)).real
    result_fbp = filtered_rec_f[: rec.shape[0], : rec.shape[0]]

    return result_fbp


image_path = "/home/boquet/simuPET/simuPET/input_data/brain_phantom_z250.npy"

if True:
    nsim = 8
    radioactivity = 0.7 * 1e7


## SIMULATE
from simuPET.simulations import simulator

boxes, img, detections, phi, s, data_weights = simulator.iterate_simulate_muPET(
    nsim=nsim, weights=True, radioactivity=radioactivity, image_path=image_path
)
sino_data = np.stack((phi, s), axis=1)
pl = plt.hist2d(*sino_data.T, bins=100)


## WEIGHT
from simuPET.simulations.scanner_sensitivity import xray_scanner

data_weights = xray_scanner(boxes, *sino_data.T)
plt.show()

## RUN
# fbp = bpf_2D(boxes, detections, img.shape[0], factor=2, weights=None)
fbp = bpf_2D(
    boxes,
    detections,
    img.shape[0],
    factor=2,
    weights=1.0 / np.maximum(data_weights, np.percentile(data_weights, 10)),
)
plt.imshow(fbp[10:-10, 10:-10])
plt.show()
plt.imshow(img.T[:, ::-1][10:-10, 10:-10])
plt.show()
plt.imshow(
    img.T[:, ::-1][10:-10, 10:-10]
    - fbp[10:-10, 10:-10]
    * np.max(img.T[:, ::-1][10:-10, 10:-10])
    / np.max(fbp[10:-10, 10:-10])
)
plt.show()
# np.save("detections.npy", detections)
# np.save("data_weights.npy", data_weights)
# np.save("img.npy", img)
# boxes.Rx is 17.5
