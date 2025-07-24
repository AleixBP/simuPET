from simuPET import array_lib as np
from plt import plt
from os import path
from .poisson import simulate_poisson_and_geometric
from .fundamental_errors import (
    total_disintegrations,
    distance_decay_gauss,
    angle_deviation_gauss,
)
from ..plotting_utils.realunits_scatter import realunits_scatter_class


class detector_boxes:
    def __init__(
        self,
        sensitivity,
        Rx,
        Ry,
        shift,
        nLayers,
        layerLength,
        detLength,
        layerThickness,
        equivalentThickness,
        origin=None,
    ):
        # limits of each box ordered as bottom, top, left, right
        self.sensitivity = sensitivity
        self.Rx, self.Ry, self.shift, self.nLayers = Rx, Ry, shift, nLayers
        (
            self.layerLength,
            self.detLength,
            self.layerThickness,
            self.equivalentThickness,
        ) = (layerLength, detLength, layerThickness, equivalentThickness)

        self.lam = -np.log(1 - sensitivity) / layerThickness
        self.layerDepth = nLayers * layerThickness
        self.fov = [2.0 * Rx, 2.0 * Ry]
        self.right_box = [
            shift - Ry,
            shift - Ry + layerLength,
            Rx + shift,
            Rx + shift + self.layerDepth,
        ]
        self.top_box = [
            shift + Ry,
            shift + Ry + self.layerDepth,
            Rx - shift - layerLength,
            Rx - shift,
        ]
        self.domain = np.array([[-Rx, Rx], [-Ry, Ry]])
        self.radii = np.array([Rx, Ry])
        self.detDimensions = np.array([layerThickness, detLength])
        self.origin = origin

    def inside_top(self, x, y):
        return (
            (y > self.top_box[0])
            & (y < self.top_box[1])
            & (x > self.top_box[2])
            & (x < self.top_box[3])
        )

    def inside_right(self, x, y):
        return (
            (y > self.right_box[0])
            & (y < self.right_box[1])
            & (x > self.right_box[2])
            & (x < self.right_box[3])
        )

    def is_inside(self, x, y):
        return self.inside_top(x, y) | self.inside_right(x, y)


class detector_boxes_3D:
    def __init__(self, boxes, depth_box):
        # depth from shallow to deep
        self.boxes = boxes
        self.ref = np.concatenate((boxes.radii, np.diff(depth_box)))
        self.depth_box = depth_box

    def inside_top(self, x, y, z):
        return (
            self.boxes.inside_top(x, y) & z > self.depth_box[0] & z < self.depth_box[1]
        )

    def inside_top(self, x, y, z):
        return (
            self.boxes.inside_top(x, y) & z > self.depth_box[0] & z < self.depth_box[1]
        )

    def is_inside(self, x, y, z):
        return self.inside_top(x, y) | self.inside_right(x, y)


def rotate(x, theta=np.pi / 2.0):  # 2d
    if theta == np.pi / 2.0:
        return np.array([-1, 1]) * x[:, ::-1]
    if theta == -np.pi / 2.0:
        return np.array([1, -1]) * x[:, ::-1]
    if np.abs(theta) == np.pi:
        return -x

    Rmat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.einsum("ij,kj->ki", Rmat, x)


def assign_detectors(
    stopped, boxes
):  # consider asking for this function as an argument to the simulator
    dont_discard_top = boxes.inside_top(*stopped.T)
    dont_discard_right = boxes.inside_right(*stopped.T)
    detected_positions = (
        stopped
        + dont_discard_right[:, np.newaxis]
        * (boxes.detDimensions / 2.0 - ((stopped - boxes.radii) % boxes.detDimensions))
        + dont_discard_top[:, np.newaxis]
        * (
            boxes.detDimensions[::-1] / 2.0
            - ((stopped - boxes.radii) % boxes.detDimensions[::-1])
        )
    )
    return detected_positions, (dont_discard_right | dont_discard_top)


def angle_wrt_y(d1, d2):
    dif = d2 - d1
    return np.arctan2(dif[:, 0], dif[:, 1])


def distance_line_to_origin(d1, d2, point=None):
    if point is None:
        point = np.zeros(d1.ndim)
    dif = d2 - d1  # thinking of the line as d1+t*dif
    dif /= np.linalg.norm(dif, axis=1)[:, np.newaxis]  # normalise
    return np.linalg.norm(
        (d1 - point) - np.einsum("ij,ij->i", d1 - point, dif)[:, np.newaxis] * dif,
        axis=1,
    )  # einsum for dot product over all the list


def angle_distance(d1, d2, point=None):
    if point is None:
        point = np.zeros(d1.ndim)
    dif = d2 - d1  # thinking of the line as d1+t*dif
    dif /= np.linalg.norm(dif, axis=1)[:, np.newaxis]  # normalise
    normal_to_line = (d1 - point) - np.einsum("ij,ij->i", d1 - point, dif)[
        :, np.newaxis
    ] * dif  # einsum for dot product over all the list #points from point to line
    angles = np.arctan2(*normal_to_line.T[::-1])
    return angles + (angles < 0) * np.pi, np.sign(angles) * np.linalg.norm(
        normal_to_line, axis=1
    )


# 1. Result: top to bottom simulation
def simulate_scanner_2D_multilayer(
    specimen, boxes, disintegrations, tracer, convention=True, sim_domain=None
):

    domain = boxes.domain
    if sim_domain is None:
        sim_domain = domain
    sensitivity = boxes.sensitivity
    margin = np.sqrt(2)

    # 2+EXTRA. Combined Poisson positron emission plus geometric distribution for layer detection. (3D-compatible)
    emissions, layers1, layers2 = simulate_poisson_and_geometric(
        specimen, sim_domain, disintegrations, sensitivity, margin * boxes.nLayers
    )  # 26.3 s ± 696 ms with %timeit for 1e-6 # 4min 35s ± 12.7 under 1e-5 so linear
    nEvents = emissions.shape[0]

    # 3. Positron decay (mean free path until annihilation).
    # 3D: Add polar angle to 2D azimuth, be careful not to choose two uniforms (prob accumulates in poles/corners). add an extra clip. Should I do projection on circle rather?
    distance_traveled = distance_decay_gauss(tracer, size=nEvents)  # this is in mm
    angle_positron = np.random.uniform(
        low=0, high=2 * np.pi, size=nEvents
    )  # TODO: make 3D
    # free_path = np.array([np.multiply(distance_traveled, np.cos(angle_positron)), np.multiply(distance_traveled, np.sin(angle_positron))]).T
    free_path = np.array(
        [
            distance_traveled * np.cos(angle_positron),
            distance_traveled * np.sin(angle_positron),
        ]
    ).T

    annihilations = emissions + free_path
    annihilations[:, 0] = annihilations[:, 0].clip(*domain[0, :])
    annihilations[:, 1] = annihilations[:, 1].clip(*domain[1, :])

    # 4. Positron annihilation and photon emission: simulation of acolinearity
    quad1_nEvents = int(np.random.binomial(nEvents, 0.5))
    angles_12 = np.random.uniform(low=0, high=np.pi / 2.0, size=nEvents)
    deviations = angle_deviation_gauss(size=nEvents)
    indices = np.arange(nEvents)
    np.random.shuffle(indices)
    slices = np.split(indices, [quad1_nEvents])  # shuffle in case

    ##!! since perfectly symmetric, will just run this twice per angles1 and 2 and then invert
    # 5. Detection of pair of photons by scanner
    detected = 4 * [None]
    dont_discard = 4 * [None]

    for i, slice in enumerate(slices):

        annis = annihilations[slice]
        angles = angles_12[slice]

        if i == 1:
            annis = rotate(annis, theta=-np.pi / 2.0)
            # angles should have been created with np.pi/2. and then rotated back with -np.pi/2.

        for j, layers in enumerate([layers1, layers2]):
            if j == 1:
                angles += deviations[slice]  # + np.pi # add error and rotate
                annis *= -1  # rotate 180

            tan_angles = np.tan(angles)
            cos_angles = np.cos(angles)
            sin_angles = np.sin(angles)
            xdist_right = boxes.Rx + boxes.shift - annis[:, 0]
            ydist_top = boxes.Ry + boxes.shift - annis[:, 1]

            """
            select_side = (ydist_top*sin_angles > xdist_right*cos_angles).astype(int)
            intersects = select_side*[annis[:, 0]+ydist_top/tan_angles, boxes.Ry+boxes.shift]
                    + (1-select_side)*[boxes.Rx+boxes.shift, annis[:, 1]+xdist_right*tan_angles] # double-check for tans or 1/tans; also:broadcast properly
            stopped = intersects + layers[slice] * boxes.equivalentThickness * np.array([cos_angles, sin_angles]) # broadcast properly
            """
            select_side = (angles > np.pi / 4.0).astype(int)
            intersects = select_side[:, np.newaxis] * np.column_stack(
                (
                    annis[:, 0] + ydist_top / tan_angles,
                    np.broadcast_to(np.array(boxes.Ry + boxes.shift), (len(annis), 1)),
                )
            ) + (1 - select_side[:, np.newaxis]) * np.column_stack(
                (
                    np.broadcast_to(np.array(boxes.Rx + boxes.shift), (len(annis), 1)),
                    annis[:, 1] + xdist_right * tan_angles,
                )
            )
            stopped = (
                intersects
                + layers[slice][:, np.newaxis]
                * boxes.equivalentThickness
                * np.array([cos_angles, sin_angles]).T
            )

            detected[2 * i + j], dont_discard[2 * i + j] = assign_detectors(
                stopped, boxes
            )

    dont1 = (
        dont_discard[0] & dont_discard[1]
    )  # np.union1d(dont_discard[0], dont_discard[1])
    dont2 = (
        dont_discard[2] & dont_discard[3]
    )  # np.union1d(dont_discard[2], dont_discard[3])
    detections = [
        np.concatenate(
            (detected[0][dont1], rotate(detected[2][dont2], theta=np.pi / 2.0))
        ),
        np.concatenate(
            (
                -1.0 * detected[1][dont1],
                -1.0 * rotate(detected[3][dont2], theta=np.pi / 2.0),
            )
        ),
    ]
    # derotate 180 and 90s accordingly # no point in undoing mirroring/rotations of angles and annis

    if convention:
        phi, s = angle_distance(*detections)
    else:
        phi, s = angle_wrt_y(*detections), distance_line_to_origin(*detections)

    return np.array(detections), phi, s


def plot_simulate_scanner_2D_multilayer(
    boxes,
    img,
    detections,
    phi,
    s,
    theoretical_analysis=None,
    alpha=0.002,
    convention=True,
):

    plt.figure(figsize=[10, 10])
    plt.scatter(*detections[0].T, alpha=alpha, c="blue")
    plt.scatter(*detections[1].T, alpha=alpha, c="blue")
    plt.show()

    if theoretical_analysis is not None:

        theo_lines, theo_angles, theo_prob = theoretical_analysis

        if convention:
            plt.figure(figsize=[5, 5])
            plt.scatter(phi / np.pi, s, alpha=alpha)
            [plt.axvline(line, c="r") for line in (theo_lines) / np.pi]
            [plt.axvline(line, c="r") for line in (theo_lines + np.pi / 2) / np.pi]
            plt.show()
            plt.figure(figsize=[5, 5])
            plt.scatter(s, phi / np.pi, alpha=alpha)
            [plt.axhline(line, c="r") for line in (theo_lines) / np.pi]
            [plt.axhline(line, c="r") for line in (theo_lines + np.pi / 2) / np.pi]
            plt.xlim(-boxes.Rx, boxes.Rx)
            plt.ylim(0.0, 1.0)
            plt.show()
            plt.hist(s)
            plt.show()
            plt.hist(phi)
            plt.show()
            plt.hist(
                phi[(s < 0.5) & (s > -0.5)],
                bins=100,
                density=True,
                stacked=True,
                alpha=0.5,
            )
            plt.plot(theo_angles, 29 * theo_prob / np.sum(theo_prob), "k--", lw=2)
            plt.plot(
                np.pi / 2 + theo_angles, 29 * theo_prob / np.sum(theo_prob), "k--", lw=2
            )
            plt.show()

            if (
                False
            ):  # takes long: changing point size to angle accuracy, could also change to s accurac (half detLength?)
                _, ax = plt.subplots()
                realunits_scatter_class(
                    phi,
                    s,
                    ax,
                    size=np.arctan(
                        boxes.detLength
                        / (boxes.Rx + boxes.shift + nLayers * layerThickness / 2)
                    ),
                    alpha=0.125,
                )
                plt.show()
        else:
            shifted_phi = (phi - np.pi) % np.pi
            plt.figure(figsize=[5, 5])
            plt.scatter((shifted_phi) / np.pi, s, alpha=alpha)
            [plt.axvline(line, c="r") for line in (np.pi / 2.0 - theo_lines) / np.pi]
            [plt.axvline(line, c="r") for line in (np.pi - theo_lines) / np.pi]
            plt.show()
            plt.hist(s)
            plt.show()
            plt.hist(shifted_phi)
            plt.show()
            plt.hist(
                shifted_phi[s < 0.5], bins=100, density=True, stacked=True, alpha=0.5
            )
            plt.plot(
                np.pi / 2.0 - theo_angles - 0.03,
                29 * theo_prob / np.sum(theo_prob),
                "k--",
                lw=2,
            )
            plt.plot(
                np.pi - theo_angles - 0.03,
                29 * theo_prob / np.sum(theo_prob),
                "k--",
                lw=2,
            )
            plt.show()

    if True:
        print(
            "percentage of different phis:",
            100
            * (len(phi) - len(np.unique(phi, axis=0, return_counts=True)[0]))
            / len(phi),
        )
        print(
            "percentage of different s:",
            100 * (len(s) - len(np.unique(s, axis=0, return_counts=True)[0])) / len(s),
        )
        uniq = np.unique(np.vstack((phi, s)).T, axis=0, return_counts=True)
        print("most repeated:", uniq[0][np.max(uniq[1]) == uniq[1]])
        print(
            "percentage of different s and phi:", 100 * (len(s) - len(uniq[0])) / len(s)
        )
        counts, reps = np.unique(uniq[1], axis=0, return_counts=True)
        print(
            "number of counts and their <<frequency>>:",
            counts,
            reps,
            (np.sum(counts * reps) - len(phi)) == 0,
        )
        print(
            "percentage of counts 1:", 100.0 * reps[0] / len(s)
        )  # or np.sum(uniq[1]==1)
        if False:
            plt.bar(counts[:-1], reps[:-1])
            plt.yscale("log")
            plt.show()
        else:
            area = np.prod(np.diff(boxes.domain))
            estim_emitt_photons = area * np.sum(img) / img.size
            plt.bar(list((0, *counts[:-1])), list((estim_emitt_photons, *reps[:-1])))
            plt.yscale("log")
            plt.show()

    return 0


def interpolate(image, spline_degree):
    from scipy.interpolate import RectBivariateSpline, NearestNDInterpolator

    """
    Interpolate image using a given spline degree. Returns a continuous function to be called with the new coordinates.

    Parameters
    ----------
    image : ndarray
        image to be interpolated
    spline_degree : int
        spline degree to use for interpolation (0: nearest neighbor, 1: linear, ...)

    Returns
    -------
    spline : function
        continuous interpolation
    """
    N, M = image.shape  # height, width

    if spline_degree == 0:

        y, x = np.meshgrid(np.arange(M), np.arange(N))
        x = x.ravel()
        y = y.ravel()
        z = image.ravel()

        spline = NearestNDInterpolator(list(zip(x, y)), z)
    else:
        spline = RectBivariateSpline(
            np.arange(M), np.arange(N), image, kx=spline_degree, ky=spline_degree
        )

    return spline


def image_linear_spline_2D(image):
    padded = np.pad(image, 1, "edge")
    return lambda x, y: linear_spline_2D(padded, x, y, ct=0.5)


def linear_spline_2D(image, x, y, ct=0.0):
    xr, xq = np.modf(x + ct)
    xq = xq.astype(int, copy=False)
    yr, yq = np.modf(y + ct)
    yq = yq.astype(int, copy=False)
    return (
        image[xq, yq] * (1 - xr) * (1 - yr)
        + image[xq + 1, yq] * (xr) * (1 - yr)
        + image[xq, yq + 1] * (1 - xr) * (yr)
        + image[xq + 1, yq + 1] * (xr) * (yr)
    )


def theoretical_lines(
    r, eps, det_thick, det_length, nLayers, p, graf=False, plot=False
):

    def to_deg(phi):
        return 180 * phi / np.pi

    ys = r + eps
    ym = ys + det_thick

    rl = r - eps
    rr = r + eps
    b = det_length - rl

    tanalphm = ym / rl
    alphm = np.arctan(tanalphm)

    tanalphs = ys / rl
    alphs = np.arctan(tanalphs)

    cotalphss = rr / b
    alphss = np.arctan(1 / cotalphss)

    cotalphl = ym / b
    alphl = np.arctan(1 / cotalphl)

    if graf:
        angles = np.linspace(0.1, np.pi / 2, 100)

        def density_mod(theta):
            return (
                nLayers
                * rl
                * (tanalphm - tanalphs)
                * ((theta >= alphm) + (theta <= alphl))
                + nLayers
                * rl
                * (np.tan(theta) - tanalphs)
                * (theta >= alphs)
                * (theta <= alphm)
                + nLayers
                * b
                * (1.0 / np.tan(theta) - cotalphss)
                * (theta >= alphl)
                * (theta <= alphss)
            )

        densities = density_mod(angles)

        a1 = nLayers / (tanalphm - tanalphs)
        a2 = nLayers / (cotalphl - cotalphss)

        def layers_mod(theta):
            return (
                nLayers * ((theta >= alphm) + (theta <= alphl))
                + a1 * (np.tan(theta) - tanalphs) * (theta >= alphs) * (theta <= alphm)
                + a2
                * (1.0 / np.tan(theta) - cotalphss)
                * (theta >= alphl)
                * (theta <= alphss)
            )

        layers = layers_mod(angles)

        p_not = 1.0 - p
        p_det = 1 - p_not**nLayers  # /norm_ct
        p_det2 = p_det**2

        def probability(lays):
            return (1 - (p_not ** (lays))) ** 2

        probs = probability(layers)

    if plot:
        print(to_deg(alphs), to_deg(alphm), to_deg(alphss), to_deg(alphl))

        plt.scatter(to_deg(angles), densities)
        plt.xlim(0, 90)
        plt.show()
        plt.scatter(to_deg(angles), layers)
        plt.xlim(0, 90)
        plt.axvline(to_deg(alphl), c="r")
        plt.axvline(to_deg(alphss), c="r")
        plt.axvline(to_deg(alphs), c="r")
        plt.axvline(to_deg(alphm), c="r")
        plt.show()

        print("Minimum sensitivity at angle", to_deg(angles[np.argmin(layers)]))
        print(
            "Mean number of layers within the domain of ^problematic^ angles:",
            np.mean(layers_mod(np.linspace(alphl, alphm, 1000))),
        )
        print(
            "Probability of detecting a photon through all layers and a pair:",
            p_det,
            p_det2,
        )

        plt.scatter(to_deg(angles), probs)
        plt.xlim(0, 90)
        plt.axvline(to_deg(alphl), c="r")
        plt.axvline(to_deg(alphss), c="r")
        plt.axvline(to_deg(alphs), c="r")
        plt.axvline(to_deg(alphm), c="r")
        plt.show()

        print(
            "Mean probability of detecting a photon within the domain of ^problematic^ angles:",
            np.mean(probability(layers_mod(np.linspace(alphl, alphm, 1000)))),
        )
        print(
            "Mean probability of detecting a photon within the first 90degs including the domain of ^problematic^ angles (to compare with probability of detecting a photon):",
            np.mean(probability(layers_mod(np.linspace(0.001, np.pi / 2.0, 1000)))),
        )
        print("Min. prob among all angles:", probs[np.argmin(layers)])
        print("All angles:", to_deg(np.array([alphl, alphs, alphss, alphm])))
        print(
            "All angles + 90:",
            to_deg(np.array([alphl, alphs, alphss, alphm]) + np.pi / 2.0),
        )

    if not graf:
        return np.array([alphl, alphs, alphss, alphm])
    else:
        return np.array([alphl, alphs, alphss, alphm]), angles, probs


def mupet_scanner_boxes(theo=False, is_plot=False):

    sensitivity = 0.006
    detLength = 0.1
    layerThickness = 0.595
    nLayers = 60
    Rx = 17.5
    Ry = Rx
    layerLength = 50.3  # Closed: 2*Ry + nLayers*layerThickness
    shift = 0  # 2.06 # Closed: 0
    equivalentThickness = layerThickness  # equivalent layer thickness

    boxes = detector_boxes(
        sensitivity,
        Rx,
        Ry,
        shift,
        nLayers,
        layerLength,
        detLength,
        layerThickness,
        equivalentThickness,
    )  # or rotate 90 from right box to top box

    if not theo:
        return boxes
    else:
        # theo_lines = theoretical_lines(Rx, shift, layerThickness*nLayers, layerLength, nLayers, sensitivity)
        theoretical_analysis = theoretical_lines(
            Rx,
            shift,
            layerThickness * nLayers,
            layerLength,
            nLayers,
            sensitivity,
            graf=True,
            plot=is_plot,
        )
        return boxes, theoretical_analysis


def generate_pet_image(disintegrations, boxes, image_path, norm_p=2, is_plot=False):

    if image_path is None:
        rel_path = path.join(path.dirname(__file__), "../")
        IN_PATH = rel_path + "input_data/"
        image_path = IN_PATH + "derenzo_mask2.npy"

    if isinstance(image_path, str):
        image = np.load(image_path)

    else:
        shp, radius, center, norm_p = image_path

        center = np.array(shp) / 2.0 if center is None else np.array(center)
        if norm_p is None:
            norm_p = 2  # 1,2,np.inf

        image = (
            np.linalg.norm(
                np.mgrid[: shp[0], : shp[1]] - center[:, np.newaxis, np.newaxis],
                ord=norm_p,
                axis=0,
            )
            < radius
        )  # circle for norm_p = 2

    N, M = image.shape
    img = (
        disintegrations * image / np.sum(image)
    )  # normalize image and set integral to expected number of disintegrations in Poisson process
    poisson_highbound = np.max(img)  # bound needed in sampling
    # spline = interpolate(img, spline_degree=0)
    spline = image_linear_spline_2D(img)

    def specimen(x, y):
        xnew = (x + boxes.Rx) * (M / (2 * boxes.Rx))
        ynew = N - (y + boxes.Ry) * (N / (2 * boxes.Ry))
        # if 'grid' in inspect.getfullargspec(spline).args :
        #    return spline(xnew, ynew, grid=False)
        return spline(ynew, xnew)

    if is_plot:
        plt.imshow(img)
        plt.show()

    return img, specimen, poisson_highbound


def simulate_muPET(
    image_path=None, radioactivity=1e6, sim_domain=None, plot=False, boxes=None
):

    # Pharmaceutical and time of scanner
    tracer = "18F"
    # radioactivity = .7 * 1e7 # in Bq #PET grant tests done in 50 MBq during 30min, but mice typically injected with 7Mbq
    time_scan = 30 * 60  # in seconds
    disintegrations = total_disintegrations(radioactivity, time_scan)

    # The scanner
    if boxes is None:
        boxes, theoretical_analysis = mupet_scanner_boxes(theo=True)
    else:
        theoretical_analysis = None

    # Data
    img, specimen, poisson_highbound = generate_pet_image(
        disintegrations, boxes, image_path
    )  #

    # Simulate
    detections, phi, s = simulate_scanner_2D_multilayer(
        specimen, boxes, poisson_highbound, tracer, sim_domain=sim_domain
    )
    if plot:
        plot_simulate_scanner_2D_multilayer(
            boxes,
            img,
            detections,
            phi,
            s,
            theoretical_analysis=theoretical_analysis,
            alpha=0.002,
        )
        sino_data = np.stack((phi, s), axis=1)
        pl = plt.hist2d(*sino_data.T, bins=100)

    return boxes, img, detections, phi, s


def iterate_simulate_muPET(nsim=1, weights=True, **kwargs):
    # dask, but not sending arrays here, would have to put it in simulate poisson and geometric
    arg_len = 3
    if weights:
        arg_len += 1
        from simuPET.simulations.scanner_sensitivity import xray_scanner

    sims = [[] for _ in range(arg_len)]
    axes = [1] + (arg_len - 1) * [0]

    # Do nsim consecutive simulations and compute the sensitivity weights on each
    # Append to list as detections, phi, s, weights
    for i in range(nsim):
        boxes, img, *dps = simulate_muPET(**kwargs)
        for j, par in enumerate(dps):
            sims[j].append(par)  # dps[j]

        if weights:
            sims[-1].append(xray_scanner(boxes, *np.stack((dps[1], dps[2]), axis=1).T))

    # Concatenate everything as if it came from a single simulation
    sims = [np.concatenate(l, axis=ax) for l, ax in zip(sims, axes)]

    return boxes, img, *sims


if __name__ == "__main__":
    boxes, img, detections, phi, s = simulate_muPET()
