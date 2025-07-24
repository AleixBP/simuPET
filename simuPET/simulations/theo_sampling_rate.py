from simuPET import array_lib as np
from simuPET.simulations.simulator import mupet_scanner_boxes


def choose_image_size(pixel_res=None, factor=2, boxes=None):
    if boxes is None:
        boxes = mupet_scanner_boxes()
    if pixel_res is None:
        pixel_res = boxes.detLength

    N_s, N_th, *_ = choose_sampling_size(
        c_mm=factor * pixel_res, boxes=boxes, verbose=False
    )

    return N_s, N_th


def choose_sampling_size(c_mm=None, boxes=None, verbose=False):
    if boxes is None:
        boxes = mupet_scanner_boxes()

    # Radial undersampling generates more artifacts
    # 100mu is 0.1mm

    # Compute necessary N_s and N_th to properly sample size c_mm
    # according to Nyquist
    if c_mm is None:
        c_mm = boxes.detLength
    As_mm = c_mm / 2
    Ath_rad = c_mm / (2.0 * boxes.Rx)  # As_mm/boxes.Rx
    N_s = round(4.0 * boxes.Rx / c_mm)
    N_th = round(2.0 * np.pi * boxes.Rx / c_mm)  # np.pi/Ath_rad

    if verbose:
        mm_px = (
            2 * boxes.Rx / N_s if N_s is not None else np.NaN
        )  # mm/pixel #Ns=img.shape[-1]

        def mm2px(x, inverse=False):
            return x / mm_px if not inverse else mm_px / x

        w_mm = boxes.detLength
        mm = "mm"
        px = "px"
        rad = "rad"
        print(
            "The detector size is ",
            w_mm,
            mm,
            ", which is equivalent to ",
            mm2px(w_mm),
            px,
        )
        print(
            "We wish to reconstruct sizes of",
            c_mm,
            mm,
            ", which is equivalent to ",
            mm2px(c_mm),
            px,
        )
        print(
            "We require distance steps of ",
            As_mm,
            mm,
            ", which is equivalent to ",
            mm2px(As_mm),
            px,
            ". That is ",
            N_s,
            " samples.",
        )
        print(
            "We require angle steps of ", Ath_rad, rad, ". That is ", N_th, " samples."
        )

    return N_s, N_th, As_mm, Ath_rad
