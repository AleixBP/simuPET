import sys

sys.path.append("../")
from simuPET import array_lib as np
from plt import plt
import tikzplotlib


from muPET.simulations import simulator

radioactivity = 0.87 * 0.5 * 1e7
boxes, img, detections, phi, s = simulator.simulate_muPET(
    radioactivity=radioactivity, plot=False, image_path=((840, 840), 200, None, None)
)

# simulator.plot_simulate_scanner_2D_multilayer(boxes, img, detections, phi, s)
uniq = np.unique(np.vstack((phi, s)).T, axis=0, return_counts=True)
counts, reps = np.unique(uniq[1], axis=0, return_counts=True)
area = np.prod(np.diff(boxes.domain))
estim_emitt_photons = area * np.sum(img) / img.size
plt.bar(list((0, *counts[:-1])), list((estim_emitt_photons, *reps[:-1])))
plt.yscale("log")
plt.show()

combis = 9.1 * 1e8

plt.bar(list((0, *counts[:-2])), list((combis, *reps[:-2])))
plt.yscale("log")
tikzplotlib.save("photon_statistics.tex")
