# simuPET
**PET theory, simulation, and reconstruction**

---

## Requirements

You will only need CuPy or NumPy + SciPy (choose in the `__init__.py` file). If you also want to reconstruct you will need astra or parallelproj.

---

## What does it do?

Given a 2D image, it simulates a PET scanner with four blocks of detectors forming a square. The simulation is in 2D, it takes into account the decay according to a Poisson, the mean free path of the postrion, the acolinearity of the two gamma photons, and the probability of detection. There are several reconstruction algorithms implemented (FBP, TV, etc.) based on sinogram resampling or ray tracing. There are also some theory modules.




