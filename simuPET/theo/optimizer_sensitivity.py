import sys
sys.path.append('../../')
from simuPET import array_lib as np
from plt import plt
import tikzplotlib

from simuPET.simulations import simulator
import simuPET.simulations.scanner_sensitivity as sensi
np.cuda.Device(3).use()

## SCANNER DIMENSIONS
if True:
    end_of_block = 48.2
    sensitivity = 0.0048
else:
    # bismuth
    end_of_block = 51.2
    sensitivity = 0.0073

Rx = Ry = 17.
start_of_block = 20.
block_thickness = end_of_block - start_of_block

shift = start_of_block - Rx
layerLength = 46.2 + 14.

nLayers = 60
detLength = 0.1
block_half_axial_depth = 46.2

block_thickness = end_of_block - Rx
layerThickness = block_thickness/nLayers
equivalentThickness = layerThickness

## SAMPLING VECTOR
thetas = np.linspace(0,np.pi,100,endpoint=False)
lim = float(Rx) #boxes.Rx*np.sqrt(2) #boxes.Rx/np.sqrt(2)
si = np.linspace(-lim,lim,100)
tt, ss = np.meshgrid(thetas, si)

## BUILD SCANNER
boxes = simulator.detector_boxes(sensitivity, Rx, Ry, shift, nLayers, layerLength, detLength, layerThickness, equivalentThickness)
sens = sensi.xray_scanner(boxes,tt,ss)
pl=plt.imshow(100*sens, extent=[0,np.pi,-lim,lim], aspect="auto"); plt.colorbar(pl); plt.clim(2,12) #plt.clim(0.02,0.12)
#tikzplotlib.save("sensi_sensisinogram.tex")

## DIFFERENTIATE Length
step = 20
boxes2 = simulator.detector_boxes(sensitivity, Rx, Ry, shift, nLayers, layerLength+step, detLength, layerThickness, equivalentThickness)
sens2 = (sensi.xray_scanner(boxes2,tt,ss)-sens)/step
pl=plt.imshow(100*sens2, extent=[0,np.pi,-lim,lim], aspect="auto"); plt.colorbar(pl); # plt.clim(0.02,0.12)
100*np.sum(sens2)/tt.size #how much did the sensitivity change on average
#100*sens2 is like the step for 1cm so around 2%
#sensi.scanner_plot(boxes2)
#tikzplotlib.save("sensi_sensisinogram_diffstep.tex")

## DIFFERENTIATE number of layers
step = 40
boxes2 = simulator.detector_boxes(sensitivity, Rx, Ry, shift, nLayers+step, layerLength, detLength, layerThickness, equivalentThickness)
sens2 = (sensi.xray_scanner(boxes2,tt,ss)-sens)/step
pl=plt.imshow(100*sens2, extent=[0,np.pi,-lim,lim], aspect="auto"); plt.colorbar(pl);
100*np.sum(sens2)/tt.size
#tikzplotlib.save("sensi_sensisinogram_diffnlayers.tex")

## DIFFERENTIATE probability
step = 0.0048
boxes2 = simulator.detector_boxes(sensitivity+step, Rx, Ry, shift, nLayers, layerLength, detLength, layerThickness, equivalentThickness)
sens2 = 0.0048*(sensi.xray_scanner(boxes2,tt,ss)-sens)/step # multiply by a step of 0.048, checking the amount of change induced by doubling
pl=plt.imshow(100*sens2, extent=[0,np.pi,-lim,lim], aspect="auto"); plt.colorbar(pl);
100*np.sum(sens2)/tt.size
#tikzplotlib.save("sensi_sensisinogram_difflambda.tex")

## AVERAGE SENSITIVITY
100*np.sum(sens)/tt.size

## DIFFERENTIATE bismuth
end_of_block2 = 51.2
block_thickness2 = end_of_block2 - start_of_block
block_thickness2 = end_of_block2 - Rx
layerThickness2 = block_thickness2/nLayers
equivalentThickness2 = layerThickness2
step = 0.0073-0.0048
boxes2 = simulator.detector_boxes(sensitivity+step, Rx, Ry, shift, nLayers, layerLength, detLength, layerThickness2, equivalentThickness2)
sens2 = step*(sensi.xray_scanner(boxes2,tt,ss)-sens)/step # multiply by a step of 0.048, checking the amount of change induced by doubling
pl=plt.imshow(100*sens2, extent=[0,np.pi,-lim,lim], aspect="auto"); plt.colorbar(pl);
100*np.sum(sens2)/tt.size
100*np.sum(sensi.xray_scanner(boxes2,tt,ss))/tt.size


## TRY AVERAGE SENSITIVITY FOR COMBINATIONS OF NLAYERS AND LENGTH WITH CONSTANT VOLUME


## CHECK for many layers
yo = []
layers = range(1,500)
for nl in layers:
    iter_boxes = simulator.detector_boxes(sensitivity, Rx, Ry, shift, nl, layerLength, detLength, layerThickness, equivalentThickness)
    iter_sens = sensi.xray_scanner(iter_boxes,tt,ss)
    yo.append(np.sum(iter_sens)/tt.size)
yo = np.array(yo)

plt.plot(layers, 100*yo)
plt.plot(layers[1:], 150*np.diff(100*yo))
#tikzplotlib.save("sensi_sensivslayers.tex")

## CHECK for many lengths
layerLength
yo = []
layers = np.linspace(layerLength-0.5*layerLength, layerLength+2*layerLength, 30)
for ll in layers:
    iter_boxes = simulator.detector_boxes(sensitivity, Rx, Ry, shift, nLayers, float(ll), detLength, layerThickness, equivalentThickness)
    iter_sens = sensi.xray_scanner(iter_boxes,tt,ss)
    yo.append(np.sum(iter_sens)/tt.size)
yo = np.array(yo)
plt.plot(layers, yo)
plt.plot(layers[1:], np.diff(yo))


## CHECK FOR LAYERS AND LENGTH
nlayers = range(1,120,4)
llayers = np.linspace(layerLength-0.5*layerLength, layerLength+0.5*layerLength, 10)
yo = np.zeros((len(nlayers), len(llayers)))
for i, nl in enumerate(nlayers):
    for j, ll in enumerate(llayers):
        iter_boxes = simulator.detector_boxes(sensitivity, Rx, Ry, shift, nl, float(ll), detLength, layerThickness, equivalentThickness)
        iter_sens = sensi.xray_scanner(iter_boxes,tt,ss)
        yo[i,j] = float(np.sum(iter_sens)/tt.size)
#around 40s for 125x10
#changing nlayers max from 500 to 120
#changing length from 3 times length to 2 times
extent = [float(llayers[-1]), float(llayers[0]) , nlayers[0],nlayers[-1]]
if True:
    yo = yo.T[::-1,::]
    extent = [nlayers[0],nlayers[-1], float(llayers[0]), float(llayers[-1])]

xx, yy = np.meshgrid(np.array(nlayers), llayers)

pl = plt.imshow(100*yo, extent=extent, aspect="auto"); plt.colorbar(pl);
pl = plt.contour(100*yo[::-1,::], 10, extent=extent, aspect="auto", colors='black');
plt.contour(xx*yy, np.linspace(1200,10000,5), extent=extent, aspect="auto", colors='red');
#tikzplotlib.save("sensi_opti2Dlengthandnumlayers.tex")


plt.contour(xx*yy, np.linspace(1000,10000,7), extent=extent, aspect="auto", colors='red');

plt.contour(xx*yy, 10, extent=extent, aspect="auto", colors='red');
yo = np.array(yo)
plt.plot(layers, yo)
plt.plot(layers[1:], np.diff(yo))

xx, yy = np.meshgrid(np.array(nlayers), llayers)
plt.imshow(xx+yy)
plt.contour(xx*yy, 10, aspect="auto", colors='black');
plt.contour(xx*yy, np.linspace(1000,10000,10), aspect="auto", colors='red');


#### SENSITIVITY MAP AT FOV
pl=plt.imshow(sens, extent=[0,np.pi,-lim,lim], aspect="auto"); plt.colorbar(pl); plt.clim(0.02,0.12)
## Turn sens from image to function using spline
img = sens
#spline = interpolate(img, spline_degree=0)
spline = image_linear_spline_2D(img)
N, M = img.shape
def specimen(x, y):
        xnew = (x + 0) * (M/(np.pi))
        ynew = N - (y + boxes.Ry) * (N/(2*boxes.Ry))
        #if 'grid' in inspect.getfullargspec(spline).args :
        #    return spline(xnew, ynew, grid=False)
        return spline(ynew, xnew)
    
plt.plot(specimen(angles,np.zeros_like(angles))) #checking for shape at center

## Then integrate the sinusoid according to the tranlsation property for each x,y in the FOV.
#xx, yy = np.meshgrid(np.linspace(-lim, lim, 10), np.linspace(-lim, lim, 10))
angles = np.linspace(0,np.pi,80,endpoint=False)
xx = np.linspace(-lim, lim, 100)
yy = np.linspace(-lim, lim, 100)
fov_sens = np.zeros((xx.size, yy.size))
for i, x in enumerate(xx):
    for j, y in enumerate(yy):
       esse = x*np.sin(angles)+y*np.cos(angles)
       if x**2+y**2<boxes.Ry**2:
           sp = specimen(angles,esse)
           fov_sens[i,j] = np.sum(sp/sp.size)
from matplotlib.colors import Colormap
cmap = plt.cm.get_cmap("viridis").copy()
Colormap.set_under(cmap,color='k')
pl = plt.imshow(fov_sens, extent=[-lim,lim,-lim,lim], cmap=cmap); plt.colorbar(pl); plt.clim(np.min(fov_sens[fov_sens!=0]),np.max(fov_sens));
#tikzplotlib.save("sensi_sensifov.tex")
fov_sens_pow = np.copy(fov_sens)
fov_sens_pow[fov_sens!=0] = (fov_sens[fov_sens!=0])**(-1/2)
np.sum(fov_sens_pow)
#fov_sens_pow[fov_sens!=0] = (100*fov_sens[fov_sens!=0])**(3/2)
pl = plt.imshow(fov_sens_pow, extent=[-lim,lim,-lim,lim], cmap=cmap); plt.colorbar(pl); plt.clim(np.min(fov_sens_pow[fov_sens_pow!=0]),np.max(fov_sens_pow));
#tikzplotlib.save("minus1bar2_sensi_sensifov.tex")
pw =3/2 #3/2, -1/2, -1
fov_sens_pow = np.copy(fov_sens)
fov_sens_pow[fov_sens!=0] = (fov_sens[fov_sens!=0])**(pw)
non_zer = np.sum(fov_sens!=0)
int_pw = np.sum(fov_sens_pow)
int_uni = non_zer*((np.sum(fov_sens)/non_zer)**pw)
avg_pw = int_pw/non_zer
avg_uni = int_uni/non_zer
[pw, avg_pw, avg_uni, avg_pw/avg_uni]
       
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
    N, M = image.shape # height, width

    if spline_degree == 0:

        y, x = np.meshgrid(np.arange(M), np.arange(N))
        x = x.ravel()
        y = y.ravel()
        z = image.ravel()

        spline = NearestNDInterpolator(list(zip(x, y)), z)
    else:
        spline = RectBivariateSpline(np.arange(M), np.arange(N), image, kx=spline_degree, ky=spline_degree)

    return spline


def image_linear_spline_2D(image):
    padded = np.pad(image, 1, "edge")
    return lambda x, y : linear_spline_2D(padded, x, y, ct=0.5)


def linear_spline_2D(image, x, y, ct=0.):
    xr, xq = np.modf(x+ct); xq = xq.astype(int, copy=False)
    yr, yq = np.modf(y+ct); yq = yq.astype(int, copy=False)
    return image[xq,yq]*(1-xr)*(1-yr) + image[xq+1,yq]*(xr)*(1-yr) \
            + image[xq,yq+1]*(1-xr)*(yr) + image[xq+1,yq+1]*(xr)*(yr)


#sensi.scanner_plot(boxes)
#sensi.plot_allt_in_scanner(boxes, 5, t=np.pi/2, n=15)
#sensi.plot_alls_in_scanner(boxes, np.pi/4, s=boxes.Rx, n=15)
#sensi.compare_with_simulation(boxes, phi, s)

from muPET.simulations import simulator
np.cuda.Device(3).use()
radioactivity=1e7
boxes, img, detections, phi, s = simulator.simulate_muPET(radioactivity=radioactivity, plot=False, 
                                                            image_path=((840,840),200,None,None))
%timeit simulator.simulate_muPET(radioactivity=radioactivity, plot=False, image_path=((840,840),200,None,None))


from muPET.simulations.scanner_sensitivity

thetas = np.linspace(0,np.pi,100)
    lim = boxes.Rx/np.sqrt(2)
    #lim += Rx/2 #some extra
    si = np.linspace(-lim,lim,100)
    tt, ss = np.meshgrid(thetas, si)

    # All lines with the same s
    plot_allt_in_scanner(boxes, 5, t=np.pi/2, n=15)

    # All lines with the same theta
    plot_alls_in_scanner(boxes, np.pi/4, s=boxes.Rx, n=15)
    