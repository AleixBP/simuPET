import sys
sys.path.append('../')

from simuPET import array_lib as np
from plt import plt
from simuPET.simulations import simulator
from simuPET.simulations.scanner_sensitivity import *
boxes, *_, phi, s = simulator.simulate_simuPET(plot=False)
sino_data = np.stack((phi,s), axis=1)
plt.hist2d(*sino_data.T, bins=100)

thetas = np.linspace(0,np.pi,100)
lim = np.sqrt(2)/2
lim += 0.5 #some extra
si = np.linspace(-lim,lim,100)
tt, ss = np.meshgrid(thetas, si)

# Plot square
plt.contourf(tt, ss, xray_square(tt,ss)); plt.show()
plt.imshow(xray_square(tt,ss))

# Plot rectangle
plt.imshow(xray_rectangle(tt,ss))
plt.imshow(xray_rectangle(tt,ss,phi=np.pi/2))
plt.imshow(xray_rectangle(tt,ss,xy=[0.5,0.]))
plt.imshow(xray_rectangle(tt,ss,xy=[0.,0.5]))
plt.imshow(xray_rectangle(tt,ss,xy=[0.5,0.5]))
plt.imshow(xray_rectangle(tt,ss, a=[1.,1.2]))
plt.imshow(xray_rectangle(tt,ss,xy=[0.5,0.5], a=[1.,1.2], phi=np.pi/2))
plt.imshow(xray_rectangle(tt,ss,xy=[0.5,0.5], a=[1.,1.2], phi=np.pi/2) \
        +xray_rectangle(tt,ss,xy=[0.5,0.5], a=[1.,1.2], phi=0) \
        +xray_rectangle(tt,ss,xy=[0.5,0.5], a=[1.,1.2], phi=np.pi) \
        +xray_rectangle(tt,ss,xy=[0.5,0.5], a=[1.,1.2], phi=3*np.pi/2))

thetas = np.linspace(0,np.pi,100)
lim = boxes.Rx/np.sqrt(2)
#lim += Rx/2 #some extra
si = np.linspace(-lim,lim,100)
tt, ss = np.meshgrid(thetas, si)

# All lines with the same s
plot_allt_in_scanner(boxes, 5, t=np.pi/2, n=15)
#plt.savefig("all_lines_same_s.eps")

# All lines with the same theta
plot_alls_in_scanner(boxes, np.pi/4, s=boxes.Rx, n=15)
#plt.savefig("all_lines_same_theta.eps")

# Choose one line and plot it on the sinogram zonage
to, so = .78, 6.
pl = plt.imshow(all_cross3_scanner(boxes, tt, ss)); plt.colorbar(pl)
plt.scatter((tt.shape[0]/np.max(tt))*to,tt.shape[0]/2.+(ss.shape[0]/(2*np.max(ss)))*so)
#plt.savefig("3-intersections.eps")
plt.show()
plot_line_in_scanner(boxes,to,so)
#plt.savefig("3-intersections-correspondingpoint.eps")
plt.show()

# Plot all lines that intersect with top box with the equation constrain
fig = plot_intersect_in_scanner(boxes, 20, 0)
#fig.savefig("single_intersections.eps")

# Plot scanner sensitivty
sens = xray_scanner(boxes,tt,ss)
pl=plt.imshow(sens, extent=[0,np.pi,-float(lim),float(lim)], aspect="auto"); plt.colorbar(pl); plt.clim(0.02,0.12)
#plt.savefig("scanner_sensitivity.eps")
np.sum(sens) #7.817729939180599e+02 #without fov (e.g.big num boxes.fov=[100,100]) #with: 6.882479547482258e+02

# Plot s-slice of scanner sensitivity
offset=0
thetas1000 = np.linspace(0,np.pi,1000)
xx = xray_scanner(boxes,thetas1000,np.zeros_like(thetas1000)+offset)
plt.scatter(thetas1000, xx)
np.max(xx)/np.min(xx) #4.612851765655523e+00 #offset 0 (no fov does not really matter because central slice)
thetas1000[np.argmin(xx)]

#Plot t-slice of scanner sensitivity
offset=np.pi/5.
s1000 = np.linspace(-lim,lim,1000)
xx = xray_scanner(boxes,np.zeros_like(s1000)+offset, s1000)
plt.scatter(s1000, xx)
np.max(xx)/np.min(xx)
s1000[np.argmin(xx)]

#Compare sensitivity with x-ray path-length
xrs = scanner_rectangle(boxes,tt,ss)
pl=plt.imshow(xrs,extent=[0,np.pi,-float(lim),float(lim)], aspect="auto"); plt.colorbar(pl)
xxrs = scanner_rectangle(boxes,thetas1000,np.zeros_like(thetas1000)+offset)
plt.scatter(thetas1000, xxrs)
np.max(xxrs)/np.min(xxrs)

# Compare the minimums
np.where(xrs==0)
ind_min_xrs = np.unravel_index(np.argmin(xrs, axis=None), xrs.shape)
tt[ind_min_xrs] #keepdims maybe but version too old
ss[ind_min_xrs]

sl=np.s_[20:80,:]
ind_min_sens = np.unravel_index(np.argmin(sens[sl], axis=None), sens[sl].shape)
tt[sl][ind_min_sens]
ss[sl][ind_min_sens]
plt.imshow(sens[sl], aspect="auto")

# Overlap with simulation (from where we get phi and s)
#boxes, *_, phi, s = simulator.simulate_simuPET(image_path=[(840,840), 200, None, None], radioactivity=.7e7)
fig = compare_with_simulation(boxes, phi, s)
#fig.savefig("simulation_theory_fit.png")