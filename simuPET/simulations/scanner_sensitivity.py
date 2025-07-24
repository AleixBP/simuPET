from simuPET import array_lib as np
from plt import plt
from simuPET.utils.bsplines import bs as bspline


def xray_square(theta, s):
    #theta between pi/4 and pi/2
    #theta = theta%np.pi/2
    theta = 2*theta%np.pi/2
    theta[theta<np.pi/4] = np.pi/2-theta[theta<np.pi/4]
    s = np.abs(s)
    s1 = np.cos(theta)
    s0 = 0.5*(np.sin(theta)-s1)
    return (bspline(0.5*s/s0, 0) + (s>=s0)*bspline((s-s0)/s1, 1))/np.sin(theta)


def xray_rectangle(theta,s,a=[1.,1.],xy=[0.,0.],phi=0):
    theto = theta-phi

    x, y = xy
    s_xy = x*np.cos(theto)+y*np.sin(theto)

    ax, ay = a
    gamma = np.sqrt((ax*np.cos(theto))**2+(ay*np.sin(theto))**2)

    condi = theto<np.pi/2
    theto = np.arctan(ay*np.tan(theto)/ax)
    theto[condi] += np.pi

    return (ax*ay/gamma)*xray_square(theto,(s-s_xy)/gamma)


def geom_xray_scanner(theta,s,a=[1.,1.],xy=[0.,0.],lam=1,blocks=4,fov=[1.,1.], extra=None):
    hpi = np.pi/2
    result = np.zeros_like(theta)
    
    if extra is None:
        for i in range(blocks):
            phi = -i*hpi
            condi = (result==0)
            result[condi] = (xray_rectangle(theta[condi],s[condi],a=a,xy=xy,phi=(phi+hpi))==0)  \
                        * (1-np.exp(-lam*xray_rectangle(theta[condi],s[condi],a=a,xy=xy,phi=phi))) \
                        * (1-np.exp(-lam*(xray_rectangle(theta[condi],s[condi],a=a,xy=xy,phi=(phi-hpi))\
                                        +xray_rectangle(theta[condi],s[condi],a=a,xy=xy,phi=(phi-2*hpi)) \
                                        )))
    else:
        for i in range(blocks):
            phi = -i*hpi
            condi = (result==0)
            result[condi] = (xray_rectangle(theta[condi],s[condi],a=extra[0,0],xy=extra[1,0],phi=(phi-hpi))==0) \
                        * (xray_rectangle(theta[condi],s[condi],a=extra[0,1],xy=extra[1,1],phi=(phi+hpi))==0)  \
                        * (1-np.exp(-lam*xray_rectangle(theta[condi],s[condi],a=a,xy=xy,phi=phi))) \
                        * (1-np.exp(-lam*(xray_rectangle(theta[condi],s[condi],a=a,xy=xy,phi=(phi-hpi))\
                                        + xray_rectangle(theta[condi],s[condi],a=a,xy=xy,phi=(phi-2*hpi))\
                                        + xray_rectangle(theta[condi],s[condi],a=a,xy=xy,phi=(phi-3*hpi)) \
                                        )))
    if blocks==4:
        condi = (result==0)
        print(np.sum(condi))
        theto = 2*theta[condi]%np.pi/2
        phi=0
        result[condi]=  (1-np.exp(-lam*(xray_rectangle(theto,s[condi],a=a,xy=xy,phi=phi)\
                                        +xray_rectangle(theto,s[condi],a=a,xy=xy,phi=(phi+hpi)) \
                                        ))) \
                    * (1-np.exp(-lam*(xray_rectangle(theto,s[condi],a=a,xy=xy,phi=(phi+2*hpi))\
                                        +xray_rectangle(theto,s[condi],a=a,xy=xy,phi=(phi+3*hpi)) \
                                        )))
    result = result*(xray_rectangle(theta,s,a=fov)>0) # field of view
    return result


def xray_scanner(boxes, theta, s):
    a = [boxes.layerLength, boxes.layerDepth]
    xy = [boxes.Rx-boxes.shift-boxes.layerLength/2., boxes.Ry+boxes.shift+boxes.layerDepth/2.]
    extra = np.array([[[boxes.layerLength-2.*boxes.Rx, boxes.layerDepth], [2.*boxes.Rx, boxes.layerDepth]], \
                    [[-boxes.shift-boxes.layerLength/2., boxes.Ry+boxes.shift+boxes.layerDepth/2.], [-boxes.shift, boxes.Ry+boxes.shift+boxes.layerDepth/2.]]])
    return geom_xray_scanner(theta,s,a=a,xy=xy,lam=boxes.lam,blocks=4,fov=boxes.fov,extra=extra)


def scanner_rectangle(boxes, tt,ss):
        a = [boxes.layerLength, boxes.layerDepth]
        xy = [boxes.Rx-boxes.shift-boxes.layerLength/2., boxes.Ry+boxes.shift+boxes.layerDepth/2.]
        xrs = np.zeros_like(tt)
        hpi = np.pi/2
        for i in range(4):
            xrs += xray_rectangle(tt,ss,xy=xy, a=a, phi=i*hpi)
        return xrs


def scanner_plot(boxes):
    import matplotlib.patches
    from plt import plot_wrapper
    Rectangle = plot_wrapper(matplotlib.patches).Rectangle
    
    fig, ax = plt.subplots()
    xy0 = np.array(boxes.top_box[::2][::-1])
    for i in range(4):
        theta=i*np.pi/2
        Rmat = np.array([ [np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)] ])
        xy=np.dot(Rmat, xy0)
        rect = Rectangle(xy, \
                            boxes.layerLength, boxes.layerDepth, angle=np.rad2deg(theta), edgecolor = 'black', \
                            fill = False, lw=1)
        ax.add_patch(rect)
    ax.set_aspect('equal')
    plt.scatter(0.,0., marker="+")
    plt.xlim(-boxes.right_box[3],boxes.right_box[3])
    plt.ylim(-boxes.top_box[1],boxes.top_box[1])
    return fig, ax


def line_plot(t,s, ct=100):
    #x , y =  -s*np.sin(t), s*np.cos(t)
    #back_forw = np.array([-1.,1.])
    #plt.plot(x+ct*np.cos(t)*back_forw, y+ct*np.sin(t)*back_forw, lw=0.5)
    #return 0
    x , y =  s*np.cos(t), s*np.sin(t)
    back_forw = np.array([-1.,1.])
    plt.plot(x-ct*np.sin(t)*back_forw, y+ct*np.cos(t)*back_forw, lw=0.5)
    return 0


def plot_line_in_scanner(boxes, t,s):
    scanner_plot(boxes)
    line_plot(t,s)
    plt.show()


def plot_alls_in_scanner(boxes,t, s=10, n=10):
    scanner_plot(boxes)
    for s in np.linspace(-s, s, n):
        line_plot(t,s)
    plt.show()


def plot_allt_in_scanner(boxes,s, t=2*np.pi, n=10):
    scanner_plot(boxes)
    for t in np.linspace(0, t, n):
        line_plot(t,s)
    plt.show()


def cross4(t,s,a,xy):
    phis=0
    hpi=-np.pi/2
    return (xray_rectangle(t,s,a=a,xy=xy,phi=(phis-hpi))>0)\
            *(xray_rectangle(t,s,a=a,xy=xy,phi=(phis))>0)\
                *(xray_rectangle(t,s,a=a,xy=xy,phi=(phis+hpi))>0)\
                    *(xray_rectangle(t,s,a=a,xy=xy,phi=(phis+2*hpi))>0)


def cross3(t,s,a,xy, phis=0):
    hpi=np.pi/2
    return (xray_rectangle(t,s,a=a,xy=xy,phi=(phis))>0)\
            *(xray_rectangle(t,s,a=a,xy=xy,phi=(phis+hpi))>0)\
                *(xray_rectangle(t,s,a=a,xy=xy,phi=(phis+2*hpi))>0)


def all_cross3(t,s,a,xy):
    cc=0
    for i in range(4):
        cc += cross3(t,s,a=a,xy=xy,phis=i*np.pi/2)
    return cc


def all_cross3_scanner(boxes, t, s):
    a = [boxes.layerLength, boxes.layerDepth]
    xy = [boxes.Rx-boxes.shift-boxes.layerLength/2., boxes.Ry+boxes.shift+boxes.layerDepth/2.]
    return all_cross3(t,s,a=a,xy=xy)


def plot_intersect_in_scanner(boxes,ct, phi=0):
    scanner_plot(boxes)
    thetas = np.linspace(0,np.pi,ct)
    lim = boxes.Rx#/np.sqrt(2)
    s = np.linspace(-lim,lim,ct)
    tt, ss = np.meshgrid(thetas, s)
    hpi = np.pi/2
    
    a = [boxes.layerLength, boxes.layerDepth]
    xy = [boxes.Rx-boxes.shift-boxes.layerLength/2., boxes.Ry+boxes.shift+boxes.layerDepth/2.]
    extra = np.array([[[boxes.layerLength-2.*boxes.Rx, boxes.layerDepth], [2.*boxes.Rx, boxes.layerDepth]], \
                    [[-boxes.shift-boxes.layerLength/2., boxes.Ry+boxes.shift+boxes.layerDepth/2.], [-boxes.shift, boxes.Ry+boxes.shift+boxes.layerDepth/2.]]])
    
    if extra is None:
        condi = (xray_rectangle(tt,ss,a=a,xy=xy,phi=phi)>0) \
                * (xray_rectangle(tt,ss,a=a,xy=xy,phi=phi+hpi)==0)
    else:
        condi = (xray_rectangle(tt,ss,a=a,xy=xy,phi=phi)>0) \
                * (xray_rectangle(tt,ss,a=extra[0,0],xy=extra[1,0],phi=(phi-hpi))==0) \
                * (xray_rectangle(tt,ss,a=extra[0,1],xy=extra[1,1],phi=(phi+hpi))==0)
    for t, s in zip(tt[condi].ravel(),ss[condi].ravel()):
        line_plot(t,s)
    
    fig=plt.gcf()
    plt.show()
    return fig


def compare_with_simulation(boxes, phi, s):
    plt.hist(phi[(s<0.5)&(s>-0.5)], bins=100, density=True, stacked=True, alpha=0.5)
    thetas1000 = np.linspace(0,np.pi,1000)
    xx = xray_scanner(boxes,thetas1000,np.zeros_like(thetas1000)+0.)
    plt.plot(thetas1000, xx/(np.pi*np.sum(xx)/xx.size), "k--")

    fig=plt.gcf()
    #plt.show()
    return fig


if __name__ == '__main__':
    from simuPET.simulations.simulator import simulate_muPET
    boxes, *_, phi, s = simulate_muPET()

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

    # All lines with the same theta
    plot_alls_in_scanner(boxes, np.pi/4, s=boxes.Rx, n=15)

    # Choose one line and plot it on the sinogram zonage
    to, so = .78, 6.
    pl = plt.imshow(all_cross3_scanner(boxes, tt, ss)); plt.colorbar(pl)
    plt.scatter((tt.shape[0]/np.max(tt))*to,tt.shape[0]/2.+(ss.shape[0]/(2*np.max(ss)))*so)
    plt.show()
    plot_line_in_scanner(boxes,to,so)
    plt.show()

    # Plot all lines that intersect with top box with the equation constrain
    plot_intersect_in_scanner(boxes, 20, 0)

    # Plot scanner sensitivty
    sens = xray_scanner(boxes,tt,ss)
    pl=plt.imshow(sens, extent=[0,np.pi,-lim,lim], aspect="auto"); plt.colorbar(pl); plt.clim(0.02,0.12)
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
    pl=plt.imshow(xrs,extent=[0,np.pi,-lim,lim], aspect="auto"); plt.colorbar(pl)
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
    compare_with_simulation(boxes, phi, s)