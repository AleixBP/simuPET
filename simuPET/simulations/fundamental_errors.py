from simuPET import array_lib as np
# changed: fwhm of aco angle error, fwhm of range into mm (taking out e-3), added exponential range distro

def total_disintegrations(activity, time):
    """
    Computes the total number of disintegrations of a given activity over a given amount of time.

    Parameters
    ----------
    mass : activity
        Radioactivity in [Bq]
    time : duration of the scanning process in [s]

    Returns
    -------
    total number of disintegrations
    """

    return activity * time

def total_disintegrations_2(tracer, mass, time):
    """
    Computes the total number of disintegrations of a given mass of radiotracer over a given amount of time.

    Parameters
    ----------
    tracer : string
        name of the radioactive tracer used for simulation. Options are ["18F", "11C", "13N", "15O"].
    mass : float
        amount of radioactive tracer injected in the patient in [g]
    time : duration of the scanning process in [min]

    Returns
    -------
    total number of disintegrations
    """
    NA = 6.02214179e23

    if tracer == "18F":
        M = 18.000937
        t_half = 109.77
    elif tracer == "11C":
        M = 11.01074
        t_half = 20
    elif tracer == "13N":
        M = 13.005739
        t_half = 9.97
    elif tracer == "15O":
        M = 15.0030654
        t_half = 2.0333333333
    else:
        print("Error: tracer must be 18F, 11C, 13N, or 15O.")

    rad_const = 1*NA*np.ln(2) / (M * t_half)

    return rad_const * time * mass


# 3. Positron decay (mean free path until annihilation)
def f_decay(x, tracer="18F"):
    """
    Probability density function of the distance in [mm] traveled by the positron after its emission. Each radioactive
    tracer has its own characteristic function for distance travelled.

    Parameters
    ----------
    x : abscissa
        scalar or array-like
    tracer : string
        name of the radioactive tracer used for simulation. Options are ["18F", "11C", "13N", "15O"]. Default is "18F".
    Returns
    -------
    scalar or `x.shape` array of f_decay(x) values.
    """

    if tracer == "18F":
        C = 0.516
        k1 = 0.379e2
        k2 = 0.031e2
    elif tracer == "11C":
        C = 0.488
        k1 = 0.238e2
        k2 = 0.018e2
    elif tracer == "13N":
        C = 0.426
        k1 = 0.202e2
        k2 = 0.014e2
    elif tracer == "15O":
        C = 0.379
        k1 = 0.181e2
        k2 = 0.009e2
    else:
        print("Error: tracer must be 18F, 11C, 13N, or 15O.")

    f = (C*np.exp(-k1*np.abs(x)) + (1-C)*np.exp(-k2*np.abs(x)))/(2*C/k1 + 2*(1-C)/k2)

    return f

def F_decay(x, tracer="18F"):
    """
    Probability cumulative function of the distance in [mm] traveled by the positron after its emission. Each radioactive
    tracer has its own characteristic function for distance travelled.

    Parameters
    ----------
    x : abscissa
        scalar or array-like
    tracer : string
        name of the radioactive tracer used for simulation. Options are ["18F", "11C", "13N", "15O"]. Default is "18F".
    Returns
    -------
    scalar or `x.shape` array of F_decay(x) values.
    """

    if tracer == "18F":
        C = 0.516
        k1 = 0.379e2
        k2 = 0.031e2
    elif tracer == "11C":
        C = 0.488
        k1 = 0.238e2
        k2 = 0.018e2
    elif tracer == "13N":
        C = 0.426
        k1 = 0.202e2
        k2 = 0.014e2
    elif tracer == "15O":
        C = 0.379
        k1 = 0.181e2
        k2 = 0.009e2
    else:
        print("Error: tracer must be 18F, 11C, 13N, or 15O.")

    F = (C*k2*(1-np.exp(-k1*x)) + (1-C)*k1*(1-np.exp(-k2*x)))/(2*(C*k2 - C*k1 + k1))

    return F


def distance_decay(tracer="18F", size=None):
    """
    This function randomly samples from the positron decay pdf f_decay to return random travelled distances py positrons
    in [mm]. Each radioactive tracer has its own characteristic function for distance travelled.

    Parameters
    ----------
    tracer : string
        name of the radioactive tracer used for simulation. Options are ["18F", "11C", "13N", "15O"]. Default is "18F".
    size : scalar, tuple or None
        size of the sample array returned
    Returns
    -------
    array or scalar of size `size` with random samples drawn from f_decay
    """

    max_fdecay = f_decay(0, tracer)
    distance = np.zeros(size).ravel()

    for i in range(len(distance)):
        distance[i] = rejection_sampling(f_decay, max_fdecay, tracer)

    distance = distance.reshape(size)

    return distance

def distance_decay_gauss(tracer="18F", size=None):
    """
    This function randomly samples from a Gaussian pdf that approximates the positron decay distribution to
    return random travelled distances by positrons in [mm]. Each radioactive tracer has its own characteristic function for
    distance travelled.

    Parameters
    ----------
    tracer : string
        name of the radioactive tracer used for simulation. Options are ["18F", "68Ga", "15O"]. Default is "18F".
    size : scalar, tuple or None
        size of the sample array returned
    Returns
    -------
    array or scalar of size `size` with random samples drawn from a Gaussian approximating f_decay (in mm)
    """

    if tracer == "18F":
        fwhm = 0.102 #e-3 #0.54e-3
    elif tracer == "68Ga":
        fwhm = 2.83 #e-3
    elif tracer == "15O":
        fwhm = 2.48 #e-3
    mean_freepath = 0
    sigma_freepath = fwhm / (2*np.sqrt(2)*np.sqrt(np.log(2)))

    distance = np.random.normal(mean_freepath, sigma_freepath, size=size)

    return distance

# 4. Photon emission: simulation of acolinearity
def f_acolinearity(x):
    """
    Probability density function of the photon emission acolinearity. It returns the deviation from 180° of the
    angle between the two photons in [mrad].

    Parameters
    ----------
    x : abscissa
        scalar or array-like
    Returns
    -------
    scalar or `x.shape` array of f_angle(x) values.
    """

    B = (346.5*((0.124*x)**2 + 1)**2 + 5330*((0.124*x)**2 + 1)- 4264)/((0.124*x)**2 + 1)**5
    M = 0
    b = 1
    m = 0
    return (b*B + m*M)


def angle_deviation(size=None):
    """
    This function randomly samples from the positron decay pdf f_angle to return the acolinearity deviation angle
    in [rad].

    Parameters
    ----------
    size : scalar, tuple or None
        size of the sample array returned
    Returns
    -------
    array or scalar of size `size` with random samples drawn from f_acolinearity
    """
    max_facol = f_acolinearity(0)
    angle = np.zeros(size).ravel()

    for i in range(len(angle)):
        angle[i] = rejection_sampling(f_acolinearity, max_facol)

    angle = angle.reshape(size)

    return angle/1e3

def angle_deviation_gauss(size=None):
    """
    This function randomly samples from a Gaussian pdf that approximates the acolinearity angle distribution to
    return acolinearity deviantion angles in [rad].

    Parameters
    ----------
    size : scalar, tuple or None
        size of the sample array returned
    Returns
    -------
    array or scalar of size `size` with random samples drawn from a Gaussian approximating f_acolinearity
    """
    #sigma_acol = np.deg2rad(0.25) / (2*np.sqrt(2)*np.sqrt(np.log(2))) #fwhm to sigma
    sigma_acol = np.deg2rad(0.47) / (2*np.sqrt(2)*np.sqrt(np.log(2))) #fwhm to sigma
    mean_acol = 0
    angle = np.random.normal(0, sigma_acol, size=size)

    return angle


def rejection_sampling(pdf, max_pdf, *args):
    
    accepted = False
    y = 0
    
    while(not accepted):
        x = np.random.normal(loc=0, scale=1)
        u = np.random.uniform(low=0.0, high=1.0)

        if u  <= pdf(x, *args) / max_pdf:
            y = x
            accepted = True
        
        return y
    

def distance_decay_gauss_alternative(tracer="18F", size=None):
    
    if False:
        def gauss_func(x, mu=0, sigma=1):
            return np.exp(-((x-mu)**2)/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
        
        max_fdecay = f_decay(0, tracer)
        
        stdev = 0.102/(2*np.sqrt(2)*np.sqrt(np.log(2)))
        point_coords = np.random.normal(loc=0, scale=stdev, size=2*size)
        
        
        m_ct = max_fdecay # missing an upper bound of fdecay/gauss, is there any?
        thin_prob = f_decay(point_coords)/(m_ct*gauss_func(point_coords, sigma=stdev))
        
        points_to_keep = thin_prob > np.random.uniform(0, 1, size)
        
        point_coords = point_coords[points_to_keep]
        
        return point_coords
    
    elif False:

        fwhm = 0.102
        #2*np.ln(2)*b = fwhm
        return np.random.laplace(loc=0, scale=fwhm/float(2*np.log(2)), size=size)
    
    elif True:
        k1 = 23.5554
        k2 = 3.7588
        c = 0.5875
        
        l1 = np.random.laplace(loc=0, scale=1/k1, size=size)
        l2 = np.random.laplace(loc=0, scale=1/k2, size=size)
        
        coin = c/(c+(1-c)*k1/k2)
        
        decide = np.random.uniform(0, 1, size)<coin
        
        return l1*decide + l2*(decide-1)
    