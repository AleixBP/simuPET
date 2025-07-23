from simuPET import array_lib as np
import plt as plt


def simulate_poisson_nohomo(func, domain, max_lam):
    """
    Simulates non-homogeneous Poisson process in n-dimensional hyper-rectangles given the rate/intensity function.

    Parameters
    ----------
    func : function
        non-negative function defined on an n-dimensional hyper-rectangle. This function represents the intensity
        of the process.
    domain : array-like
        `(n,2)` array of upper/lower bounds for each dimension, as `[[lower_1,upper_1], ..., [lower_n,upper_n]]`,
        representing the n-dimensional hyper-rectangle where `func` is defined.
    max_lam : float
        an upper bound for `func` over the `domain`.

    Returns
    -------
    `(m,n)` array of simulated points, where `m` is the number of points sampled from the non-homogeneous Poisson
    process.
    """

    # Extract relevant parameters of the domain
    area = np.prod(np.diff(domain))
    dim = domain.shape[0]

    # Generate homogeneous Poisson process with rate corresponding to the upper bound
    num_points = int(np.random.poisson(lam=max_lam*area)) #int because cupy returns array
    point_coords = np.random.uniform(low=domain[:, 0], high=domain[:, 1], size=(num_points, dim))

    # Thin the homogeneous Poisson process to obtain the non-homogeneous one  sing the acceptance ratio func(x)/max_lam
    # Pass n-dimensional locations to func() as n vectors of the same size
    thin_prob = func(*point_coords.T)/max_lam
    # Evaluate the rejection event (biased coin flips)
    points_to_keep = thin_prob > np.random.uniform(0, 1, num_points)
    # Thin the observations
    point_coords = point_coords[points_to_keep, :]

    # Handle compatibility with estimate.py for 1 dimension
    if dim == 1:
        point_coords = point_coords[:, 0]

    return point_coords


def poisson_pf(ks, mu):
    # Compute Poisson probabilities for consecutive array of ks
    # Surely faster with gamma function / Stirling / tabulation
    logs = np.log(mu)*ks-mu-np.cumsum(np.log(ks))-np.sum(np.log(np.arange(1, np.min(ks))))
    # lucky behavior for min as sum of empty list is zero
    return np.exp(logs)


def mean_var_test(func, domain, max_l, num_sim=10000, plot_converg=False, plot_distro=False, plot_positions=False):

    # Check expectation and variance of the number of simulated points in the domain
    # should be the same as the integral of the rate/intensity function
    from scipy.integrate import nquad
    int_lam, _ = nquad(func, domain)

    # num_sim_points = 0
    num_sim_points = np.zeros(num_sim)
    points_coords = []

    for sim in range(num_sim):
        sim_points = simulate_poisson_nohomo(func, domain, max_l)
        # num_sim_points += len(sim_points)
        num_sim_points[sim] = len(sim_points)  # to count number of simulated points
        if plot_positions:
            points_coords += [sim_points]  # to assess positions of simulated points

        # Plot every power of 10 to check how mean and var converge to the integral
        if plot_converg and sim > 0 and np.log10(sim+1) % 1 == 0.:
            plt.scatter(sim, np.mean(num_sim_points[:sim]), c="b")
            plt.scatter(sim, np.var(num_sim_points[:sim]), c="r")

    if plot_converg:
        print("Plot: mean number of simulations (blue) and the variance (red)" +
              "converging to the integral (line) as more points are simulated.")
        plt.axhline(int_lam, 0, num_sim, c="k")
        plt.semilogx()
        plt.show()

    if plot_distro:
        print("Plot: histogram of the distribution of the number of simulated points" +
              "compared with a Poisson distribution with parameter the integral of the rate/intensity.")
        num_range = [np.min(num_sim_points), np.max(num_sim_points)]
        a_num_arange = np.arange(num_range[0], num_range[1]+1)
        num_arange = a_num_arange[:-1]
        sim_pdf, _ = np.histogram(num_sim_points, bins=a_num_arange-.5, density=True)  # center the bins

        plt.plot(num_arange, sim_pdf, "b-", label="simulated")
        plt.plot(num_arange, poisson_pf(num_arange, int_lam), "r-", label="theoretical")
        plt.legend()
        plt.show()

    if plot_positions:
        print("Plot: projected histogram of point locations x,y... distribution should match shape" +
              " of rate/intensity function")
        # Projection of a Poisson process is a Poisson process, project nD into 1D and compare with the projection of
        # the rate/intensity function
        dim = len(domain)

        points_coords = np.concatenate(points_coords)

        if dim == 1:
            xproj_points_coords = points_coords
        else:
            xproj_points_coords = points_coords[..., 0]  # works with 1D if no if in simulate_poisson_nohomo

        norm_xproj_sim_lam, bin_edges = np.histogram(xproj_points_coords, bins=50, density=True)
        # xproj_sim_lam = np.mean(num_sim_points)*norm_xproj_sim_lam
        bin_centers = (bin_edges[1:] + bin_edges[0:bin_edges.size - 1]) / 2

        if dim == 1:
            xproj_theo_lam = func(bin_centers)
        else:
            xproj_theo_lam = [nquad(lambda *args: func(x, *args), domain[1:])[0] for x in bin_centers]
            # integral over the projection

        # plt.scatter(bin_centers, xproj_sim_lam, c="b", label="simulated")
        plt.scatter(bin_centers, norm_xproj_sim_lam, c="b", label="simulated")
        plt.plot(bin_centers, xproj_theo_lam/int_lam, c="r", label="theoretical")
        plt.legend()
        plt.show()

    return int_lam, np.mean(num_sim_points), np.var(num_sim_points)


def simulate_poisson_and_geometric(func, domain, max_lam, sensitivity, nLayers):

    # Extract relevant parameters of the domain
    area = np.prod(np.diff(domain))
    dim = domain.shape[0]

    # Generate homogeneous Poisson process with rate corresponding to the upper bound
    num_points = int(np.random.poisson(lam=max_lam*area)) #int because cupy returns array

    # Geometric
    layers1 = np.random.geometric(sensitivity, size=num_points)
    layers2 = np.random.geometric(sensitivity, size=num_points)
    dont_discard = np.logical_and(layers1 <= nLayers, layers2 <= nLayers)
    layers1 = layers1[dont_discard]
    layers2 = layers2[dont_discard]
    num_points = len(layers1)

    # Back to Poisson
    point_coords = np.random.uniform(low=domain[:, 0], high=domain[:, 1], size=(num_points, dim))

    # Thin the homogeneous Poisson process to obtain the non-homogeneous one  sing the acceptance ratio func(x)/max_lam
    # Pass n-dimensional locations to func() as n vectors of the same size
    thin_prob = func(*point_coords.T)/max_lam
    # Evaluate the rejection event (biased coin flips)
    points_to_keep = thin_prob > np.random.uniform(0, 1, num_points)
    # Thin the observations
    point_coords = point_coords[points_to_keep, :]

    # Back to geometric
    layers1 = layers1[points_to_keep]
    layers2 = layers2[points_to_keep]

    # Handle compatibility with estimate.py for 1 dimension
    if dim == 1:
        point_coords = point_coords[:, 0]

    return point_coords, layers1, layers2