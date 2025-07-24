from simuPET import array_lib as np
from plt import plt

mixed = True


r = 14.0
off = 5
if mixed:
    num_points = 7  # 5
    solid = np.pi
    solid_bool = True
else:
    num_points = 16
    solid = 2 * np.pi
    solid_bool = False
angles = np.linspace(0, solid, num_points, endpoint=solid_bool)
sr = r + off  # r*1.33 #1.33 is from lai 2021
x = sr * np.cos(angles)
y = sr * np.sin(angles)

# plt.axes().set_aspect('equal');
circle = plt.Circle((0, 0), r, color="b", fill=False)
fig, ax = plt.subplots(figsize=(10, 10))
ax.add_patch(circle)
ax.set_aspect("equal")
for i in range(num_points):
    for j in range(i, num_points):
        plt.plot([x[i], x[j]], [y[i], y[j]], "r", alpha=0.3)
# plt.savefig("sampling_mupetno_scheme_16_off1.pdf")

plt.scatter([0.0], [0.0], color="b")

# 5 times the DOI and 5 times the pitch


if mixed:
    rat_doi = 5
    rat_pitch = 5
    all_scanner = False
    precise = False
    tweak = True

else:
    # Individual Plot
    rat_doi = 2
    rat_pitch = 1
    all_scanner = True
    precise = True
    layers = 1
    # layers = 2

det_size = (solid * sr / num_points) / rat_pitch


# left
lengl = r + off if not tweak else r
ofl = r if all_scanner else 0  # r
num_points_mu1 = int(np.round((lengl + np.abs(ofl)) / det_size))
yml = np.linspace(ofl, -lengl, num_points_mu1, endpoint=True)
xml = -r - np.linspace(1, off, rat_doi, endpoint=True)

if precise:
    # yml = np.linspace(-r-off, r+off, num_points_mu1-1, endpoint=True)
    # xml = -r-np.linspace(1, off, 2, endpoint=True)
    yml = np.linspace(-r - 1, r + 1, 5, endpoint=True)
    diff = np.abs(yml[1] - yml[0])
    xml = -r - np.linspace(1, diff + 1, layers, endpoint=True)


# right
lengr = r
ofr = r + off if all_scanner else 0  # r+off
num_points_mu = int(np.round((lengr + np.abs(ofr)) / det_size))
ymr = np.linspace(ofr, -lengr, num_points_mu, endpoint=True)
xmr = r + np.linspace(1, off, rat_doi, endpoint=True)
# if all_scanner: ymr = yml + np.abs(ymr[1]-ymr[0])

if precise:
    # ymr = yml
    # xmr = xmr
    ymr = yml
    xmr = r + np.linspace(1, diff + 1, layers, endpoint=True)


# top
lengt = r + off
oft = r
num_points_mu = int(np.round((lengt + np.abs(oft)) / det_size))
xmt = np.linspace(oft, -lengt, num_points_mu, endpoint=True)
ymt = r + np.linspace(1, off, rat_doi, endpoint=True)

if precise:
    # xmt = np.linspace(-r-off, r+off, num_points_mu-1, endpoint=True)
    # ymt = r+np.linspace(1, off, 2, endpoint=True)
    xmt = np.linspace(
        -r - 1 - diff * (layers - 1),
        r + 1 + diff * (layers - 1),
        7 - 2 * (2 - layers),
        endpoint=True,
    )
    ymt = r + np.linspace(1, diff, layers, endpoint=True)


# bottom
lengb = r if not tweak else r + off
ofb = r + off
num_points_mu = (
    int(np.round((lengb + np.abs(ofb)) / det_size)) if not tweak else 19
)  # 15 # int((num_points*5*5-num_points_mu1*2*rat_doi)/rat_doi)
xmb = np.linspace(ofb, -lengb, num_points_mu, endpoint=True)
ymb = -r - np.linspace(1, off, rat_doi, endpoint=True)
# if all_scanner: xmb = xmt + np.abs(ymr[1]-ymr[0])

if precise:
    # xmb = xmt
    # ymb = ymb
    xmb = xmt
    ymb = -r - np.linspace(1, diff, layers, endpoint=True)


alpha = 0.2  # min 0.002
circle = plt.Circle((0, 0), r, color="b", fill=False)
fig, ax = plt.subplots(figsize=(10, 10))
ax.add_patch(circle)
ax.set_aspect("equal")
block_to_block(xml, yml, xmr, ymr, r)
block_to_block(xml, yml, xmb, ymb, r)
block_to_block(xmr, ymr, xmb, ymb, r)

block_to_block(xmt, ymt, xmb, ymb, r)
block_to_block(xml, yml, xmt, ymt, r)
block_to_block(xmr, ymr, xmt, ymt, r)
# plt.savefig("sampling_mupet_scheme_1layers.pdf")


# for mixed
import tikzplotlib

alpha = 0.01  # 0.02#min 0.002
circle = plt.Circle((0, 0), r, color="b", fill=False)
fig, ax = plt.subplots(figsize=(10, 10))
ax.add_patch(circle)
ax.set_aspect("equal")
for i in range(num_points):
    for j in range(i, num_points):
        plt.plot([x[i], x[j]], [y[i], y[j]], "r", alpha=0.3)
block_to_block(xml, yml, xmr, ymr, r, alpha)
block_to_block(xml, yml, xmb, ymb, r, alpha)
block_to_block(xmr, ymr, xmb, ymb, r, alpha)
plt.scatter([0.0], [0.0], color="b")
# plt.savefig("sampling_mupet_nomupet_scheme_7insteadof5c.pdf")
# tikzplotlib.save("sampling_mupet_nomupet_scheme_7insteadof5c.tex")


def block_to_block(x1, y1, x2, y2, r, alpha=0.2):
    for ll in x1:
        for yl in y1:
            for lr in x2:
                for yr in y2:
                    # plt.plot([ll,lr], [yl,yr], 'k',alpha=alpha)
                    if circle_line_segment_intersection(
                        [0, 0], r, [ll, yl], [lr, yr]
                    ) and not (np.abs(ll) > r and np.abs(lr) > r and ll * lr > 0):
                        plt.plot([ll, lr], [yl, yr], "k", alpha=alpha)
                    else:
                        pass


# https://stackoverflow.com/questions/30844482/what-is-most-efficient-way-to-find-the-intersection-of-a-line-and-a-circle-in-py
def circle_line_segment_intersection(
    circle_center, circle_radius, pt1, pt2, full_line=True, tangent_tol=1e-9
):
    """Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.

    :param circle_center: The (x, y) location of the circle center
    :param circle_radius: The radius of the circle
    :param pt1: The (x, y) location of the first point of the segment
    :param pt2: The (x, y) location of the second point of the segment
    :param full_line: True to find intersections along full line - not just in the segment.  False will just return intersections within the segment.
    :param tangent_tol: Numerical tolerance at which we decide the intersections are close enough to consider it a tangent
    :return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at which the circle intercepts a line segment.

    Note: We follow: http://mathworld.wolfram.com/Circle-LineIntersection.html
    """

    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx**2 + dy**2) ** 0.5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius**2 * dr**2 - big_d**2

    # if discriminant < 0:  # No intersection between circle and line
    #    return []
    return discriminant >= 0
