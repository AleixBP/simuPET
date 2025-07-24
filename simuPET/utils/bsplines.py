from simuPET import array_lib as np


def bs(eval, deg, positive=True):
    if positive:
        ax = eval
    else:
        ax = np.abs(eval)

    # h_sup = 0.5*(1+deg)
    if deg == 0:
        return ax < 0.5
    elif deg == 1:
        return (ax < 1.0) * (1.0 - ax)
    elif deg == 3:
        return (ax < 2.0) * (
            (ax < 1) * (2.0 / 3.0 - ax**2 + 0.5 * ax**3)
            + (ax >= 1) * (1.0 / 6.0) * ((2.0 - ax) ** 3)
        )


if __name__ == "__main__":
    print(bs(np.linspace(-2, 2, 200), 3))
