import numpy as np
from plotting_routines import plot_with_slider


def main():
    t = np.load("t.npy")
    x = np.load("x.npy")
    u = np.load("u.npy")

    plot_with_slider(t, x, u)

if __name__ == '__main__':
    main()
