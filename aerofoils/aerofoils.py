"""Main module."""

import numpy as np


class NACA:

    def __init__(self, m, p, t, finite_trailing_edge=True):
        self.m = m
        self.p = p
        self.t = t
        self.finite_trailing_edge = finite_trailing_edge

    def __call__(self, x):
        theta = self._angle(x, self.m, self.p)
        z = self._mean_camber_line(x, self.m, self.p)
        y = self._poly(x, self.t, self.finite_trailing_edge)

        xu = x - y * np.sin(theta)
        xl = x + y * np.sin(theta)

        yu = z + y * np.cos(theta)
        yl = z - y * np.cos(theta)

        return z, (xl, yl), (xu, yu)

    @staticmethod
    def _poly(x, t, finite_trailing_edge=False):

        pows = np.expand_dims(np.maximum(0.5, np.arange(5)), axis=-1)

        a0 = +0.2969
        a1 = -0.1260
        a2 = -0.3516
        a3 = +0.2843

        if finite_trailing_edge:
            a4 = -0.1015  # For finite thick TE
        else:
            a4 = -0.1036  # For zero thick TE

        a = np.array([a0, a1, a2, a3, a4])

        # (5,) (5, N) -> (N,)
        return 5. * t * a.dot(x**pows)

    @staticmethod
    def _mean_camber_line(x, m, p):
        return np.where(np.less_equal(x, p), 
                        m / p**2 * x * (2*p - x), 
                        m / (1 - p)**2 * (1 - 2*p + x) * (1 - x))

    @staticmethod
    def _angle(x, m, p):
        dyc_dx = np.where(np.less_equal(x, p), 
                          m / p**2 * (2*p - 2*x), 
                          m / (1-p)**2 * (2*p - 2*x))
        theta = np.arctan(dyc_dx)
        return theta
