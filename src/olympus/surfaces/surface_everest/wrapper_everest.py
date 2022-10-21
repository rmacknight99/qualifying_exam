#!/usr/bin/env python

from olympus import Logger
from olympus.surfaces import GaussianMixture


class Everest(GaussianMixture):
    def __init__(self, noise=None):
        """Gaussian Mixture surface derived from the GaussianMixture generator using ``random_seed=8848``.

        Args:
            noise (Noise): Noise object that injects noise into the evaluations of the surface. Default is None.
        """
        peak = 8848
        value_dim = 1
        GaussianMixture.__init__(
            self,
            param_dim=2,
            num_gauss=4,
            cov_scale=0.1,
            diagonal_cov=False,
            noise=noise,
            random_seed=peak,
        )

    @property
    def minima(self):
        # TODO: find minimum numerically and hardcode
        message = "Unknown minima: these need to be found numerically"
        Logger.log(message, "WARNING")
        return None

    @property
    def maxima(self):
        # TODO: find minimum numerically and hardcode
        message = "Unknown maxima: these need to be found numerically"
        Logger.log(message, "WARNING")
        return None
