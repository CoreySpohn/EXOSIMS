import astropy.units as u
import numpy as np

from EXOSIMS.PlanetPopulation.DulzPlavchan import DulzPlavchan
from EXOSIMS.util._numpy_compat import copy_if_needed


class AlbedoByRadiusDulzPlavchanEarthsOnly(DulzPlavchan):
    """Planet Population module based on occurrence rate tables from Shannon Dulz and
    Peter Plavchan.

    NOTE: This assigns constant albedo based on radius ranges.
    NOTE: This module filters planets to Earth candidates only based on specific criteria.

    Attributes:
        SAG13coeffs (float 4x2 ndarray):
            Coefficients used by the SAG13 broken power law. The 4 lines
            correspond to Gamma, alpha, beta, and the minimum radius.
        Gamma (float ndarray):
            Gamma coefficients used by SAG13 broken power law.
        alpha (float ndarray):
            Alpha coefficients used by SAG13 broken power law.
        beta (float ndarray):
            Beta coefficients used by SAG13 broken power law.
        Rplim (float ndarray):
            Minimum radius used by SAG13 broken power law.
        SAG13starMass (astropy Quantity):
            Assumed stellar mass corresponding to the given set of coefficients.
        mu (astropy Quantity):
            Gravitational parameter associated with SAG13starMass.
        Ca (float 2x1 ndarray):
            Constants used for sampling.
        ps (float nx1 ndarray):
            Constant geometric albedo values.
        Rb (float (n-1)x1 ndarray):
            Planetary radius break points for albedos in earthRad.
        Rbs (float (n+1)x1 ndarray):
            Planetary radius break points with 0 padded on left and np.inf
            padded on right

    """

    def __init__(
        self,
        starMass=1.0,
        occDataPath=None,
        esigma=0.175 / np.sqrt(np.pi / 2.0),
        ps=[0.2, 0.5],
        Rb=[1.4],
        **specs,
    ):
        self.ps = np.array(ps, ndmin=1, copy=copy_if_needed)
        self.Rb = np.array(Rb, ndmin=1, copy=copy_if_needed)
        specs["prange"] = [np.min(ps), np.max(ps)]
        DulzPlavchan.__init__(
            self, starMass=starMass, occDataPath=occDataPath, esigma=esigma, **specs
        )

        # check to ensure proper inputs
        assert (
            len(self.ps) - len(self.Rb) == 1
        ), "input albedos must have one more element than break radii"
        self.Rbs = np.hstack((0.0, self.Rb, np.inf))

        # albedo is constant for planetary radius range
        self.pfromRp = True

    def gen_plan_params(self, n):
        """Generate semi-major axis (AU), eccentricity, geometric albedo, and
        planetary radius (earthRad)

        Semi-major axis and planetary radius are jointly distributed.
        Eccentricity is a Rayleigh distribution. Albedo is a constant value
        based on planetary radius.

        Only Earth-like candidates are returned, based on specific filtering criteria.

        Args:
            n (integer):
                Number of samples to generate

        Returns:
            tuple:
            a (astropy Quantity array):
                Semi-major axis in units of AU
            e (float ndarray):
                Eccentricity
            p (float ndarray):
                Geometric albedo
            Rp (astropy Quantity array):
                Planetary radius in units of earthRad

        """
        n = self.gen_input_check(n)

        # Initialize arrays to store Earth candidates
        a_earth = np.array([]) * u.AU
        e_earth = np.array([])
        p_earth = np.array([])
        Rp_earth = np.array([]) * u.earthRad

        # Generate a large initial batch (10x requested number)
        initial_batch_size = n * 10
        batch_size = initial_batch_size

        # Keep generating planets until we have enough Earth candidates
        while len(a_earth) < n:
            # Generate a batch of planets
            a_batch, e_batch, p_batch, Rp_batch = self._gen_plan_params_unfiltered(
                batch_size
            )

            # Filter to just Earth candidates
            # NB: a has not yet been scaled by \sqrt{L} so effectively
            # everything is at 1 solar luminosity
            inds = (
                (a_batch < 1.67 * u.AU)
                & (a_batch > 0.95 * u.AU)
                & (Rp_batch < 1.4 * u.R_earth)
                & (Rp_batch > 0.8 / np.sqrt(a_batch.to_value(u.AU)) * u.R_earth)
            )

            # Append Earth candidates to our arrays
            a_earth = np.hstack((a_earth, a_batch[inds]))
            e_earth = np.hstack((e_earth, e_batch[inds]))
            p_earth = np.hstack((p_earth, p_batch[inds]))
            Rp_earth = np.hstack((Rp_earth, Rp_batch[inds]))

            # If we're still not finding enough, increase the batch size
            if len(a_earth) < n:
                batch_size = batch_size * 2  # Double the batch size for next iteration

        # Return exactly n planets
        return a_earth[:n], e_earth[:n], p_earth[:n], Rp_earth[:n]

    def _gen_plan_params_unfiltered(self, n):
        """Generate unfiltered planet parameters.
        This is the original implementation without Earth candidate filtering.
        """
        n = self.gen_input_check(n)
        # generate semi-major axis and planetary radius samples
        a, Rp = self.gen_sma_radius(n)

        # check for constrainOrbits == True for eccentricity samples
        # constants
        C1 = np.exp(-(self.erange[0] ** 2) / (2.0 * self.esigma**2))
        ar = self.arange.to("AU").value
        if self.constrainOrbits:
            # restrict semi-major axis limits
            arcon = np.array(
                [ar[0] / (1.0 - self.erange[0]), ar[1] / (1.0 + self.erange[0])]
            )
            # clip sma values to sma range
            sma = np.clip(a.to("AU").value, arcon[0], arcon[1])
            # upper limit for eccentricity given sma
            elim = np.zeros(len(sma))
            amean = np.mean(ar)
            elim[sma <= amean] = 1.0 - ar[0] / sma[sma <= amean]
            elim[sma > amean] = ar[1] / sma[sma > amean] - 1.0
            elim[elim > self.erange[1]] = self.erange[1]
            elim[elim < self.erange[0]] = self.erange[0]
            # additional constant
            C2 = C1 - np.exp(-(elim**2) / (2.0 * self.esigma**2))
            a = sma * u.AU
        else:
            C2 = self.enorm
        e = self.esigma * np.sqrt(-2.0 * np.log(C1 - C2 * np.random.uniform(size=n)))
        # generate albedo from planetary radius
        p = self.get_p_from_Rp(Rp)

        return a, e, p, Rp

    def get_p_from_Rp(self, Rp):
        """Generate constant albedos for radius ranges

        Args:
            Rp (astropy Quantity array):
                Planetary radius with units of earthRad

        Returns:
            float ndarray:
                Albedo values

        """
        Rp = np.array(Rp.to("earthRad").value, ndmin=1, copy=copy_if_needed)
        p = np.zeros(Rp.shape)
        for i in range(len(self.Rbs) - 1):
            mask = np.where((Rp >= self.Rbs[i]) & (Rp < self.Rbs[i + 1]))
            p[mask] = self.ps[i]

        return p
