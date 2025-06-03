import astropy.units as u
import jax.numpy as jnp
import numpy as np
from orbix.constants import Msun2kg, rad2arcsec
from orbix.system import Planets
from tqdm import tqdm

from EXOSIMS.Completeness.BrownCompleteness import BrownCompleteness


class OrbixCompleteness(BrownCompleteness):
    """Completeness model that uses orbix to calculate the detection probability."""

    def orbix_setup(self, trig_solver, SS):
        """Setup completeness by generating a set of orbits and propagating them."""
        PPop = self.PlanetPopulation
        TK, TL = SS.TimeKeeping, SS.TargetList

        # sample quantities
        a, e, p, Rp = PPop.gen_plan_params(self.Nplanets)
        i, W, w = PPop.gen_angles(self.Nplanets)

        _a, _e, _p, _Rp = jnp.array(a), jnp.array(e), jnp.array(p), jnp.array(Rp)
        _i, _W, _w = jnp.array(i), jnp.array(W), jnp.array(w)

        # Mean anomaly should be uniformly distributed
        _M0 = jnp.array(np.random.uniform(high=2.0 * np.pi, size=self.Nplanets))
        _t0 = jnp.zeros(self.Nplanets)

        # Mp doesn't matter for completeness
        _Mp = jnp.ones(self.Nplanets)
        # Distance and mass will be handled later
        _dist = jnp.ones(self.Nplanets)
        _Ms = jnp.ones(self.Nplanets) * Msun2kg
        self._planets = Planets(_Ms, _dist, _a, _e, _W, _i, _w, _M0, _t0, _Mp, _Rp, _p)

        # Propagate the orbits
        end_time = TK.missionLife_d
        self.comp_times = jnp.arange(0, end_time, 1)

        # Shapes: (Nplanets, Ntimes)
        self.s, self.dMag = self._planets.j_s_dMag(trig_solver, self.comp_times)

        # Alpha depends on the star's distance, computing it for each star
        self.alpha_factors = rad2arcsec / TL.dist.to_value(u.AU)

        # Compute the completeness per intTime for each star
        fZ0 = jnp.array([SS.ZodiacalLight.fZ0.to_value(SS.fZ_unit)])
        # Get the s/dMag values at initial timestep
        # (shouldn't matter which timestep as long as we have a large enough sample)
        _s = self.s[:, 0].reshape(-1, 1)
        _dMag = self.dMag[:, 0].reshape(-1, 1)
        self.s = np.array(self.s)
        self.dMag = np.array(self.dMag)

        # Load the dMag0 grid of the detection mode
        mode = SS.base_det_mode
        _dMag0s = SS.dMag0s[mode["hex"]]
        # This is an array of the completeness per intTime for each star
        # with the integration times provided by the dMag0 grid
        self.orbix_comp = np.zeros((TL.nStars, _dMag0s[0].int_times.shape[0]))
        self.comp_div_intTime = np.zeros_like(self.orbix_comp)
        self.best_intTime = np.zeros(TL.nStars)
        self.best_comp_div_intTime = np.zeros(TL.nStars)

        for sInd in tqdm(range(TL.nStars), desc="Generating completeness per intTime"):
            _dMag0Grid = _dMag0s[sInd]
            _alpha = _s * self.alpha_factors[sInd]
            self.orbix_comp[sInd] = _dMag0Grid.pdet_alpha_dMag(
                trig_solver, _alpha, _dMag, fZ0
            )[0]
            self.comp_div_intTime[sInd] = self.orbix_comp[sInd] / _dMag0Grid.int_times

            if np.any(self.orbix_comp[sInd] > 0):
                # Find the best intTime for this star
                self.best_intTime[sInd] = _dMag0Grid.int_times[
                    np.argmax(self.comp_div_intTime[sInd])
                ]
                self.best_comp_div_intTime[sInd] = np.max(self.comp_div_intTime[sInd])
