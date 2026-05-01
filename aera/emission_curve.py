"""
Defintion of the EmissionCurve class.

The EmissionCurve class represents a cubic emission curve and
offers the access to several properties of such a cubic emission
curve like (see below for the defintions of these quantities):
- the integral of the emission curve,
- the overshoot integral of the emission curve,
- the maximum annual overshoot of the emission curve,
- the "cost" (measure of the appropriateness) of the emission curve,
- etc.

The main usage of the EmissionCurve class is to find an
optimal (see below how "optimal" is defined) future
emission curve given several variables. See classmethod
`get_cheapest_curve` for extra information.

The optimal future emission curve is characterized as follows:
- The integral of the emission curve from "year X"
   to "target year" is approx. equal (as close as possible)
   to the remaining emission budget (REB).
- The exceedence emissions are as small as possible.
"""

from __future__ import annotations

import dataclasses
import copy as cp
import numpy as np
import numpy.typing as npt
import pandas as pd

@dataclasses.dataclass
class EmissionCurve:
    """
    Representation of a cubic emission curve.

    The emission curve is a continuation of a modelled/observed
    emission timeseries and is defined as:

        E(t) = a*t^3 + b*t^2 + c*t + d,

    with t as the number of years after the stocktake, and free
    parameters `a`, `b`, `c`, and `d`.
    """
    target_year_rel: int | npt.NDArray[np.int64]
    c: float | npt.NDArray[np.float64]
    d: float
    slope_t0: float
    reb: float
    slope_tm1: float
    target_year_max: int | None = None

    def __post_init__(self) -> None:
        if self.is_vector:
            self.c = np.asarray(self.c, dtype=float)[:, None]
            self.target_year_rel = np.asarray(self.target_year_rel, dtype=int)[None, :]
            self.target_year_max = int(np.max(self.target_year_rel))

    @property
    def is_vector(self) -> bool:
        """Check if the EmissionCurve instance is a vector of curves."""
        return np.ndim(self.target_year_rel) > 0 or np.ndim(self.c) > 0

    @property
    def a(self) -> float | npt.NDArray[np.float64]:
        """
        Parameter `a` of the emission curve.

        `a` depends on the choice of c and d and the length of the
        curve ty (targetyear) and is calculated as follows:

        a = (-2 * b * ty - c) / (3 * ty**2)
        """
        return (
            (-2 * self.b * self.target_year_rel - self.c) /
            (3 * self.target_year_rel**2)
        )

    @property
    def b(self) -> float | npt.NDArray[np.float64]:
        """
        Parameter `b` of the emission curve.

        `b` depends on the choice of c and d and the length of the
        curve ty (targetyear) and is calculated as follows:

        b = (-2 * c * ty - 3 * d) / ty**2
        """
        return (
            (-2 * self.c * self.target_year_rel - (3 * self.d)) /
            self.target_year_rel**2
        )

    @property
    def integral(self) -> float | npt.NDArray[np.float64]:
        """Calculate the integral over the entire emission curve."""
        values = self.get_values()
        if self.is_vector:
            return np.nansum(values, axis=2)
        return np.nansum(values)

    @property
    def overshoot_integral(self) -> float | npt.NDArray[np.float64]:
        """
        Calculate the integrated exceedence emissions (the ones that
        lead to an overshoot or undershoot) from t=0 to
        t=target_year_rel.
        """
        values = self.get_values()

        if self.is_vector:
            return 0.5 * (np.nansum(np.abs(values), axis=2) - np.abs(self.integral))
        return 0.5 * (np.nansum(np.abs(values)) - np.abs(self.integral))

    @property
    def slope_t1(self) -> float | npt.NDArray[np.float64]:
        """Slope of the curve at year X."""
        return self.get_values(t=1) - self.get_values(t=0)

    @property
    def slope_change(self) -> float | npt.NDArray[np.float64]:
        """Difference to slope right now."""
        return self.c - self.slope_t0

    @property
    def overshoot(self) -> float | npt.NDArray[np.float64]:
        """Maximum annual overshoot emission."""
        values = self.get_values()

        if self.is_vector:
            min_vals = np.nanmin(values, axis=2)
            max_vals = np.nanmax(values, axis=2)
            minmax_range = np.abs(max_vals - min_vals)
            return minmax_range - np.abs(values[:, :, 0])

        minmax_range = np.abs(np.max(values) - np.min(values))
        return minmax_range - np.abs(values[0])

    @property
    def curvature(self) -> float | npt.NDArray[np.float64]:
        """Curvature of the curve (sum of square of 2nd derivates)."""
        deriv2 = self.get_values_deriv2()

        if self.is_vector:
            return np.nansum(deriv2**2, axis=2) + (self.c - self.slope_tm1)**2
        return np.nansum(deriv2**2) + (self.c - self.slope_tm1)**2

    @property
    def reb_diff(self) -> float | npt.NDArray[np.float64]:
        """Difference to the remaining emission budget."""
        return self.integral - self.reb

    @property
    def cost(self) -> float | npt.NDArray[np.float64]:
        """
        Cost/quality of this emission curve.

        First, all curves that do not match the REB within ±5 Pg C are
        excluded by attributing a cost of the order of 1e7
        Second, all curves that have exceedence emissions above
        ±10 Pg C are excluded by attributing a cost of the order of 1e3
        Among the remaining curves, the one with the smalles curvature
        is chosen.

        """
        return (
            (1e7 * np.abs(self.reb_diff) *
                np.heaviside(np.abs(self.reb_diff) - 5, 1)) +
            (1e3 * np.abs(self.overshoot_integral) *
                np.heaviside(np.abs(self.overshoot_integral) - 10, 1)) +
            1 * self.curvature
        )

    def get_t_values(self, t: int | npt.ArrayLike | None = None) -> npt.NDArray[np.int64]:
        """Get the time values for which the emission curve is defined.

        Args:
            t (int or array-like, optional): Time values to return. If None,
                returns the default time values from 0 to target_year_rel.
        Returns:
            np.ndarray: Time values for which the emission curve is defined.

        """
        if t is None:
            if self.is_vector:
                t_out = np.arange(self.target_year_max + 1, dtype=int)
                t_out = t_out[None, None, :]
            else:
                t_out = np.arange(int(self.target_year_rel) + 1, dtype=int)
            return t_out

        return np.asarray(t, dtype=int)

    def get_values(self, t: int | npt.ArrayLike | None = None) -> npt.NDArray[np.float64]:
        """
        Emissions values for every time t.

        Args:
            t (array-like): Times for which the values should be
                calculated.
        Returns:
            values (array-like): List of values.
        """
        t = self.get_t_values(t)

        if self.is_vector:
            values = (
                self.a[:, :, None] * t**3 +
                self.b[:, :, None] * t**2 +
                self.c[:, :, None] * t +
                self.d
            )
            valid = t <= self.target_year_rel[:, :, None]
            values = np.where(valid, values, np.nan)
            return np.asarray(values, dtype=float)

        return np.asarray(
            self.a * t**3 + self.b * t**2 + self.c * t + self.d,
            dtype=float,
        )

    def get_values_deriv2(self, t: int | npt.ArrayLike | None = None) -> npt.NDArray[np.float64]:
        """
        2nd derivative of the emission curve at every time t.

        Args:
            t (array-like): Times for which the second derviative
                should be calculated.
        Returns:
            deriv2 (array-like): List of second derviatives.
        """
        t = self.get_t_values(t)

        if self.is_vector:
            valid = t <= self.target_year_rel[:, :, None]
            deriv2 = 6 * self.a[:, :, None] * t + 2 * self.b[:, :, None]
            return np.asarray(np.where(valid, deriv2, np.nan), dtype=float)

        return np.asarray(self.a * 6 * t + self.b * 2, dtype=float)

    def get_equation_str(self) -> str:
        """Write out equation for plotting."""
        return (
            f'{self.a:.10f}*t**3 + {self.b:.10f}*t**2 + '
            f'{self.c:.10f}*t + {self.d:.10f}'
        )

    @classmethod # ML
    def get_cheapest_curve(
            cls, s_total_emission: pd.Series, year_x: int,
            reb: float, slope_tm1: float, previous_slope: float | None = None) -> EmissionCurve:
        """
        Factory method to get the best emission curve.

        This function searches the optimal emission curve by
        varying the parameter c.

        - The integral of the emission curve from year_x
           to target year is approx. equal (as close as possible)
           to the remaining emission budget (REB).
        - The exceedence emissions are as small as possible

        Args:
            cls (Python class):
            s_total_emission (pd.Series): Total GHG emission (CO2-eq)
                timeseries.
            year_x (int): Current year in which the emissions for the next
                five years should be calculated.
            reb (float): Remaining emission budget until the acidity target
                 is reached in Pg C.
            slope_tm1 (float): Slope of emission curve in year_x-1
                previous_slope (float): Slope of the curve estimated by the
                previous stocktake
        Returns:
            emission_curve (EmissionCurve): Optimal emission curve.
        """
        # Slope of the curve estimated by the previous stocktake for
        # the year of this stocktake is chosen as a starting guess
        c0 = previous_slope

        if previous_slope is None:
            # If no previous stocktake exist that used the AERA, use
            # the slope at year X
            c0 = s_total_emission.diff().loc[year_x]

        # d is the present day emissions
        d = s_total_emission.loc[year_x]

        # The future emission curve needs to be at least 5 years long.
        # When the acidity target is almost reached and emissions
        # are small, the minimum length is increased to avoid overly
        # strong reactions to decadal or interannual variability that
        # may look like an anthropogenic trend in acidification.
        # The maximum length is 150 years but can be extended for high
        # acidity targets to avoid an increase in present-day
        # emissions to get faster to these acidity levels. Thus, the
        # polynom length is extended by one year for each 5 Pg C that
        # exceed 500 Pg C.

        reb_tmp = cp.deepcopy(reb)
        d_tmp = cp.deepcopy(d)

        if np.abs(d_tmp) > 10:
            d_tmp = 10 * np.sign(d_tmp)

        reb_tmp = reb_tmp - 500

        if reb_tmp < 0:
            reb_tmp = 0

        # Variable length of polynom dependend on present day emissions
        target_year_rel_max = int(150 + reb_tmp/5.0)
        target_year_rel_min = int(5 + (((100-(d_tmp**2))/100.) * 45))

        # The max and min of the rate of change are chosen so large that
        # they will very likely never appear and hence cover the entire range
        c_change_min = -2.5
        c_change_max = 2.5
        # We choose a c_change_delta value that is a good compromise between
        # fine sampling of the slope change parameter space and computational
        # efficiency
        c_change_delta = 0.001

        slope_changes = np.arange(
            c_change_min, c_change_max + c_change_delta, c_change_delta
        )
        c_vals = c0 + slope_changes
        ty_vals = np.arange(target_year_rel_min, target_year_rel_max + 1)

        ec = cls(ty_vals, c_vals, d, c0, reb, slope_tm1)

        cost = ec.cost
        best_idx = np.unravel_index(np.argmin(cost), cost.shape)
        i_c, i_ty = best_idx

        return cls(
            target_year_rel=int(ty_vals[i_ty]),
            c=float(c_vals[i_c]),
            d=float(d),
            slope_t0=float(c0),
            reb=float(reb),
            slope_tm1=float(slope_tm1),
        )

