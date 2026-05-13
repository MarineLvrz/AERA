"""Core functions of the AERA algorithm.

Contains the following functions:
- calculate_anth_temperature : Calculates the anthropogenic temperature.
- calculate_absolute_target_temperature: Calculates the absolute target
    temperature [K].
- calculate_remaining_emission_budget : Calculates the remaining
    emission budget (REB). The REB is the amount of CO2-fe emission that
    are still allowed to be emitted in the future.
- extrapolated_runmean_anth_temp: Calculates the extrapolated running
    mean of the observed/simulated temperature. Calls the functions
    extrapolated_runmean and runmean.
- get_adaptive_emissions : MAIN FUNCTION. Calculates "optimal"
    near-future CO2 emissions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from aera import constants
from aera import utils
from aera import io
from aera import emission_curve


def runmean(array, winlen):
    """Calculates a running mean of any timeseries using a window
    length (winlen).

    Args:
        array: timeseries over which the running mean is calculated
        winlen: window length of running mean
    """
    return np.convolve(array, np.ones((winlen)) / winlen, mode='same')


def extrapolated_runmean(array, winlen):
    """Extrapolates a running mean at the beginning and end of the 
    time series. At the beginning, the window size is simply reduced. 
    At the end, which is critical for the t_anth estimation at the 
    stocktake, the running mean is linearly extrapolated. This is 
    done by calculating the slope over the last 31 (generally winlen) 
    years and adding a linear spline with this slope to the last 
    valid running mean value 17 (generally int(winlen / 2) +1) years 
    before stocktake to obtain the estimates for the last 15 
    (generally int(winlen / 2)) years before stocktake.

    Args:
        array: timeseries over which the running mean is calculated
        winlen: window length of running mean
    """
    array_runmean = runmean(array, winlen)

    # Reduce window size down to int(winlen / 2) + 1
    # for the first int(winlen / 2) elements.
    array_runmean[:int(winlen / 2)] = np.array([
        np.mean(array[:int(winlen / 2) + i + 1])
        for i in range(int(winlen / 2))
    ])

    # Replace the last int(winlen / 2) elements by a linear extrapolation.
    array_runmean[-int(winlen / 2):] =\
        array_runmean[-int(winlen / 2) - 1]\
        + np.polyfit(np.arange(winlen), array[-winlen:], deg=1)[0]\
        * np.arange(1, int(winlen / 2) + 1)

    return array_runmean

def extrapolated_runmean_anth_temp(year_x, model_start_year, s_temp, winlen):
    """Calculate an extrapolated running mean of the simulated temperature.

    Args:
        year_x: Year of the sticktake
        model_start_year (int): Year in which the historical
            simulation (pre-cursor for the adaptive scenario
            simulation) was started.
        s_temp: Simulated termperature timeseries
        winlen: window length of running mean for temperature fit

    Returns:
        Extrapolated running mean of the simulated temperature 

    """
    temp = np.array(s_temp.loc[model_start_year:year_x])

    return extrapolated_runmean(temp, winlen)



def calculate_absolute_target_temperature(
        temperature_target_rel, model_start_year, s_temp, winlen,
        temperature_target_type, costum_anth_temperature_func=None):
    """Calculate the absolute target temperature.

    Args:
        temperature_target_rel (float): Relative temperature target
            (e.g. 1.5K).
        model_start_year (int): Year in which the historical
            simulation (pre-cursor for the adaptive scenario
            simulation) was started.
        s_temp: Simulated temperature timeseries
        winlen: window length of running mean for temperature fit
        temperature_target_type (int): Switch for different types of temperature
            targets.
            - 1: Temperature target estimated by observed remaining
                 warming from 2020 onwards
            - 2: Simulated warming anomaly with the reference period
                 1850-1900
        costum_anth_temperature_func (function): See documentation in
            `get_adaptive_emissions`.

    Returns:
        temperature_target_abs (float): Absolute target temperature [K].

    """
    model_start_year = max(1850, model_start_year)

    if temperature_target_type == 1:
        # Calculate anthropogenic warming in 2020
        if costum_anth_temperature_func is None:
            temperature_anth_2020 = extrapolated_runmean_anth_temp(
            2020, model_start_year, s_temp, winlen)[-1]
        else:
            temperature_anth_2020 = costum_anth_temperature_func(
                2020, model_start_year, s_temp,
            )[-1]

        # Absolute target temperature is based on observed
        # anthropogenic warming in 2020
        temperature_target_abs = (temperature_anth_2020 +
            (temperature_target_rel - constants.OBS_ANTH_WARMING_2020))
        print(f'Anthropogenic temperature 2020: {temperature_anth_2020}')
    elif temperature_target_type == 2:
        temperature_target_abs = np.nanmean(
            s_temp.loc[model_start_year:1900].values) + temperature_target_rel
    else:
        print('Invald temperature target type chosen (options are: {1, 2}).')
    return temperature_target_abs


def _calculate_previous_emission_slope(year_x, meta_file):
    """Calculate the slope at Year X by using the emission
    curve from the previous stocktake.

    To make the emission curve as smooth as possible the
    AERA algorithm has to use the previously calculated
    emission curve parameters (i.e. a, b, and c).

    Args:
        year_x (int): Current year in which the emissions for the next
            five years should be calculated.
        meta_file (str or pathlib.Path): File for temporary data which
            should be transfered from one run of the AERA algorithm
            to the next.

    """
    meta_file = Path(meta_file)
    if not meta_file.exists():
        return
    ds = xr.open_dataset(meta_file)
    try:
        a = ds.ec_a.sel(year_stocktake=year_x-5)
        b = ds.ec_b.sel(year_stocktake=year_x-5)
        c = ds.ec_c.sel(year_stocktake=year_x-5)
    except KeyError:
        return
    t = 5
    return 3 * a * t**2 + 2 * b * t + c


def calculate_remaining_emission_budget(
        temperature_anth, total_ghg_emission, temperature_target_abs, year_x,
        model_start_year, temperature_abs_ts):
    """Calculate remaining emission budget.

    Args:
        temperature_anth (array-like): Time series of anthropogenic temperature
            (without any natural variablity).
        total_ghg_emission (array-like): Time series of total emissions.
        temperature_target_abs (float): Absolute target temperature.
        year_x (int): Current year in which the emissions for the next
            five years should be calculated.
        model_start_year (int): Year in which the historical
            simulation (pre-cursor for the adaptive scenario simulation)
            was started.
        temperature_abs_ts (array-like): Time series of measured/simulated
            temperature (including natural variablity).

    Returns:
        reb (float): Remaining emission budget until the target
            temperature is reached in Pg C.

    """
    # Substract the reference period temperature (1850-1900)
    dtemperature_ref_yearx = (
        temperature_anth.loc[year_x]) - temperature_abs_ts.loc[model_start_year:1900].mean()
    print('Relative anthropogenic warming in Year X: ', dtemperature_ref_yearx)
    print('Cumulative past GHGs emissions: ',
          total_ghg_emission.loc[model_start_year:year_x-1].sum())

    # Calculate TCRE (Cum. Emissions divided by anthropogenic warming)
    slope = total_ghg_emission.loc[model_start_year:year_x -
                               1].sum() / dtemperature_ref_yearx
    # Multiply TCRE with remaing allowable warming
    reb = (temperature_target_abs - temperature_anth.loc[year_x]) * slope
    print('REB: ', reb)
    return reb


def get_adaptive_emissions(
        temperature_target_rel, temperature_target_type, year_x,
        model_start_year, df, meta_file, costum_anth_temperature_func=None):
    """Calculate "optimal" near-future CO2 emissions.

    A full time series with CO2 emissions is returned, but only the next
    five years are used in an AERA simulation. However, some
    models calculate monthly emission data for the second half of the year
    using already the annual emissions from the following year. Such models
    therefore need at least one year more than these five years.

    Args:
        temperature_target_rel (float): Temperature target (e.g. 1.5K).
        temperature_target_type (int): Switch for different types of temperature
            targets.
            - 1: Temperature target estimated by additing remaining
                 warming until the target is reached based on 
                 observations in 2020 to simulated anthropogenic 
                 warming in 2020. Thus, all models have the same 
                 remaining warming after 2020 independend of their 
                 warming over the historical period.
            - 2: Temperature target estimated based on simulated 
                 warming anomaly with the reference period 1850-1900
        year_x (int): Current year in which the emissions for the next
            five years should be calculated.
        model_start_year (int): Year in which the historical
            simulation (pre-cursor for the adaptive scenario simulation)
            was started.
        df (pd.DataFrame): Pandas dataframe with years (int) as index
            and the following columns (see utils.get_base_df which
            provides a skeleton of this dataframe):
            - temp:  Global annual mean temperature time series for
              the period (in Kelvin).
            - ff_emission: Global annual mean fossil fuel CO2
              emission time series (in Pg C / yr).
            - lu_emission: Global annual mean land use change
              CO2 emission time series (in Pg C / yr).
            - non_co2_emission: Global annual mean non-CO2 emission (in
              CO2-eq Pg C / yr)
        meta_file (str or pathlib.Path): File for temporary data which
            should be transfered from one run of the AERA algorithm
            to the next.
        costum_anth_temperature_func (function): ONLY FOR ADVANCED USE CASES!
            Costum, user-defined function that calculates the
            anthropogenic temperature. This function is given the
            following args (same as given to `get_adaptive_emissions`):
                - year_x
                - model_start_year
                - s_temp (timeseries of simulated temperature)
            It must return a list/numpy-array with the anthropogenic
            temperature for the years from `model_start_year` until
            the given `year_x` (in the calculation of the
            anthropogenic temperature of the year 2020, "2020" is
            given instead of `year_x`).

    Returns:
        s_ff_emission (pd.Series): Annual globally integrated fossil fuel
            CO2 emission time series (in Pg C / yr).

    """
    # Some CMIP5 models start later than 1850. The earliest start year
    # is 1850 because no observed temperature record exists before.
    # Simulated parameters before 1850 are not used
    model_start_year = max(
        1850, model_start_year)
    utils.validate_df(df, year_x, model_start_year)
    
    # For a temperature target, we need to consider ff + luc + nonCO2 emissions, 
    # because all of them contribute to the anthropogenic warming (for the TCRE relationship). 
    ghg_emission_cols = ['ff_emission', 'lu_emission', 'non_co2_emission']
    s_total_ghg_emission = df[ghg_emission_cols].sum(skipna=True, axis=1)

    # ML 12.05.26 we want both targets to return the 3 emission time series
    co2_emission_cols = ['ff_emission', 'lu_emission'] # ML
    s_total_co2_emission = df[co2_emission_cols].sum(skipna=True, axis=1)

    # Define window length for extrapolated running mean
    winlen = 31

    # Calculate the temperature target
    temperature_target_abs = calculate_absolute_target_temperature(
        temperature_target_rel, model_start_year, df['temperature'], winlen,
        temperature_target_type=temperature_target_type)

    # Extract the temperature time series until the time of the stocktake
    s_temperature_anth = df['temperature'].loc[model_start_year:year_x].copy()
    
    # Extract anthropogenic warming
    if costum_anth_temperature_func is None:
        s_temperature_anth.loc[:] = extrapolated_runmean_anth_temp(
            year_x,model_start_year, df['temperature'], winlen)
    else:
        s_temperature_anth.loc[:] = costum_anth_temperature_func(
            year_x, model_start_year, df['temperature'],
        )

    # Extract again the temperature time series until the time of the
    # stocktake, simulated/measured temperature and only anthropogenic
    # temperature will be needed later
    s_temperature_abs = df['temperature'].loc[model_start_year:year_x].copy()

    # Calculate remaining emissions budget
    reb = calculate_remaining_emission_budget(
        s_temperature_anth, s_total_ghg_emission, temperature_target_abs, year_x,
        model_start_year, s_temperature_abs)

    # Read in slope at Year_X as estimated at previous stocktake
    previous_slope = _calculate_previous_emission_slope(year_x, meta_file)
    if previous_slope is not None:
        previous_slope = float(previous_slope)

    # Calculate the slope of the emissions curve at year X-1
    slope_tm1 = s_total_ghg_emission.loc[year_x]-s_total_ghg_emission.loc[year_x-1]
    slope_tm1 = float(slope_tm1)

    # Calculate the future emission curves
    ec = emission_curve.EmissionCurve.get_cheapest_curve(
        s_total_ghg_emission, year_x, reb, slope_tm1, previous_slope)

    # ML, 24.04.26 we remove 'temperature_target_rel,' that is useless and not supported
    # anymore with the new implementation of get_cheapest_curve 
    
    # Add 5 (arbitrary number) years more to extand the emission curve
    # further in case of extrapolation problems if models need
    # emissions from the year ahead to calculate monthly emissions in
    # the 2nd half of the year
    additional_years = 5
    t = np.arange(1, ec.target_year_rel + additional_years + 2)
    year1 = int(year_x + 1)
    year2 = int(year1 + ec.target_year_rel) + additional_years
    s_total_ghg_emission.loc[year1:year2] = ec.get_values(t=t)
    print('CO2-fe emissions [Pg C] (fossil fuel CO2 + landuse + non-CO2) '
          'over next years:')
    print(s_total_ghg_emission.loc[year1:year2-5])

    # Calculate co2 emissions as the difference between
    # estimated total emissions and prescribed nonCO2
    # emissions
    s_total_co2_emission = (
        s_total_ghg_emission - df['non_co2_emission'])
    s_total_co2_emission.name = 'total_co2_emission'

    # Calculate Fossil fuel emissions as the difference between
    # estimated total emissions and prescribed land-use and nonCO2
    # emissions
    s_ff_emission = (
        s_total_ghg_emission - df['lu_emission'] - df['non_co2_emission'])
    s_ff_emission.name = 'ff_emission'

    # Store data to metafile for debug and post-analysis
    io.store_metadata(
        meta_file, temperature_target_rel, temperature_target_abs, year_x, s_temperature_anth, s_total_ghg_emission, s_total_co2_emission, s_ff_emission, ec)

    return s_ff_emission.loc[year1:year2]
