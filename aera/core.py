"""
Core functions of the AERA algorithm.

Contains the following functions:

- runmean : Calculate a running mean of any timeseries using a window
    length (winlen).
- extrapolated_runmean : Extrapolate a running mean at the beginning and end of the 
    time series.
- extrapolated_runmean_anth_arag : Calculate the extrapolated running
    mean of the observed/simulated aragonite saturation state. Calls the functions
    extrapolated_runmean and runmean.
- calculate_relative_target_aragonite : Calculate the relative target
    for aragonite saturation state [-].
- _calculate_previous_emission_slope : Calculate the slope at year X by using the emission
    curve from the previous stocktake.
- calculate_remaining_emission_budget : Calculate the remaining
    emission budget (REB). The REB is the amount of CO2-fe emission that
    are still allowed to be emitted in the future.
- get_adaptive_emissions : MAIN FUNCTION. Calculate "optimal"
    near-future CO2 emissions.
"""

from pathlib import Path

import numpy as np
import xarray as xr

from aera import utils
from aera import io
from aera import emission_curve


def runmean(array, winlen):
    """
    Calculates a running mean of any timeseries using a window
    length (winlen).

    Args:
        array: timeseries over which the running mean is calculated
        winlen: window length of running mean.

    Returns:
        Running mean of any timeseries given a window length.
    """    
    return np.convolve(array, np.ones((winlen)) / winlen, mode='same')


def extrapolated_runmean(array, winlen):
    """
    Extrapolates a running mean at the beginning and end of the 
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

    Returns:
        Extrapolated running mean to handle the beginning and the end 
        of the timeseries of interest
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


def extrapolated_runmean_anth_arag(year_x, model_start_year, s_arag, winlen): # ML
    """
    Calculate an extrapolated running mean of the simulated aragonite
    saturation state.

    Args:
        year_x (int): Year of the stocktake
        model_start_year (int): Year in which the historical
            simulation (pre-cursor for the adaptive scenario
            simulation) was started.
        s_arag: Simulated aragonite saturation state timeseries
        winlen: window length of running mean for saturation state fit

    Returns:
        Extrapolated running mean of the simulated aragonite saturation state. 
    """
    arag = np.array(s_arag.loc[model_start_year:year_x])
    
    return extrapolated_runmean(arag, winlen)


# Only needed for aragonite, because for temperature it is given as input
def calculate_relative_target_aragonite(  # ML
    arag_target_abs, s_arag, model_start_year):
    """
    Calculate the relative target for aragonite saturation state.
    Relative target aragonite, computed based on the method used for 
    computing the absolute temperature target for target type 2, see 
    calculate_absolute_target_temperature().
    We expect a negative relative aragonite target, because aragonite 
    saturation state is decreasing.
    
    Args:
        arag_target_abs (float): Absolute aragonite saturation state target
            (e.g. 2.75 or 1.0 []).
        s_arag: Simulated aragonite saturation state timeseries.
        model_start_year (int): Year in which the historical simulation 
        (pre-cursor for the adaptive scenario simulation) was started.
        
    Returns:
        arag_target_rel (float): relative aragonite saturation state target.
    """
    
    arag_target_rel = arag_target_abs - np.nanmean(s_arag.loc[model_start_year:1900].values)
    
    return arag_target_rel


def _calculate_previous_emission_slope(year_x, meta_file):
    """
    Calculate the slope at year X by using the emission
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

    Returns: 
        Emission slope at year X.
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


def calculate_remaining_emission_budget( # ML
        omega_arag_anth, total_co2_emission, arag_target_abs, year_x,
        model_start_year, arag_abs_ts):
    
    """
    Calculate remaining emission budget.

    Args:
        omega_arag_anth (array-like): Time series of anthropogenic aragonite 
        saturation state (without any natural variablity).
        total_co2_emission (array-like): Time series of total emissions.
        arag_target_abs (float): Absolute target aragonite saturation state.
        year_x (int): Current year in which the emissions for the next
            five years should be calculated.
        model_start_year (int): Year in which the historical simulation 
            (pre-cursor for the adaptive scenario simulation) was started.
        arag_abs_ts (array-like): Time series of measured/simulated
            aragonite saturation state (including natural variablity).

    Returns:
        reb (float): Remaining emission budget until the acidification target 
            is reached in Pg C.
    """
    
    # Substract the reference period aragonite (1850-1900)
    darag_ref_yearx = (
        omega_arag_anth.loc[year_x]) - arag_abs_ts.loc[model_start_year:1900].mean()
    print('Relative anthropogenic acidification in Year X: ', darag_ref_yearx)
    print('Cumulative past emissions: ',
          total_co2_emission.loc[model_start_year:year_x-1].sum())
    # Calculate TCRE (Cum. Emissions divided by anthropogenic warming)
    slope = total_co2_emission.loc[model_start_year:year_x -
                               1].sum() / darag_ref_yearx
    # Multiply TCRE with remaing allowable warming
    reb = (arag_target_abs - omega_arag_anth.loc[year_x]) * slope
    print('REB: ', reb)
    
    return reb



def get_adaptive_emissions( # ML
        arag_target_abs, year_x,
        model_start_year, df, meta_file, costum_anth_arag_func=None):
    """
    Calculate "optimal" near-future CO2 emissions.

    A full time series with CO2 emissions is returned, but only the next
    five years are used in an AERA simulation. However, some
    models calculate monthly emission data for the second half of the year
    using already the annual emissions from the following year. Such models
    therefore need at least one year more than these five years.

    Args:
        arag_target_abs (float): TeAragonite saturation state target (e.g. 2.75 []).
        year_x (int): Current year in which the emissions for the next
            five years should be calculated.
        model_start_year (int): Year in which the historical
            simulation (pre-cursor for the adaptive scenario simulation) was started.
        df (pd.DataFrame): Pandas dataframe with years (int) as index
            and the following columns (see utils.get_base_df which
            provides a skeleton of this dataframe):
            - OmegaA:  Global or regional annual mean aragonite saturation state
              time series for the period ().
            - ff_emission: Global annual mean fossil fuel CO2
              emission time series (in Pg C / yr).
            - lu_emission: Global annual mean land use change
              CO2 emission time series (in Pg C / yr).
            - non_co2_emission: Global annual mean non-CO2 emission (in
              CO2-eq Pg C / yr)
        meta_file (str or pathlib.Path): File for temporary data which
            should be transfered from one run of the AERA algorithm
            to the next.

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

    # ML 12.05.26 we want both targets to return the 3 emission time series
    ghg_emission_cols = ['ff_emission', 'lu_emission', 'non_co2_emission']
    s_total_ghg_emission = df[ghg_emission_cols].sum(skipna=True, axis=1)

    # We discard non-co2, because in the case of aragonite saturation
    # state, we want only ff + LUC in the computation of the TCRE.
    # Non-CO2 do not impact ocean acidification.
    co2_emission_cols = ['ff_emission', 'lu_emission'] # ML
    s_total_co2_emission = df[co2_emission_cols].sum(skipna=True, axis=1)

    # Define window length for extrapolated running mean
    winlen = 31

    # Calculate relative aragonite target
    arag_target_rel = calculate_relative_target_aragonite(arag_target_abs, df['omega_arag'], model_start_year)

    # Initialise the variable for anthropogenic aragonite time series until the time of the stocktake
    s_omega_arag_anth = df['omega_arag'].loc[model_start_year:year_x].copy()

    # Extract anthropogenic aragonite time series
    # Second option does not exist for aragonite
    if costum_anth_arag_func is None:
        s_omega_arag_anth.loc[:] = extrapolated_runmean_anth_arag(
            year_x,model_start_year, df['omega_arag'], winlen)
    else:
        s_omega_arag_anth.loc[:] = costum_anth_arag_func(
            year_x, model_start_year, df['omega_arag'],
        )

    # Extract again the aragonite saturation state time series until 
    # the time of the stocktake, simulated/measured anthropogenic aragonite 
    # saturation state will be needed later
    s_arag_abs = df['omega_arag'].loc[model_start_year:year_x].copy()

    # Calculate remaining emissions budget
    reb = calculate_remaining_emission_budget(
        s_omega_arag_anth, s_total_co2_emission, arag_target_abs, year_x,
        model_start_year, s_arag_abs)

    # Read in slope at Year_X as estimated at previous stocktake
    previous_slope = _calculate_previous_emission_slope(year_x, meta_file)
    if previous_slope is not None:
        previous_slope = float(previous_slope)

    # Calculate the slope of the emissions curve at year X-1
    slope_tm1 = s_total_co2_emission.loc[year_x]-s_total_co2_emission.loc[year_x-1]
    slope_tm1 = float(slope_tm1)

    # Calculate the future emission curves
    # get_cheapest_curve actually does not need the argument 'arag_target_rel'
    ec = emission_curve.EmissionCurve.get_cheapest_curve( 
        s_total_co2_emission, year_x, reb, slope_tm1, previous_slope)

    # Add 5 (arbitrary number) years to extend the emission curve further in
    # case of extrapolation problems if models need emissions from the year 
    # ahead to calculate monthly emissions in the 2nd half of the year
    additional_years = 5
    t = np.arange(1, ec.target_year_rel + additional_years + 2)
    year1 = int(year_x + 1)                                          # Year following the current stocktake
    year2 = int(year1 + ec.target_year_rel) + additional_years       # 5 year after the target has been reached
    s_total_co2_emission.loc[year1:year2] = ec.get_values(t=t)
    # We discard non-CO2 for the reasons stated above
    print('CO2 emissions [Pg C] (fossil-fuel CO2 + land use) ' 
          'over next years:')
    print(s_total_co2_emission.loc[year1:year2-5])

    # Calculate ghg emissions as the sum of
    # estimated total co2 emissions and prescribed nonCO2
    # emissions
    s_total_ghg_emission = (
        s_total_co2_emission + df['non_co2_emission']) # ML 13.05.26
    s_total_ghg_emission.name = 'total_ghg_emission'

    # Calculate fossil fuel emissions as the difference between
    # estimated emissions of interest and prescribed land use emissions
    # We only need to subtract LUC emissions in the case of OA, non-CO2 
    # emissions are not considered into 's_total_co2_emission'
    s_ff_emission = (s_total_co2_emission - df['lu_emission']) # ML
    s_ff_emission.name = 'ff_emission'
    
    # Store data to metafile for debug and post-analysis
    io.store_metadata(
        meta_file, arag_target_rel, arag_target_abs, year_x, s_omega_arag_anth, s_total_ghg_emission, s_total_co2_emission, s_ff_emission, ec)

    return s_ff_emission.loc[year1:year2]
