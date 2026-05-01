"""
Utility functions.

Contains the following important functions:
- validate_df : Used to check if an end-user given pandas.DataFrame
    contains all the neccessary data and is correctly formatted.
- _load_dat_df: Allows to read and assign prescribed input data.
- get_base_df : A helper function that facilitates the creation of
    the pandas.DataFrame instance which is needed by
    `aera.core.get_adaptive_emissions`.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import aera


MIN_YEAR = 1751
MAX_YEAR = 2300


def validate_df(df, year_x, model_start_year): # ML
    """
    Validate whether all neccessary data is contained in `df`.
    Args: 
        df (pandas.DataFrame): Template dataframe which can be
            filled with data and then used for the call to
            `aera.get_adaptive_emissions`.
        year_x (int): Year of the stocktake
        model_start_year (int): Year in which the historical
            simulation (pre-cursor for the adaptive scenario
            simulation) was started.
    """

    model_start_year = max(1850, model_start_year)
    if model_start_year >= 1900:
        raise ValueError(
            'The historical run is too short for this algorithm.')

    col_years_dict = {
        'OmegaA': np.arange(model_start_year, year_x+1),
        'ff_emission': np.arange(model_start_year, year_x+1),
        'lu_emission': np.arange(model_start_year, MAX_YEAR+1),
        'non_co2_emission': np.arange(model_start_year, MAX_YEAR+1),
        }
    for col_name, years in col_years_dict.items():
        col = df[col_name].dropna()
        if not np.all(np.isin(years, col.index.values)):
            missing_years_idx = np.argwhere(~np.isin(years, col.index.values))
            missing_years = years[missing_years_idx].flatten()
            raise ValueError(
                f'Neccessary data in column {col_name} is missing.\n'
                'Data for the following years is missing but must be'
                f' available:\n {missing_years}.')


def _load_dat_df(f, column_names, delim_whitespace=False):
    """
    Allows to read and assign prescribed input data.
    Args:
        f (?): file containing the time series
        column_names (string): name of the column
        delim_whitespace (bool): False by default
    Returns:
        df (pandas.DataFrame): Template dataframe which can be
            filled with data.
    """
    df = pd.read_table(
        f, header=None, index_col=0, sep=r"\s+") # ML
    df.columns = column_names
    df.index.name = 'year'
    df.index = [int(x) for x in df.index.values]
    df = df.reindex(
        np.arange(df.index.min(), df.index.max())).interpolate()
    return df


def get_base_df(
        ):
    """
    Return dataframe which is used by get_adaptive_emissions.

    The dataframe contains non-CO2 emission and land use emission 
    data provided by Terhaar et al. (2022). This data is contained 
    in the `aera` module and can be found within the official 
    repository.

    Note: The returned pandas.DataFrame cannot be passed to
    `aera.core.get_adaptive_emissions` directly!
    The following steps are still neccessary before calling
    `get_adaptive_emissions`:
    - Fill "OmegaA" column with aragonite saturation state
     data from the model.
    - Fill "ff_emission" column with CO2 emission data from
      year 2026 on.
    - If model-specific data is available for "lu_emission",
      and "non_co2_emission" columns, please overwrite the 
      prefilled 'standard' data

    Args:
    Returns:
        df (pandas.DataFrame): Template dataframe which can be
            filled with data and then used for the call to
            `aera.get_adaptive_emissions`.
    """
    data_dir = Path(aera.__file__).parent / 'data' # Location where this file was installed as part of AERA package. Take the parent file and go into data.
    non_co2_emission_file = data_dir / 'nonco2_emis_ssp126_v3.dat'
    lu_emission_file = data_dir / 'lu_emis_ssp126_bern3d_adj_GCB2020_v1.dat'
    ff_emission_file = data_dir / 'co2_ff_GCP_plus_NDC_v1.dat' # ML
    #ff_emission_file = data_dir / 'co2_ff_GCP_updated_v1.dat' # ML, most up to date ff emissions

    print(f'Use the following non-CO2 emission file: {non_co2_emission_file}')
    print(f'Use the following land use emission file: {lu_emission_file}')
    print(f'Use the following historical fossil fuel CO2 emission '
          f'file: {ff_emission_file}')

    df_list = []
    
    # Read and assign prescribed input data
    df_non_co2_emission = _load_dat_df(
        non_co2_emission_file, ['non_co2_emission'], delim_whitespace=True)
    df_list.append(df_non_co2_emission)

    df_ff_emission = _load_dat_df(
        ff_emission_file, ['ff_emission'], delim_whitespace=True)
    df_list.append(df_ff_emission)

    df_lu_emission = _load_dat_df(
        lu_emission_file, ['lu_emission'], delim_whitespace=True)
    df_list.append(df_lu_emission)

    df = pd.concat(df_list, axis=1)

    # Take into account timesteps that are not assigned to any prescribed input data
    df['OmegaA'] = np.nan                      # ML
    df.loc[2026:, 'ff_emission'] = np.nan      # ML
    df.loc[:1849, 'lu_emission'] = np.nan      # ML
    df.loc[:1849, 'non_co2_emission'] = np.nan # ML
    df.index.name = 'year'
    # Reorder columns
    df = df[['non_co2_emission', 'lu_emission', 
             'ff_emission', 'OmegaA']] # ML

    return df.loc[MIN_YEAR:2499]