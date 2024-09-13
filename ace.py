from pathlib import Path
import ssl
from ai import cdas
import pandas as pd
import numpy as np


def read_ace_mag(start, end,cache_dir='./cdas-data'):
    dataset = 'AC_H3_MFI'
    vlist = ['BGSEc', 'BRTN']
    try:
        data = cdas.get_data('sp_phys',dataset,start, end ,variables=vlist)
    except:
        print(f"CDAS Error loading {dataset} data for this date range")
        return pd.DataFrame()

    map_mag = {'date_time':data['EPOCH'],
        'Bx':data['BX_GSE'],
        'By':data['BY_GSE'],
        'Bz':data['BZ_GSE']}
    df = pd.DataFrame(data=map_mag)
    df['B'] = np.sqrt(data['BX_GSE']**2 + data['BY_GSE']**2 + data['BZ_GSE']**2)
    df['ddoy'] = df.date_time.dt.day_of_year \
                + (df.date_time.dt.hour + (df.date_time.dt.minute)/60 \
                + (df.date_time.dt.second)/3600)/24
    df.set_index('date_time', inplace=True)

    # Rudimentary quality filter
    df.where(df > -1.0e+29, np.nan, inplace=True)
    df.where(df < 1.0e+29, np.nan, inplace=True)

    # Store metadata
    df.attrs['data_source'] = f'Wind MFI H0 1min dataset [{dataset}]'
    df.attrs['timezone'] = 'UTC'    
    df.attrs['coord_system'] = 'GSE'
    df.Bx.attrs['unit'] = 'nT'
    df.By.attrs['unit'] = 'nT'
    df.Bz.attrs['unit'] = 'nT'
    df.B.attrs['unit'] = 'nT'

    units = {}
    for c in df:
        if 'unit' in df[c].attrs:
            units[c] = df[c].attrs['unit']

    if len(units) > 0:
        df.attrs['units'] = units
    return df

def read_ace_ion(start, end, cache_dir='./cdas-data'):
    """
    Load plasma ion data variables for Wind mission (SWE instrument)
    using the AI.CDAS package

    start/end: datetime objects for start/end time of interest
    cache_dir: (optional) directory for storing downloaded data.  
                Defaults to './cdas-data/'

    -----
    Returns a pandas DataFrame
    """

    ssl._create_default_https_context = ssl._create_unverified_context 

    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True)
    cdas.set_cache(True, cache_dir)


    dataset = 'AC_H0_SWE'

    vlist = ['Np','Tpr','V_GSE','alpha_ratio']
    try:
        data = cdas.get_data('sp_phys', dataset, start, end, variables=vlist)
    except:
        print(f"CDAS Error loading {dataset} data for this date range")
        return pd.DataFrame()

    map_ion = {'date_time':data['EPOCH'],
      
        'Vx':data['VX_(GSE)'],
        'Vy':data['VY_(GSE)'],
        'Vz':data['VZ_(GSE)'],
        'Np':data['H_DENSITY'],
        'Tp':data['H_TEMP_RADIAL'],
        'alpha_ratio':data['NA/NP'],
        }
    df = pd.DataFrame(data=map_ion)
    df['Vsw'] = np.sqrt(data['VX_(GSE)']**2 + data['VY_(GSE)']**2 + data['VZ_(GSE)']**2)
    df['ddoy'] = df.date_time.dt.day_of_year \
                + (df.date_time.dt.hour + (df.date_time.dt.minute)/60 \
                + (df.date_time.dt.second)/3600)/24
    df.set_index('date_time', inplace=True)

    # Rudimentary quality filter
    df.where(df > -1.0e+31, np.nan, inplace=True)
    df.where(df < 1.0e+31, np.nan, inplace=True)

    # Store this last to avoid problems with df.where()
    # df['delta_time'] = pd.to_timedelta(data['DEL_TIME'], unit='milli')

    # Set metadata
    df.attrs['data_source'] = f'Wind SWE Key Parameters dataset [{dataset}]'
    df.attrs['timezone'] = 'UTC'
    df.attrs['coord_system'] = 'GSE'
    df.Np.attrs['unit'] = 'cm^{-3}'
    df.alpha_ratio.attrs['unit'] = 'cm^{-3}'
    df.Tp.attrs['unit'] = 'K'
    df.Vx.attrs['unit'] = 'km/s'
    df.Vy.attrs['unit'] = 'km/s'
    df.Vz.attrs['unit'] = 'km/s'


    units = {}
    for c in df:
        if 'unit' in df[c].attrs:
            units[c] = df[c].attrs['unit']

    if len(units) > 0:
        df.attrs['units'] = units
    return df

def read_ace_epam(start, end, cache_dir='./cdas-data'):
    """
    Load plasma ion data variables for Wind mission (SWE instrument)
    using the AI.CDAS package

    start/end: datetime objects for start/end time of interest
    cache_dir: (optional) directory for storing downloaded data.  
                Defaults to './cdas-data/'

    -----
    Returns a pandas DataFrame
    """

    ssl._create_default_https_context = ssl._create_unverified_context 

    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True)
    cdas.set_cache(True, cache_dir)


    dataset = 'AC_H1_EPM'

    vlist = ['P1p','P2p','P3p','P4p','P5p','P6p','P7p','P8p']
    try:
        data = cdas.get_data('sp_phys', dataset, start, end, variables=vlist)
    except:
        print(f"CDAS Error loading {dataset} data for this date range")
        return pd.DataFrame()

    map_epm = {'date_time':data['EPOCH'],
      
        'P1p':data['P1P_.047-.066MEV_IONS'],
        'P2p':data['P2P_.066-.114MEV_IONS'],
        'P3p':data['P3P_.114-.190MEV_IONS'],
        'P4p':data['P4P_.190-.310MEV_IONS'],
        'P5p':data['P5P_.310-.580MEV_IONS'],
        'P6p':data['P6P_.580-1.05MEV_IONS'],
        'P7p':data['P7P_1.05-1.89MEV_IONS'],
        'P8p':data['P8P_1.89-4.75MEV_IONS'],
        }
         
    df = pd.DataFrame(data=map_epm)
    
    df['ddoy'] = df.date_time.dt.day_of_year \
                + (df.date_time.dt.hour + (df.date_time.dt.minute)/60 \
                + (df.date_time.dt.second)/3600)/24
    df.set_index('date_time', inplace=True)

    # Rudimentary quality filter
    df.where(df > -1.0e+31, np.nan, inplace=True)
    df.where(df < 1.0e+31, np.nan, inplace=True)

    # Store this last to avoid problems with df.where()
    # df['delta_time'] = pd.to_timedelta(data['DEL_TIME'], unit='milli')

    # Set metadata
    df.attrs['data_source'] = f'Wind EPM Key Parameters dataset [{dataset}]'
    df.attrs['timezone'] = 'UTC'
    df.attrs['coord_system'] = 'GSE'
    df.P1p.attrs['unit'] = '1/(cm**2-s-sr-MeV)'
    df.P2p.attrs['unit'] = '1/(cm**2-s-sr-MeV)'
    df.P3p.attrs['unit'] = '1/(cm**2-s-sr-MeV)'
    df.P4p.attrs['unit'] = '1/(cm**2-s-sr-MeV)'
    df.P5p.attrs['unit'] = '1/(cm**2-s-sr-MeV)'
    df.P6p.attrs['unit'] = '1/(cm**2-s-sr-MeV)'
    df.P7p.attrs['unit'] = '1/(cm**2-s-sr-MeV)'
    df.P8p.attrs['unit'] = '1/(cm**2-s-sr-MeV)'
   


    units = {}
    for c in df:
        if 'unit' in df[c].attrs:
            units[c] = df[c].attrs['unit']

    if len(units) > 0:
        df.attrs['units'] = units
    return df

def read_ace_swics(start, end, cache_dir='./cdas-data'):
    """
    Load plasma ion data variables for Wind mission (SWE instrument)
    using the AI.CDAS package

    start/end: datetime objects for start/end time of interest
    cache_dir: (optional) directory for storing downloaded data.  
                Defaults to './cdas-data/'

    -----
    Returns a pandas DataFrame
    """

    ssl._create_default_https_context = ssl._create_unverified_context 

    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True)
    cdas.set_cache(True, cache_dir)


    dataset = 'AC_H3_SW2'

    vlist = ['vHe2','vthHe2','C6to5','O7to6','O8to6','avqFe','FetoO']
    try:
        data = cdas.get_data('sp_phys', dataset, start, end, variables=vlist)
    except:
        print(f"CDAS Error loading {dataset} data for this date range")
        return pd.DataFrame()

    map_swics = {'date_time':data['EPOCH'],
      
        'vHe2':data['HE++_SPEED'],
        'vthHe2':data['HE++_THERMAL_SPEED'],
        'C6to5':data['C+6/C+5_RATIO'],
        'O7to6':data['O+7/O+6_RATIO'],
        'O8to6':data['O+8/O+6_RATIO'],
        'avqFe':data['<Q>_FE'],
        'FetoO':data['FE/O_RATIO'],
   
        }
         
    df = pd.DataFrame(data=map_swics)
    
    df['ddoy'] = df.date_time.dt.day_of_year \
                + (df.date_time.dt.hour + (df.date_time.dt.minute)/60 \
                + (df.date_time.dt.second)/3600)/24
    df.set_index('date_time', inplace=True)

    # Rudimentary quality filter
    df.where(df > -1.0e+31, np.nan, inplace=True)
    df.where(df < 1.0e+31, np.nan, inplace=True)

    # Store this last to avoid problems with df.where()
    # df['delta_time'] = pd.to_timedelta(data['DEL_TIME'], unit='milli')

    # Set metadata
    df.attrs['data_source'] = f'Wind EPM Key Parameters dataset [{dataset}]'
    df.attrs['timezone'] = 'UTC'
    df.attrs['coord_system'] = 'GSE'
    df.vHe2.attrs['unit'] = 'km/s'
    df.vthHe2.attrs['unit'] = 'km/s'
    df.C6to5.attrs['unit'] = 'ratio'
    df.O7to6.attrs['unit'] = 'ratio'
    df.O8to6.attrs['unit'] = 'ratio'
    df.avqFe.attrs['unit'] = 'ionic_charge'
    df.FetoO.attrs['unit'] = 'ratio'

   


    units = {}
    for c in df:
        if 'unit' in df[c].attrs:
            units[c] = df[c].attrs['unit']

    if len(units) > 0:
        df.attrs['units'] = units
    return df