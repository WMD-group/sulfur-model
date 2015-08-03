from numpy import genfromtxt, log, isnan
from scipy.interpolate import interp1d
from scipy.constants import R
        
def free_energy(T, filename, P=1e5):
    """
    Get the chemical potential of a species from tabulated data.

    Arguments:
        T: Scalar or numpy array of temperature in K

        filename: Path to .dat file containing thermochemical data. Data
        format is fixed; space-separated based NIST/JANAF thermochemical
        tables, with two header rows.

        P: Scalar or numpy array of pressure in Pa. T and P arrays are
        combined in the standard way for Numpy operations; if they have
        the same shape they will be acted on pairwise. If one is a row
        and the other is a column, the output will be a matrix of all
        combinations.

    Outputs:
        free_energy: Chemical potential in J mol-1
    """

    Tref = 298.15
    Pref = 1e5
    
    data = genfromtxt(filename, skip_header=2)

    # Strip out rows with NaN in free energy column (Numpy refresher:
    # boolean in square brackets selects rows, ~ negates answer)
    
    data = data[~isnan(data[:,3])]
    
    T_table = data[:,0]

    try:
        Tref_index = T_table.tolist().index(Tref)
        H_Tref = data[Tref_index,4]

    except ValueError: # Interpolate H_Tr if not in table
        H_func = interp1d(T_table, data[:,4], kind='cubic')
        H_Tref= H_func(Tref)

    G_HTr_T = data[:,3]
    G = G_HTr_T * -T_table + H_Tref

    G = G + R * T_table * log(P/Pref)
    
    G_func = interp1d(T_table, G, kind='cubic', bounds_error=False, fill_value=0.0)

    return G_func(T)

