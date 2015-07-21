#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os # get correct path for datafiles when called from another directory
from itertools import izip
from collections import namedtuple
from scipy.special import erf, erfc

script_directory = os.path.dirname(__file__)
# Append a trailing slash to make coherent directory name - this would select the
# root directory in the case of no prefix, so we need to check
if script_directory:
    script_directory  = script_directory + '/'
data_directory = script_directory +  '../data/'

data_sets = {'LDA':'sulfur_lda.json', 'PBEsol':'sulfur_pbesol.json', 'PBE0':'sulfur_pbe0.json', 'PBE0_scaled':'sulfur_pbe0_96.json', 'B3LYP':'sulfur_b3lyp.json'}


# Add module to path
import sys
sys.path.append(script_directory+'../')

from sulfur.core import get_potentials, unpack_data, reference_energy, solve_composition

import scipy.constants as constants
eV2Jmol = constants.physical_constants['electron volt-joule relationship'][0] * constants.N_A
k = constants.physical_constants['Boltzmann constant in eV/K'][0]

### Parameters for PBE0_scaled fits ###
S8_poly = [ -3.80993101e-13,   1.80778884e-09,  -4.01150741e-06,
        -2.45657416e-03,   7.62018747e-01]
S2_poly = [ -8.65418878e-14,   4.00096901e-10,  -8.56622340e-07,
        -1.84833136e-03,   1.20725661e+00]
gaussian_height_poly = [   49.92188405,   -96.05901548,  1275.83961616]
T_tr_poly = [   1.82766941,   -8.29481895,   72.7179064 ,  507.70392149]
gaussian_b = 10
gaussian_c = 80

def mu_S8_fit(T,P):
    return np.polyval(S8_poly,T) + k*T*np.log(P/1E5)
def mu_S2_fit(T,P):
    return np.polyval(S2_poly,T) + k*T*np.log(P/1E5)
def T_tr(P):
    return np.polyval(T_tr_poly, np.log10(P))
def mu_fit(T,P):
    t_tr = T_tr(P)
    mu_S8_contrib = mu_S8_fit(T,P)*erfc((T-t_tr)/gaussian_b) * (1./(2.*8.)) * eV2Jmol
    mu_S2_contrib = mu_S2_fit(T,P)*(erf((T-t_tr)/gaussian_b)+1) * (1./(2.*2.)) * eV2Jmol
    gaussian_contrib = -(np.polyval(gaussian_height_poly, np.log10(P)))*np.exp(-(T-(t_tr-gaussian_b))**2/(2.*gaussian_c**2))
    return mu_S8_contrib + mu_S2_contrib + gaussian_contrib

def plot_T_composition(T, n, labels, title, filename=False):
    axis=plt.gca()
    fig=plt.gcf()
    axis.set_color_cycle(["#222222","#a6cee3","#1f78b4","#b2df8a","#33a02c","#fb9a99","#e31a1c","#fdbf6f","#ff7f00","#cab2d6","#6a3d9a","#bbbb55","#b04040","#b15928"])
    plt.plot(T,n, linewidth=5)
    axis.set_position([0.1,0.15,0.6,0.75])
    plt.legend(labels, loc='center left',bbox_to_anchor=(1,0.5))
    plt.xlabel('Temperature / K')
    plt.ylabel('Mole fraction $x_i$')
    plt.title(title)
    plt.xlim(0,1500) and plt.ylim(0,1)
    fig.set_size_inches(8,6)

    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()

def plot_n_pressures(functional, T=False, P_list=False, P_ref=1E5, compact=False, filename=False):

    if not T:
        T = np.linspace(10,1500,200)
    if not P_list:
        P_list = [1E2, 1E5, 1E7]

    db_file = data_directory + data_sets[functional]
    labels, thermo, a = unpack_data(db_file, ref_energy=reference_energy(db_file, units='eV'))

    if compact:
        fig_dimensions = (8 / 2.54, 14 / 2.54)
    else:
        fig_dimensions = (17.2 / 2.54, 12 / 2.54)

    plt.figure(figsize = fig_dimensions)

    axis_list = []
    subplot_index = 1
    xdim, ydim = len(P_list), 1
    
    for P in P_list:
        n = []
        mu = []
        lines = []
        for t in T:
            n_t, mu_t = solve_composition(a, get_potentials(thermo, T=t, P_ref=P_ref), P=P/P_ref, T=t)
            n.append(n_t * a) # Multiply by a; convert between species mole fraction and atom mole fraction
            mu.append(mu_t)

        axis_list.append(plt.subplot(ydim,xdim,subplot_index))
        axes=plt.gca()
        fig=plt.gcf()
        axes.set_color_cycle(["#222222","#a6cee3","#1f78b4","#b2df8a","#33a02c","#fb9a99","#e31a1c","#fdbf6f","#ff7f00","#cab2d6","#6a3d9a","#bbbb55","#b04040","#b15928"])
        for n_species in [list(x) for x in zip(*n)]: # Filthy python to convert list by T to list by species
            line, = axes.plot(T,n_species, linewidth=3)
            lines.append(line)
        plt.xlabel('Temperature / K')
        if subplot_index == 1:
            plt.ylabel('Mole fraction $x_i$')
        plt.title('P = {0} bar'.format(P*1E-5))

        subplot_index += 1

    # Scale down axes to make way for legend
    for ax in axis_list:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height*0.30, box.width, box.height * 0.70])
        ax.set_xticks( np.arange(0,max(T)+1,500) )
    legend_ax = fig.add_axes([0.1, 0.01, 0.8, 0.2], frame_on=False, xticks=[], yticks=[]) # Invisible axes holds space for legend
    legend_ax.legend(lines,labels, ncol=3, loc='center', fontsize='small')
    #'fig.tight_layout()


    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()

def compute_data(functionals=['PBE0_scaled'], T=[298.15], P=[1E5]):
    """
    Solve S_x equilibrium over specified sets of DFT data, temperatures and pressures

    Arguments
        functionals: iterable of strings identifying DFT dataset; must be a key in 'data_sets' dict
        T: iterable of temperatures in K
        P: iterable of pressures in Pa
    Returns
        data: dict containing Calc_n_mu namedtuples, with keys corresponding to 'functionals'.
              Each namedtuple contains the nested lists n[P][T], mu[P][T] and list labels.
              n: atom frac of S of each species, corresponding to labels
              mu: free energy of mixture on atom basis in J mol-1
              labels: identity labels of each species in mixture
    """
    Calc_n_mu = namedtuple('Calc_n_mu','n mu labels')
    
    P_ref = 1E5
    eqm_data = {}
    for functional in functionals:
        db_file = data_directory + data_sets[functional]
        labels, thermo, a = unpack_data(db_file, ref_energy=reference_energy(db_file, units='eV'))
        n = []
        mu = []
        for p in P:
            n_p, mu_p = [], []
            for t in T:
                n_p_T, mu_p_T = solve_composition(a, get_potentials(thermo, T=t, P_ref=P_ref), P=p/P_ref, T=t)
                n_p.append([n_i*a_i for n_i,a_i in izip(n_p_T,a)]) # Convert to atom frac
                mu_p.append(mu_p_T)
            n.append(n_p)
            mu.append(mu_p)
        eqm_data.update({functional:Calc_n_mu(n, mu, labels)})                        
    return eqm_data

def plot_mu_functionals(data, T, P, functionals=False, filename=False, compact=False):
    """
    Plot free energy against T for a range of datasets.

    Arguments:
        data: dict containing Calc_n_mu namedtuples, with keys corresponding to 'functionals'.
              Each namedtuple contains the nested lists n[P][T], mu[P][T] and list labels.
              n: atom frac of S of each species, corresponding to labels
              mu: free energy of mixture on atom basis in J mol-1
              labels: identity labels of each species in mixture
        T: Iterable of temperatures in K, corresponding to T ranges in data
        P: Iterable of P values in Pa corresponding to data. Used for labelling: all pressures will be plotted
        functionals: iterable containing keys of data to use. If False, all functionals in 'data_sets' will be plotted.
        filename: Filename for plot output. If False (default), print to screen instead.
        compact: Boolean, setting width to 8cm for publication if True
    """

    if functionals == False:
        functionals = data_sets.keys()

    if compact:
        fig_dimensions = (8 / 2.54, 8 / 2.54)
    else:
        fig_dimensions = (17.2 / 2.54, 12 / 2.54)
        
    for i_p, p in enumerate(P):

        plt.figure(figsize = fig_dimensions)

        for functional in functionals:
            mu_kJmol = np.array(data[functional].mu[i_p]) * 1E-3
            plt.plot(T,mu_kJmol, label=functional)
        ax = plt.gca()
        ax.set_xticks( np.arange(0,max(T)+1,500) )
        plt.legend(loc='best', fontsize='small')
        plt.title('P = {0} bar'.format(p * 1E-5))
        plt.xlabel('Temperature / K')
        plt.ylabel('$\mu_S$ / kJ mol$^{-1}$')
        plt.tight_layout()
        if filename:
            plt.savefig('{0}_{1}.pdf'.format(filename,p*1E-5))
        else:
            plt.show()
        plt.close()

def tabulate_data(data,T,P,path=''):
    """
    Write tables of composition and free energy
    
    Arguments:
            data: dict containing Calc_n_mu namedtuples, with keys corresponding to 'functionals'.
              Each namedtuple contains the nested lists n[P][T], mu[P][T] and list labels.
              n: atom frac of S of each species, corresponding to labels
              mu: free energy of mixture on atom basis in J mol-1
              labels: identity labels of each species in mixture
            T: Iterable containing temperature values in K corresponding to data
            P: Iterable containing pressure values in Pa corresponding to data
            path: directory for csv files to be written in
    """

    import string
    if path:
        if path[-1] != '/':
            path = path + '/'

    for functional in data.keys():
        with open(path + 'mu_{0}.csv'.format(functional.lower()), 'w') as f:
            linelist = ['# T/K,' + string.join(['mu ({0} Pa) / J mol-1'.format(p) for p in P],',') + '\n']
            for t_index, t in enumerate(T):
                linelist.append( '{0},'.format(t) + string.join(['{0:1.4f}'.format(mu_p[t_index]) for mu_p in data[functional].mu],',') + '\n')
            f.writelines(linelist)

    for functional in data.keys():
        with open(path + 'n_{0}.csv'.format(functional.lower()), 'w') as f:
            for p_index, p in enumerate(P):
                linelist = ['# P = {0} Pa\n'.format(p)]
                linelist.append('# T/K, ' + string.join(['x({0})'.format(x) for x in data[functional].labels],',') + '\n')
                for t_index, t in enumerate(T):
                    linelist.append('{0},'.format(t) + string.join(['{0:1.4f}'.format(n) for n in data[functional].n[p_index][t_index]],',') + '\n')
                f.writelines(linelist)

def plot_surface(functional='PBE0_scaled', T_range=(300,1200), P_range=(1,7), resolution=1000, tolerance = 1e4, parameterised=True, filename=False):
    """Generate a surface plot showing recommended S models. Can be slow!

    Arguments:
        functional: id of dataset used. PBE0_scaled is strongly recommended as it has good agreement with experimental data.
        T_range: Tuple containing T range in K
        P_range: Tuple containing (log10(Pmin), log10(Pmax))
        resolution: Number of points on x and y axis. Note that a full free energy minimisation is carried out at each point, so print-ready resolutions will take some time to compute.
        tolerance: Error threshold for transition region in Jmol-1
        parameterised: Boolean. If True, use parameterised fit (polynomials, erf and gaussian). If False, solve equilibrium at all points (slow!)
        filename: String containing output file. If False, print to screen.
    

    """
    import ase.thermochemistry
    import ase.db

    T = np.linspace(min(T_range), max(T_range), resolution)
    P = 10**np.linspace(min(P_range),max(P_range),resolution)[:, np.newaxis]

    if parameterised:
        mu_mixture = mu_fit(T,P)
        mu_S2 = mu_S2_fit(T,P) * eV2Jmol / 2.
        mu_S8 = mu_S8_fit(T,P) * eV2Jmol / 8.
    else:
        data = compute_data(T=T, P=P, functionals=[functional])
        mu_mixture = np.array(data[functional].mu)
        db_file = data_directory+data_sets[functional]
        labels, thermo, a = unpack_data(db_file,ref_energy=reference_energy(db_file, units='eV'))
        S2_thermo = thermo[labels.index('S2')]
        S8_thermo = thermo[labels.index('S8')]

        v_get_gibbs_energy=np.vectorize(ase.thermochemistry.IdealGasThermo.get_gibbs_energy)
        mu_S2 = v_get_gibbs_energy(S2_thermo,T, P, verbose=False) * eV2Jmol / 2.
        mu_S8 = v_get_gibbs_energy(S8_thermo,T, P, verbose=False) * eV2Jmol / 8.

    plt.figure()
    CS = plt.contour(T,np.log10(P).flatten(),np.minimum(abs(mu_S2 - mu_mixture),abs(mu_S8 - mu_mixture)), [1000])
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Error')

    if filename:
        plt.savefig(filename)
    else:
        plt.show()

    # plt.figure()
    # CS = plt.contour(T,np.log10(P).flatten(),abs(mu_S8 - mu_mixture))
    # plt.clabel(CS, inline=1, fontsize=10)
    # plt.title('Error of S8')
    # plt.show()

def check_fit():
    """Sanity check for polynomial fitting"""
    import ase.thermochemistry
    import ase.db

    T = np.linspace(100,1000,10)
    P = np.array([1E3])
    data = compute_data(T=T, P=P, functionals=['PBE0_scaled'])
    mu_mixture = np.array(data['PBE0_scaled'].mu)
    db_file = data_directory+data_sets['PBE0_scaled']
    labels, thermo, a = unpack_data(db_file,ref_energy=reference_energy(db_file, units='eV'))
    S2_thermo = thermo[labels.index('S2')]
    S8_thermo = thermo[labels.index('S8')]

    v_get_gibbs_energy=np.vectorize(ase.thermochemistry.IdealGasThermo.get_gibbs_energy)
    mu_S2 = v_get_gibbs_energy(S2_thermo,T, P, verbose=False) * eV2Jmol / 2.
    mu_S8 = v_get_gibbs_energy(S8_thermo,T, P, verbose=False) * eV2Jmol / 8.

    plt.plot(T, mu_mixture.transpose(), 'bx', ms=20, label="Mixture (solver)")
    plt.plot(T, mu_fit(T,P),'r+', ms=20, label="Mixture (fit)")
    plt.plot(T, mu_S2, 'go', label=r"S$_2$ (model)")
    plt.plot(T, mu_S2_fit(T,P) * eV2Jmol/2.,'k^', label=r"S$_2$ (fit)")
    plt.plot(T, mu_S8_fit(T,P) * eV2Jmol/8., 'k*', label=r"S$_8$ (fit)")
    plt.legend()
    plt.show()
                                
def main():
    # check_fit()
    # T = np.linspace(50,1500,50)
    # P = 1E5

    # data = compute_data(T=T, functionals=['PBE0_scaled'])
    # plot_T_composition(T, data['PBE0_scaled'].n[0], data['PBE0_scaled'].labels, 'PBE0, P = 1E5' , filename=False)


    # T = np.arange(50,1500,5)
    # P = [10**x for x in range(1,7)]
    # data = compute_data(T=T, P=P, functionals = data_sets.keys())
    # tabulate_data(data,T,P, path='data')

    # plot_mu_functionals(data, T, P, filename=False, compact=False)

    # plot_surface(resolution=200, parameterised=False, filename='tmp.pdf')
                
if __name__ == '__main__':
    main()

