#! /usr/bin/env python
import argparse

import os.path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import gridspec
from scipy.special import erf, erfc

import ase.thermochemistry
import ase.db

import shelve
from itertools import cycle

from matplotlib import rc, rcParams
#rc('font',**{'family':'serif', 'weight':'normal'})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('font',**{'family':'sans-serif', 'sans-serif':['Helvetica Neue']})
rc('text', usetex=True)
# rcParams['text.latex.preamble'] = [r'\boldmath']
rcParams['text.latex.preamble'] = [r'\usepackage{helvet} \usepackage{sfmath}']
rc('legend',**{'fontsize':10})

import os # get correct path for datafiles when called from another directory
import sys # PATH manipulation to ensure sulfur module is available
from itertools import izip
from collections import namedtuple
Calc_n_mu = namedtuple('Calc_n_mu','n mu H labels')

script_directory = os.path.dirname(__file__)
# Append a trailing slash to make coherent directory name - this would select the
# root directory in the case of no prefix, so we need to check
if script_directory:
    script_directory  = script_directory + '/'
module_directory = os.path.abspath(script_directory + '..')
data_directory = os.path.abspath(script_directory +  '../data')
sys.path.insert(0,module_directory)

from sulfur import get_potentials, unpack_data, reference_energy, solve_composition, mix_enthalpies

ordered_species = ['S2','S3_ring','S3_bent','S4_buckled','S4_eclipsed','S5_ring','S6_stack_S3','S6_branched','S6_buckled','S6_chain_63','S7_ring','S7_branched','S8']

data_sets = {'LDA':'sulfur_lda.json', 'PBEsol':'sulfur_pbesol.json', 'PBE0':'sulfur_pbe0.json', 'PBE0_scaled':'sulfur_pbe0_96.json', 'B3LYP':'sulfur_b3lyp.json'}

species_colors = {'S2':'#222222','S3_ring':'#a6cee3','S3_bent':'#1f78b4','S4_buckled':'#b2df8a','S4_eclipsed':'#33a02c','S5_ring':'#fb9a99','S6_stack_S3':'#e31a1c','S6_branched':'#fdbf6f','S6_buckled':'#ff7f00','S6_chain_63':'#cab2d6','S7_ring':'#6a3d9a','S7_branched':'#bbbb55','S4_C2h':'#b04040','S8':'#b15928'}

species_markers = {'S2':'8','S3_ring':'>','S3_bent':'<','S4_buckled':'^','S4_eclipsed':'o',
                   'S5_ring':'d','S6_stack_S3':'D','S6_branched':'H','S6_buckled':'h','S6_chain_63':'*',
                   'S7_ring':'p','S7_branched':'s','S8':'x'}

# LaTeX formatted names for species. Keys correspond to database keys
species_names = {'S2':r'S$_2$ (D$_{\infty \mathrm{h}}$)','S3_ring':r'S$_3$ (D$_{3\mathrm{h}}$)','S3_bent':r'S$_3$ (C$_{2\mathrm{v}}$)','S4_buckled':r'S$_4$ (D$_{2\mathrm{d}}$)','S4_eclipsed':r'S$_4$ (C$_{2\mathrm{v}}$)','S4_C2h':r'S$_4$ (C$_{2\mathrm{h}}$)','S5_ring':r'S$_5$ (C$_\mathrm{s}$)','S6_stack_S3':r'S$_6$ (D$_{3 \mathrm{h}}$)','S6_branched':r'S$_6$ (C$_1$, branched)','S6_buckled':r'S$_6$ (C$_{2\mathrm{v}}$)','S6_chain_63':r'S$_6$ (C$_1$, chain)','S7_ring':r'S$_7$ (C$_{\mathrm{s}}$)','S7_branched':r'S$_7$ (C$_\mathrm{s}$, branched)','S8':r'S$_8$ (D$_{4\mathrm{d}}$)'}

# Alternative / LaTeX escaped names for DFT functionals. May also be useful for changing capitalisation, LDA vs LSDA etc.
functional_names = {'PBE0_scaled':r'PBE0 (scaled)'} 

# Add module to path
import sys
sys.path.append(script_directory+'../')

import scipy.constants as constants
eV2Jmol = constants.physical_constants['electron volt-joule relationship'][0] * constants.N_A
k = constants.physical_constants['Boltzmann constant in eV/K'][0]

### Parameters for PBE0_scaled fits ###
S8_poly = [ -3.810e-13,   1.808e-09,  -4.012e-06,
        -2.457e-03,   7.620e-01]
S2_poly = [ -8.654e-14,   4.001e-10,  -8.566e-07,
        -1.848e-03,   1.207e00]
gaussian_height_poly = [ 6.663e01,   -2.041e02,  1.414e03]
T_tr_poly = [   1.828,   -8.295,   7.272e01 ,  5.077e02]
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
    plt.xlim(400,1500) and plt.ylim(0,1)
    fig.set_size_inches(8,6)

    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close(fig)

def plot_frequencies(functionals=False, figsize=False, filename=False):
    """
    Plot calculated vibrational mode frequencies of S8 compared to spectroscopic data

    Arguments:
        functionals: iterable of strings identifying DFT dataset; each string must be a key in 'data_sets' dict
        figsize: 2-tuple of figure dimensions in inches
        filename: path to output file. If False (default), print to screen instead.

    """
    if not functionals:
        functionals = data_sets.keys()

    if not figsize:
        figsize = (8.4/2.54, 8.4/2.54)

    fig = plt.figure(figsize=figsize)

    NIST_S8_f = [475,218,471,471,191,191,475,475,152,152,56,56,411,243,437,437,248,248]

    index=0
    ticklabels=[]
    for functional in functionals:
        index += 1
        db_file = data_directory + '/' + data_sets[functional]
        db = ase.db.connect(db_file)
        freqs = db.get_dict('S8').data.frequencies
        plt.plot([index]*len(freqs),freqs, '_', markersize=20, label=functional)
        if functional in functional_names:
            ticklabels.append(functional_names[functional])
        else:
            ticklabels.append(functional)

    index +=1
    plt.plot([index]*len(NIST_S8_f), NIST_S8_f, 'k_', markersize=20, label='Expt')
    ticklabels.append('Expt')

    plt.xlim(0.5,len(ticklabels)+0.5)
    axis = plt.gca()
    axis.xaxis.set_ticks(range(1,len(ticklabels)+1))
    axis.xaxis.set_ticklabels(ticklabels, rotation=35, ha='right')

    # fontsize=10
    # for tick in axis.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(fontsize)
    # for tick in axis.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(fontsize)

    plt.ylabel('Frequency / cm$^{-1}$')
    plt.ylim(0,500)

    plt.subplots_adjust(left=0.2,bottom=0.25)

    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close(fig)

def plot_composition(T, P, data, functionals=False, filename=False):
    """
    Plot composition vs T over a range of pressures and DFT functionals in a neat tiled array

    Arguments:        
        T: iterable of temperatures in K
        P: iterable of pressures in Pa
        data: dict containing Calc_n_mu namedtuples, with keys corresponding to 'functionals'.
              Each namedtuple contains the nested lists n[P][T], mu[P][T] and list labels.
              n: atom frac of S of each species, corresponding to labels
              mu: free energy of mixture on atom basis in J mol-1
              labels: identity labels of each species in mixture
        functionals: iterable of strings identifying DFT dataset; each string must be a key in 'data_sets' dict
        filename: path to output file. If False (default), print to screen instead.

    """

    if functionals == False:
        functionals = data.keys()

    fig = plt.figure(figsize =  (17.2 / 2.54, 17 / 2.54))
    gs = gridspec.GridSpec(len(functionals), len(P), bottom=0.25)

    tick_length = 4
    tick_width = 0.5
    
    for row, functional in enumerate(functionals):
        color_cycle = [species_colors[species] for species in data[functional].labels]
        for col, p in enumerate(P):
            ax = plt.subplot(gs[row,col])
            ax.set_color_cycle(color_cycle)
            for i, species in enumerate(data[functional].labels):
                ax.plot(T, [data[functional].n[col][t_index][i] for t_index in range(len(T))])
            ml = MultipleLocator(400)
            ax.xaxis.set_major_locator(ml)
            ax.axes.set_ylim([0,1])
            ax.axes.set_xlim([400,1500])
            
            if row == 0:
                ax.set_title("$10^{" + "{0:d}".format(int(np.log10(p))) + "}$ Pa", fontweight='normal')
                ax.set_xticklabels('',visible=False)
            elif row != len(functionals) -1:
                ax.set_xticklabels('',visible=False)
            else:
                ax.axes.set_xlabel('Temperature / K')

            if col == 0:
                if functional in functional_names:
                    functional_label = functional_names[functional]
                else:
                    functional_label = functional
                ax.axes.set_ylabel(functional_label)
                ax.set_yticks([0,1])
                ax.set_yticklabels(['0','1'])
                ml = MultipleLocator(0.2)
                ax.yaxis.set_minor_locator(ml)
                ax.tick_params('both',length=tick_length,width=tick_width, which='both')
            else:
                ax.set_yticklabels('',visible=False)
                ax.tick_params('both',length=tick_length,width=tick_width, which='both')

    plt.legend([plt.Line2D((0,1),(0,0), color=species_colors[species]) for species in ordered_species],
               [species_names[species] for species in ordered_species], ncol=4, loc='center', bbox_to_anchor=(0.5,0.1), bbox_transform=fig.transFigure, fontsize=11)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close(fig)

def plot_n_pressures(functional, T=False, P_list=False, P_ref=1E5, compact=False, filename=False):

    if not T:
        T = np.linspace(10,1500,200)
    if not P_list:
        P_list = [1E2, 1E5, 1E7]

    db_file = data_directory + '/' + data_sets[functional]
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
        axes.set_title("$10^{" + "{0:d}".format(int(np.log10(P)) + "}$ Pa", fontweight='normal'))
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
    plt.close(fig)

def compute_data(functionals=['PBE0_scaled'], T=[298.15], P=[1E5], ref_energy='expt', enthalpy=False):
    """
    Solve S_x equilibrium over specified sets of DFT data, temperatures and pressures

    Arguments
        functionals: iterable of strings identifying DFT dataset; must be a key in 'data_sets' dict
        T: iterable of temperatures in K
        P: iterable of pressures in Pa
        ref_energy: Select reference energy. If 'expt' (default), use experimental enthalpy 
                    of alpha-S as reference. If 'S8', use 1/8 * ground state energy of S8 
                    in chosen data set as reference energy. If a floating point number, the 
                    value of ref_energy is used with units of eV/atom.
        enthalpy: Boolean flag. If True, also compute enthalpy. 
                  (This costs extra time and is not usually required.)
    Returns
        data: dict containing Calc_n_mu namedtuples, with keys corresponding to 'functionals'.
              Each namedtuple contains the nested lists n[P][T], mu[P][T], H[P][T] and list labels.
              n: atom frac of S of each species, corresponding to labels
              mu: free energy of mixture on atom basis in J mol-1
              H: enthalpy of mixture on atom basis in J mol-1. False if not computed.
              labels: identity labels of each species in mixture
    """
    
    P_ref = 1E5
    eqm_data = {}
    for functional in functionals:
        db_file = data_directory + '/' + data_sets[functional]
        if type(ref_energy) != str and np.isscalar(ref_energy):  # (Strings are scalar!)
            labels, thermo, a = unpack_data(db_file, ref_energy=ref_energy)
        elif ref_energy == 'expt':
            labels, thermo, a = unpack_data(db_file, ref_energy=reference_energy(db_file, units='eV', ref='expt'))
        elif ref_energy == 'S8':
            labels, thermo, a = unpack_data(db_file, ref_energy=reference_energy(db_file, units='eV', ref='S8'))
        else:
            raise Exception("ref_energy key {0} not recognised")
        n = []
        mu = []

        if enthalpy:
            n, mu, H = [], [], []
            for p in P:
                n_p, mu_p, H_p = [], [], []
                for t in T:                
                    n_p_T, mu_p_T = solve_composition(
                        a, get_potentials(thermo, T=t, P_ref=P_ref), P=p/P_ref, T=t)
                    H_p_T = mix_enthalpies(n_p_T, thermo, t)
                    n_p.append(n_p_T)
                    mu_p.append(mu_p_T)
                    H_p.append(H_p_T)
                n.append(n_p)
                mu.append(mu_p)
                H.append(H_p)
                                    
            eqm_data.update({functional:Calc_n_mu(n, mu, H, labels)})
        else:
            for p in P:
                n_p_mu_p_double = [
                    ([n_i*a_i for n_i,a_i in izip(n_p_T,a)], mu_p_T) for
                    (n_p_T, mu_p_T) in (
                        solve_composition(a, get_potentials(thermo, T=t, P_ref=P_ref), P=p/P_ref, T=t)
                        for t in T)
                    ]
                n_p, mu_p = [x[0] for x in n_p_mu_p_double], [x[1] for x in n_p_mu_p_double]
                n.append(n_p)
                mu.append(mu_p)

            eqm_data.update({functional:Calc_n_mu(n, mu, False, labels)})
    return eqm_data

def plot_mu_functionals(data, T, P, functionals=False,  T_range=False, mu_range=False, filename=False, compact=False):
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
        T_range: 2-tuple in K of temperature range to display. If False, use range of T.
        mu_range: 2-tuple in kJ mol-1 of mu range to display. If False, use matplotlib default
        filename: Filename for plot output. If False (default), print to screen instead.
        compact: Boolean, setting width to 8cm for publication if True
    """

    ########## Literature data for S2, S8 ##########

    s2_data = np.genfromtxt(data_directory + '/S2.dat', skip_header=2)
    s8_data = np.genfromtxt(data_directory + '/S8.dat', skip_header=2)

    # Fourth column contains -(G-H(Tr))/T in J/molK
    T_s2 = s2_data[:,0]
    DG_s2 = s2_data[:,3] * T_s2 * -1E-3
    mu_s2 = (DG_s2 + 128.600)/2.

    T_s8 = s8_data[:,0]
    DG_s8 = s8_data[:,3] * T_s8 * -1E-3
    mu_s8 = (DG_s8 + 101.416)/8.

    R = 8.314 * 1E-3 # Gas constant in kJ mol-1 K-1


    ######## Plotting ########

    if not T_range:
        T_range = (min(T), max(T))

    if functionals == False:
        functionals = data_sets.keys()

    if compact:
        fig_dimensions = (8 / 2.54, 8 / 2.54)
    else:
        fig_dimensions = (17.2 / 2.54, 8 / 2.54)

    bottom = 0.4
    fig = plt.figure(figsize = fig_dimensions)
    gs = gridspec.GridSpec(1,len(P), bottom=bottom)
        
    for i_p, p in enumerate(P):

        if i_p == 0:
            ax = plt.subplot(gs[i_p])
            left_ax = ax
            ax.axes.set_ylabel('$\mu_S$ / kJ mol$^{-1}$')
        else:
            ax = plt.subplot(gs[i_p], sharey=left_ax)

        for functional in functionals:
            if functional in functional_names:
                functional_name = functional_names[functional]
            else:
                functional_name = functional
            mu_kJmol = np.array(data[functional].mu[i_p]) * 1E-3
            ax.plot(T,mu_kJmol, label=functional_name)

        # Plot literature data with pressure correction
        ax.plot(T_s2, mu_s2 + R*T_s2*np.log(p/1E5)/2, label=r'S$_2$ (lit.)', linestyle=':')
        ax.plot(T_s8, mu_s8 + R*T_s8*np.log(p/1E5)/8, label=r'S$_8$ (lit.)', linestyle=':')

        if i_p > 0:
            plt.setp(ax.get_yticklabels(), visible=False) # I don't understand why ax.set_yticklabels doesn't work here, but it wipes out the first column too.

        ml = MultipleLocator(400)
        ax.xaxis.set_major_locator(ml)

        ax.axes.set_xlim(T_range)
        ax.axes.set_title('P = {0} bar'.format(p * 1E-5))
        ax.axes.set_xlabel('Temperature / K')

    if mu_range:
        ax.axes.set_ylim(mu_range)
    
    plt.legend(ncol=4, loc='center', bbox_to_anchor=(0.5,bottom/3.), bbox_transform=fig.transFigure, fontsize=11)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close(fig)

def plot_mu_contributions( T, P, data, functionals, T_range=(400,1500), filename=False, figsize=(17.2 / 2.54, 17 / 2.54), bottom=0.4, T_units='K', T_increment=400, mu_range=False):
    """
    Plot free energy of mixture, showing contributions of components.

    Arguments:
        data: dict containing Calc_n_mu namedtuples, with keys corresponding to 'functionals'.
              Each namedtuple contains the nested lists n[P][T], mu[P][T] and list labels.
              n: atom frac of S of each species, corresponding to labels
              mu: free energy of mixture on atom basis in J mol-1
              labels: identity labels of each species in mixture
        T: Iterable of temperatures in K, corresponding to T ranges in data
        P: Iterable of P values in Pa corresponding to data. Used for labelling: all pressures will be plotted
        functionals: iterable containing keys of data to use. If False, all functionals
              in 'data_sets' will be plotted.
        T_range: 2-tuple containing temperature range of plot [default value (200,1500)]
        filename: Filename for plot output. If False (default), print to screen instead.
        figsize: 2-tuple containing figure dimensions in inches
        bottom: fraction of figure vertically reserved for legend.
        T_units: Temperature unit. If set to 'C', T_range and displayed axis are in degrees C. Input data must still be in K.
        T_increment: Float; Spacing between temperature markers on x-axis
        mu_range: 2-tuple. If set, force displayed y-axis range in kJ/mol

    """
    
    fig = plt.figure(figsize=figsize)

    # Allow extra space for functional labels on y axes
    if len(functionals) > 1:
        left = 0.2
    else:
        left=0.15
    gs = gridspec.GridSpec(len(functionals), len(P), left=left, bottom=bottom)

    tick_length = 4
    tick_width = 0.5

    if T_units == 'C':
        T_offset=-273.15
        T_label = r'Temperature / $^\circ$C'
    else:
        T_offset=0
        T_label = r'Temperature / K'
    
    for row, functional in enumerate(functionals):
        db_file = data_directory + '/' + data_sets[functional]
        labels, thermo, a = unpack_data(db_file, ref_energy=reference_energy(db_file, units='eV'))
        
        color_cycle = [species_colors[species] for species in data[functional].labels]
        for col, p in enumerate(P):
            if col == 0:
                ax = plt.subplot(gs[row,col])
                left_ax = ax
            else:
                ax = plt.subplot(gs[row,col], sharey=left_ax)
               
            ax.set_color_cycle(color_cycle)
            ax.plot([t + T_offset for t in T], [get_potentials(thermo,T=t,P_ref=p) * 1E-3 / a for t in T])
            
            ax.plot([t + T_offset for t in T], [mu * 1E-3 for mu in data[functional].mu[col]], 'm:', linewidth=4)
            ax.tick_params('both',length=tick_length,width=tick_width, which='both')

            ml = MultipleLocator(T_increment)
            ax.xaxis.set_major_locator(ml)
            ax.axes.set_xlim(T_range)

            if mu_range:
                ax.axes.set_ylim(mu_range)

            if row == 0:
                ax.set_title("$10^{" + "{0:d}".format(int(np.log10(p))) + "}$ Pa", fontweight='normal')
            if row != len(functionals) -1:
                ax.set_xticklabels('',visible=False)
            if row == len(functionals) -1:
                ax.axes.set_xlabel(T_label)
                
            # Only specify name of functional if more than one is used
            if col == 0:
                if len(functionals) == 1:
                    functional_label =  r'$\mu$ / kJ mol$^{-1}$'
                elif functional in functional_names:
                    functional_label = functional_names[functional] + '\n' + r'$\mu$ / kJ mol$^{-1}$'
                else:
                    functional_label = functional + '\n' + r'$\mu$ / kJ mol$^{-1}$'
                ax.axes.set_ylabel(functional_label)
            else:
                plt.setp(ax.get_yticklabels(), visible=False) # I don't understand why ax.set_yticklabels doesn't work here, but it wipes out the first column too.
                
    plt.legend([plt.Line2D((0,1),(0,0), color=species_colors[species]) for species in ordered_species] + [plt.Line2D((0,1),(0,0), color='m', linestyle=':', linewidth=4)],
               [species_names[species] for species in ordered_species] + ['Mixture'], ncol=4, loc='center', bbox_to_anchor=(0.5,bottom/3.), bbox_transform=fig.transFigure, fontsize=11)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close(fig)

def tabulate_data(data,T,P,path='',formatting=('kJmol-1')):
    """
    Write tables of composition and free energy. Write enthalpy in table if data available.
    
    Arguments:
            data: dict containing Calc_n_mu namedtuples, with keys corresponding to 'functionals'.
              Each namedtuple contains the nested lists n[P][T], mu[P][T] and list labels.
              n: atom frac of S of each species, corresponding to labels
              mu: free energy of mixture on atom basis in J mol-1
              H: enthalpy of mixture on atom basis in J mol-1
              labels: identity labels of each species in mixture
            T: Iterable containing temperature values in K corresponding to data
            P: Iterable containing pressure values in Pa corresponding to data
            path: directory for csv files to be written in
            logP: Boolean
            formatting: iterable of strings. 'logP' for log pressures. Set units with
                        'Jmol-1'|'Jmol'|'J/mol'|'kJmol-1'|'kJmol'|'kJ/mol'. Reduce decimal precision with 'short'.
    """

    import string
    if path:
        if path[-1] != '/':
            path = path + '/'

    if formatting and any([x in formatting for x in ('Jmol-1','Jmol','J/mol')]):
        energy_units = 'J mol-1'
        energy_units_factor = 1.0
    elif formatting and any([x in formatting for x in ('kJmol-1','kJmol','kJ/mol')]):
        energy_units = 'kJ mol-1'
        energy_units_factor = 1.0e-3
    else:
        raise Exception("no valid units in format string {0}".format(formatting))

    if 'short' in formatting:
        energy_string='{0:1.2f}'
    else:
        energy_string='{0:1.4f}'
    for functional in data.keys():
        with open(path + 'mu_{0}.csv'.format(functional.lower()), 'w') as f:
            if 'logP' in formatting:
                linelist = ['# T/K,' + string.join(['mu (10^{0:.2f} Pa) / {1}'.format(np.log10(p), energy_units) for p in P],',') + '\n']
            else:
                linelist = ['# T/K,' + string.join(['mu ({0} Pa) / {1}'.format(p, energy_units) for p in P],',') + '\n']
            for t_index, t in enumerate(T):
                linelist.append( '{0},'.format(t) + string.join([energy_string.format(mu_p[t_index]*energy_units_factor) for mu_p in data[functional].mu],',') + '\n')
            f.writelines(linelist)

    for functional in data.keys():
        with open(path + 'n_{0}.csv'.format(functional.lower()), 'w') as f:
            for p_index, p in enumerate(P):
                if 'logP' in formatting:
                    linelist = ['# P = 10^{0:.2f} Pa\n'.format(np.log10(p))]
                else:
                    linelist = ['# P = {0} Pa\n'.format(p)]
                linelist.append('# T/K, ' + string.join(['x({0})'.format(x) for x in data[functional].labels],',') + '\n')
                for t_index, t in enumerate(T):
                    linelist.append('{0},'.format(t) + string.join(['{0:1.4f}'.format(n) for n in data[functional].n[p_index][t_index]],',') + '\n')
                f.writelines(linelist)

    for functional in data.keys():
        if type(data[functional].H) != bool:
            with open(path + 'H_{0}.csv'.format(functional.lower()), 'w') as f:
                if 'logP' in formatting:
                    linelist = ['# T/K,' + string.join(['H (10^{0:.2f} Pa) / {1}'.format(np.log10(p), energy_units) for p in P],',') + '\n']
                else:
                    linelist = ['# T/K,' + string.join(['H ({0} Pa) / {1}'.format(p, energy_units) for p in P],',') + '\n']
                for t_index, t in enumerate(T):
                    linelist.append( '{0},'.format(t) + string.join([energy_string.format(H_p[t_index]*energy_units_factor) for H_p in data[functional].H],',') + '\n')
                f.writelines(linelist)

             

def plot_mix_contribution(T, P, data, functional='PBE0_scaled', filename=False, figsize=(8.4/2.52, 8.4/2.54)):
    """
    Plot contribution of mixing entropy and minor phases to free energy.

    Arguments:
        T: Iterable containing temperature values in K corresponding to data
        P: Iterable containing pressure values in Pa corresponding to data
        data: dict containing Calc_n_mu namedtuples, with keys corresponding to 'functionals'.
              Each namedtuple contains the nested lists n[P][T], mu[P][T] and list labels.
              n: atom frac of S of each species, corresponding to labels
              mu: free energy of mixture on atom basis in J mol-1
              labels: identity labels of each species in mixture
        functional: Dataset to plot; must be a key in data
        filename: Filename for plot output. If False (default), print to screen instead.
        figsize: Figure dimensions in inches
    
    """

    T = np.array(T)

    db_file = data_directory + '/' + data_sets[functional]
    labels, thermo, a = unpack_data(db_file, ref_energy=reference_energy(db_file, units='eV'))
    S2_thermo = thermo[labels.index('S2')]
    S8_thermo = thermo[labels.index('S8')]

    def get_gibbs_wrapper(thermo, T, P):
        return(ase.thermochemistry.IdealGasThermo.get_gibbs_energy(thermo,T,P,verbose=False))
    v_get_gibbs_energy=np.vectorize(get_gibbs_wrapper)

    fig = plt.figure(figsize=figsize)
    linestyles=['-','--',':','-.']

    linecycler = cycle(linestyles)
    for p_index, p in enumerate(P):
        mu_S2 = v_get_gibbs_energy(S2_thermo,T, p) * eV2Jmol / 2.
        mu_S8 = v_get_gibbs_energy(S8_thermo,T, p) * eV2Jmol / 8.

        plt.plot(T,( data[functional].mu[p_index] - np.minimum(mu_S2, mu_S8))*1e-3, label=r'$10^{{{0:1.0f}}}$ Pa'.format(np.log10(p)), linestyle=linecycler.next(), color='k')

    plt.xlabel('Temperature / K')
    plt.ylabel(r'$\Delta \mu_{\mathrm{mixture}}$ / kJ mol$^{-1}$')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.2), ncol=2)
    plt.subplots_adjust(left=0.26,bottom=0.3)

    ax = plt.gca()
    ml = MultipleLocator(400)
    ax.xaxis.set_major_locator(ml)
    ax.axes.set_xlim([400,1500])


    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close(fig)

def plot_surface(functional='PBE0_scaled', T_range=(400,1500), P_range=(1,7), resolution=1000, tolerance = 1e4, parameterised=True, filename=False, plot_param_err=False, nodash=False):
    """Generate a surface plot showing recommended S models. Can be slow!

    Arguments:
        functional: id of dataset used. PBE0_scaled is strongly recommended as it has good agreement with experimental data.
        T_range: Tuple containing T range in K
        P_range: Tuple containing (log10(Pmin), log10(Pmax))
        resolution: Number of points on x and y axis. Note that a full free energy minimisation is carried out at each point, so print-ready resolutions will take some time to compute.
        tolerance: Error threshold for transition region in Jmol-1
        parameterised: Boolean. If True, use parameterised fit (polynomials, erf and gaussian). If False, solve equilibrium at all points (slow!)
        filename: String containing output file. If False, print to screen.
        plot_param_err: Boolean; request an additional plot showing error of parameterisation
        nodash: Boolean; Skip drawing coexistence line
    

    """

    figsize = (8.3/2.54, 8.3/2.54)


    T = np.linspace(min(T_range), max(T_range), resolution)
    P = 10**np.linspace(min(P_range),max(P_range),resolution)[:, np.newaxis]

    if parameterised:
        mu_mixture = mu_fit(T,P)
        mu_S2 = mu_S2_fit(T,P) * eV2Jmol / 2.
        mu_S8 = mu_S8_fit(T,P) * eV2Jmol / 8.
    else:
        cache = shelve.open('cache')
        if cache.has_key('surface'):
            T, P, data = cache['surface']
        else:
            data = compute_data(T=T, P=P, functionals=[functional])
            cache['surface'] = (T, P, data)
            cache.close()
        mu_mixture = np.array(data[functional].mu)
        db_file = data_directory+'/'+data_sets[functional]
        labels, thermo, a = unpack_data(db_file,ref_energy=reference_energy(db_file, units='eV'))
        S2_thermo = thermo[labels.index('S2')]
        S8_thermo = thermo[labels.index('S8')]

        def get_gibbs_wrapper(thermo, T, P):
            return(ase.thermochemistry.IdealGasThermo.get_gibbs_energy(thermo,T,P,verbose=False))
        v_get_gibbs_energy=np.vectorize(get_gibbs_wrapper)
        mu_S2 = v_get_gibbs_energy(S2_thermo,T, P) * eV2Jmol / 2.
        mu_S8 = v_get_gibbs_energy(S8_thermo,T, P) * eV2Jmol / 8.

    fig = plt.figure(figsize = figsize)
    CS = plt.contour(T,np.log10(P).flatten(),np.minimum(abs(mu_S2 - mu_mixture),abs(mu_S8 - mu_mixture)), [1000])
    plt.contourf(T,np.log10(P).flatten(),np.minimum(abs(mu_S2 - mu_mixture),abs(mu_S8 - mu_mixture)), [1000,1e10], colors=[(0.7,0.7,1.00)])
    # plt.clabel(CS, inline=1, fontsize=10)  # Contour line labels

    if not nodash:
        plt.plot(T_tr(P),np.log10(P),'k--', linewidth=3)
    plt.xlim(min(T_range),max(T_range))
    plt.text(500, 4, r'S$_{8}$')
    plt.text(1000, 4, r'S$_{2}$')

    plt.xlabel('Temperature / K')
    plt.ylabel('$\log_{10}( P / \mathrm{Pa})$')

    fig.subplots_adjust(bottom=0.15, left=0.15)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(base=200))

    if filename:
        plt.savefig(filename)
    else:
        plt.show()

    if plot_param_err:
        mu_param = mu_fit(T,P)
        fig2 = plt.figure(figsize=figsize)
        CS2 =plt.contour(T,np.log10(P).flatten(), (mu_param - mu_mixture)*1e-3, cmap='Greys')
        plt.clabel(CS2, inline=1, fontsize=10, colors='k', fmt='%2.1f')
        plt.xlabel('Temperature / K')
        plt.ylabel('$\log_{10}( P / \mathrm{Pa})$')

        fig2.subplots_adjust(left=0.15, bottom=0.15)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(base=200))

        if filename:
            err_filename = os.path.dirname(filename) + '/param_error.' + os.path.basename(filename)
            plt.savefig(err_filename)
        else:
            plt.show()
        plt.close(fig)

def check_fit():
    """Sanity check for polynomial fitting"""

    T = np.linspace(100,1000,10)
    P = np.array([1E3])
    data = compute_data(T=T, P=P, functionals=['PBE0_scaled'])
    mu_mixture = np.array(data['PBE0_scaled'].mu)
    db_file = data_directory+'/'+data_sets['PBE0_scaled']
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

def plot_energies(functionals=data_sets.keys(), filename=False, figsize=False):
    if not figsize:
        figsize = (8.3/2.54, 12/2.54)
    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(1,1,left=0.2, bottom=0.35)
    ax = plt.subplot(gs[0,0])
    colors = ['r','g','b','k']
    colorcycler = cycle(colors)
    
    for functional in functionals:
        color=colorcycler.next()
        db_file = data_directory+'/'+data_sets[functional]
        c = ase.db.connect(db_file)
        E_ref = c.get_atoms('S8').get_total_energy()/8.
        for species in ordered_species:
            atoms = c.get_atoms(species)
            E = atoms.get_total_energy()
            N = atoms.get_number_of_atoms()
            ax.plot(N, ((E/N)-E_ref)*eV2Jmol*1e-3, marker=species_markers[species], fillstyle='none', color=color)

    # Value from reference data
    ref_col=(0.4,0.4,0.4)
    S2_ref_DH0 = 128.300
    S8_ref_DH0 = 104.388

    ref_value = S2_ref_DH0/2. - S8_ref_DH0/8.
    ax.plot(2,ref_value,marker='8',fillstyle='full',color=ref_col, linestyle='none')

    plt.xlim(1.5,8.5)
    plt.xlabel(r'$N$ / S atoms')
    plt.ylabel(r'$\frac{E_0}{N} - \frac{E_{0,\mathrm{S}_8}}{8}$ / kJ mol$^{-1}$')

    colorcycler=cycle(colors) # reset colorcycler
    plt.legend([plt.Line2D((0,1),(0,0), color='k', linestyle='none',
                           marker=species_markers[s], fillstyle='none') for s in ordered_species] +
                           [plt.Line2D((0,1),(0,0), color=colorcycler.next(), marker=False) for f in functionals] +
                           [plt.Line2D((0,1),(0,0),color=ref_col, marker='8', fillstyle='full', linestyle='none')],
               [species_names[s] for s in ordered_species] + functionals + [r'S$_2$ [ref.]'],
               ncol=3, loc='center', bbox_to_anchor=(0.5,0.12), numpoints=1, fontsize=8, bbox_transform=fig.transFigure)
    

    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close(fig)
    
def main(plots='all', tables='all', T_range=(400,1500)):
    """
    Solve sulfur equilibrium from ab initio data; generate plots and tables.

    A cache file is used to avoid redundant calculations.

    arguments:
        plots: list of strings indicating desired plots.
                ['all'] is equivalent to ['energies', 'composition','mu_functionals',
                                          'mu_all_functionals','mu_contributions',
                                          'mu_annealing','mix_contribution',
                                          'surface','freqs']

        tables: list of strings indicating results sets to include as tables.
                ['all'] is equivalent to ['LDA','PBEsol','PBE0','PBE0_scaled','B3LYP']
                In addition the strings 'long', 'short', 'linear', 'logP' can be used
                to set the formatting (default is 'short' and 'logP' behaviour)

        T_range: 2-Tuple of upper and lower temperature limits
    
    """

    ### Open cache file for plot data

    cache = shelve.open('cache')
    if 'T_range' in cache and cache['T_range'] == T_range:
        cache_T_okay = True
    else:
        cache_T_okay = False
        cache['T_range'] = T_range
        

    ### Comparison of DFT energies
    if 'all' in plots or 'energies' in plots:
        plot_energies(functionals=['LDA','PBEsol','B3LYP','PBE0'], filename='plots/energies.pdf', figsize=False)
    
    ### Plot composition breakdown with PBE0_scaled at 3 pressures ###
    if 'composition' in 'plots':

        if cache_T_okay and cache.has_key('PBE0_composition'):
            (T, P, data) = cache['PBE0_composition']
        else:
            T = np.linspace(T_range[0],T_range[1],100)
            P = [10**x for x in (1,5,7)]
            data = compute_data(T=T, P=P, functionals=data_sets.keys())
            cache['PBE0_composition'] = (T, P, data)
            cache.sync()

        plot_composition(T,P, data, filename='plots/composition_all.pdf')
        plot_composition(T, P, data, functionals=['LDA', 'PBEsol', 'PBE0_scaled'], filename='plots/composition_selection.pdf')

    ### Plots over 3 pressures: mu depending on T, calculation method; mu with
    ### component contributions; mu with component contributions over smaller T
    ### range
    if any(flag in plots for flag in ('all','mu_functionals','mu_all_functionals',
                                      'mu_contributions','mu_annealing')):
        
        if cache_T_okay and cache.has_key('all_functionals_three_pressures'):
            (T, P, data) = cache['all_functionals_three_pressures']
        else:
            T = np.linspace(T_range[0],T_range[1],100)
            P = [10**x for x in (1,5,7)]
            data = compute_data(T=T, P=P, functionals = data_sets.keys())
            cache['all_functionals_three_pressures'] = (T, P, data)
            cache.sync()
    if 'all' in plots or 'mu_all_functionals' in plots:
        plot_mu_functionals(data, T, P, mu_range=(-200,100), filename='plots/mu_all_functionals.pdf', compact=False, functionals=('LDA','PBEsol','B3LYP','PBE0','PBE0_scaled'))
    if 'all' in plots or 'mu_functionals' in plots:
        plot_mu_functionals(data, T, P, mu_range=(-200,100), filename='plots/mu_functionals.pdf', compact=False, functionals=('LDA','PBEsol','PBE0_scaled'))  
    if 'all' in plots or 'mu_contributions' in plots:
        plot_mu_contributions(T, P, data, functionals=['PBE0_scaled'], filename='plots/mu_contributions.pdf', figsize=(17.2/2.54, 10/2.54), T_range=[400,1500], T_increment=400)
    if 'all' in plots or 'mu_annealing' in plots:
        plot_mu_contributions(T,P,data,functionals=['PBE0_scaled'],filename='plots/mu_for_annealing.pdf', figsize=(17.2/2.54, 10/2.43), T_range=(100,600), T_units='C', T_increment=100, mu_range=(-100,50))

    ### Plot contribution of mixing and secondary phases over a range of pressures
    if 'all' in plots or 'mix_contribution' in plots:
        if cache_T_okay and cache.has_key('PBE0_four_pressures'):
            (T, P, data) = cache['PBE0_four_pressures']
        else:
            T = np.linspace(400,1500,100)
            P = [10**x for x in (1.,3.,5.,7.)]
            data = compute_data(T=T, P=P, functionals=['PBE0_scaled'])
            cache['PBE0_four_pressures'] = (T, P, data)
            cache.sync()
        plot_mix_contribution(T, P, data, functional='PBE0_scaled', filename='plots/mu_mix_contribution.pdf', figsize=(8.4/2.52, 8.4/2.54))

    ### Tabulate data over log pressure range ###
    if 'linear' in tables:
        formatting=[]
    else:
        formatting=['logP']
    if any(flag in tables for flag in ('Jmol','Jmol-1','J/mol')):
        formatting.append('Jmol-1')
    else:
        formatting.append('kJmol-1')

        # Compact tables

    if 'all' in tables or 'short' in tables:
        T = np.arange(400,1500,50)
        P = np.power(10,np.linspace(1,7,10))
        data = compute_data(T=T, P=P, functionals = data_sets.keys(), enthalpy=True)
        tabulate_data(data,T,P, path=data_directory+'/alpha_ref', formatting=formatting+['short'])

        data = compute_data(T=T, P=P, functionals = data_sets.keys(), ref_energy='S8', enthalpy=True)
        tabulate_data(data,T,P, path=data_directory+'/S8_ref', formatting=formatting+['short'])

        # Larger tables
    if 'all' in tables or 'long' in tables:
        T = np.arange(200,1500,10)
        P = np.power(10,np.linspace(1,7,20))
        data = compute_data(T=T, P=P, functionals = data_sets.keys(), enthalpy=True)
        tabulate_data(data,T,P, path=data_directory+'/precise/alpha_ref', formatting=formatting)
        data = compute_data(T=T, P=P, functionals = data_sets.keys(), ref_energy='S8', enthalpy=True)
        tabulate_data(data,T,P, path=data_directory+'/precise/S8_ref', formatting=formatting)

    ### Contour plots (high resolution -> Lots eqm solutions -> v. slow data calculation)
    cache.close()
    if 'all' in plots or 'surface' in plots:
        plot_surface(resolution=200, parameterised=False, filename='plots/surface.pdf', plot_param_err=True)

    # Vibrational frequencies
    if 'all' in plots or 'freqs' in plots:
        plot_frequencies(functionals=['LDA','PBEsol','B3LYP','PBE0','PBE0_scaled'], figsize=False, filename='plots/empirical_freqs.pdf')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plots", type=str, default="all",
                        help="Plots to generate (as space-separated string). Options are: all"
                        " energies composition mu_functionals mu_all_functionals"
                        " mu_contributions mu_annealing mix_contribution surface freqs ")
    parser.add_argument("-t", "--tables", type=str, default="all",
                        help="Space-separated string: list of results sets to include as tables."
                " 'all' is equivalent to 'LDA PBEsol PBE0 PBE0_scaled B3LYP'"
                " In addition the strings 'long', 'short', 'linear', 'logP' can be used"
                " to set the formatting (default is 'short' and 'logP' behaviour)")
    parser.add_argument("-T", "--temp_range", type=float, nargs=2, default=(400.,1500.),
                        help="Lower and upper temperature limit in K")
    args = parser.parse_args()
    plots = args.plots.split()
    tables = args.tables.split()
    
    main(plots=plots, tables=tables, T_range=args.temp_range)
