#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import gridspec

from matplotlib import rc, rcParams
rc('font',**{'family':'serif', 'weight':'normal'})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
rcParams['text.latex.preamble'] = [r'\boldmath']

import os # get correct path for datafiles when called from another directory
import sys # PATH manipulation to ensure sulfur module is available
from itertools import izip
from collections import namedtuple

script_directory = os.path.dirname(__file__)
# Append a trailing slash to make coherent directory name - this would select the
# root directory in the case of no prefix, so we need to check
if script_directory:
    script_directory  = script_directory + '/'
module_directory = os.path.abspath(script_directory + '..')
data_directory = os.path.abspath(script_directory +  '../data')
sys.path.insert(0,module_directory)

from sulfur import get_potentials, unpack_data, reference_energy, solve_composition

ordered_species = ['S2','S3_ring','S3_bent','S4_buckled','S4_eclipsed', 'S4_C2h','S5_ring','S6_stack_S3','S6_branched','S6_buckled','S6_chain_63','S7_ring','S7_branched','S8']

data_sets = {'LDA':'sulfur_lda.json', 'PBEsol':'sulfur_pbesol.json', 'PBE0':'sulfur_pbe0.json', 'PBE0_scaled':'sulfur_pbe0_96.json', 'B3LYP':'sulfur_b3lyp.json'}

species_colors = {'S2':'#222222','S3_ring':'#a6cee3','S3_bent':'#1f78b4','S4_buckled':'#b2df8a','S4_eclipsed':'#33a02c','S5_ring':'#fb9a99','S6_stack_S3':'#e31a1c','S6_branched':'#fdbf6f','S6_buckled':'#ff7f00','S6_chain_63':'#cab2d6','S7_ring':'#6a3d9a','S7_branched':'#bbbb55','S4_C2h':'#b04040','S8':'#b15928'}

# LaTeX formatted names for species. Keys correspond to database keys
species_names = {'S2':r'S$_2$ (D$_{\infty \mathrm{h}}$)','S3_ring':r'S$_3$ (D$_{3\mathrm{h}}$)','S3_bent':r'S$_3$ (C$_{2\mathrm{v}}$)','S4_buckled':r'S$_4$ (D$_{2\mathrm{d}}$)','S4_eclipsed':r'S$_4$ (C$_{2\mathrm{v}}$)','S4_C2h':r'S$_4$ (C$_{2\mathrm{h}}$)','S5_ring':r'S$_5$ (C$_\mathrm{s}$)','S6_stack_S3':r'S$_6$ (D$_{3 \mathrm{h}}$)','S6_branched':r'S$_6$ (C$_1$, branched)','S6_buckled':r'S$_6$ (C$_{2\mathrm{v}}$)','S6_chain_63':r'S$_6$ (C$_1$, chain)','S7_ring':r'S$_7$ (C$_{\mathrm{s}}$)','S7_branched':r'S$_7$ (C$_\mathrm{s}$, branched)','S8':r'S$_8$ (D$_{4\mathrm{d}}$)'}

# Alternative / LaTeX escaped names for DFT functionals. May also be useful for changing capitalisation, LDA vs LSDA etc.
functional_names = {'PBE0_scaled':r'PBE0 (scaled)'} 

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

def plot_composition(T, P, data, functionals=data_sets.keys(), filename=False):
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

    fig = plt.figure(figsize =  (17.2 / 2.54, 17 / 2.54))
    gs = gridspec.GridSpec(len(functionals), len(P), bottom=0.25)

    tick_length = 4
    tick_width = 0.5
    
    for row, functional in enumerate(functionals):
        color_cycle = [species_colors[species] for species in data[functional].labels]
        for col, p in enumerate(P):
            ax = plt.subplot(gs[row,col])
            ax.set_color_cycle(color_cycle)
            ax.plot(T, data[functional].n[col][:])
            ml = MultipleLocator(400)
            ax.xaxis.set_major_locator(ml)
            ax.axes.set_ylim([0,1])
            ax.axes.set_xlim([200,1500])
            
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
        db_file = data_directory + '/' + data_sets[functional]
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

def plot_mu_functionals(data, T, P, functionals=False,  T_range=False, filename=False, compact=False):
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
        T_range: 2-tuple in K of temperature range to display.
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

    
    plt.legend(ncol=4, loc='center', bbox_to_anchor=(0.5,bottom/3.), bbox_transform=fig.transFigure, fontsize=11)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def plot_mu_contributions( T, P, data, functionals, T_range=(200,1500), filename=False, figsize=(17.2 / 2.54, 17 / 2.54), bottom=0.4, T_units='K', T_increment=400, mu_range=False):
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
        
def main():
    # T = np.linspace(50,1500,50)
    # P = 1E5
    # data = compute_data(T=T, functionals=['PBE0_scaled'])
    # plot_T_composition(T, data['PBE0_scaled'].n[0], data['PBE0_scaled'].labels, 'PBE0, P = 1E5' , filename=False)

    T = np.arange(50,1500,100)
    P = [10**x for x in (1,5,7)]
    data = compute_data(T=T, P=P, functionals = data_sets.keys())
    tabulate_data(data,T,P, path=data_directory)
    
    #plot_composition(T, P, data, functionals=('LDA','PBEsol','PBE0_scaled'), filename='composition.pdf')
    #plot_composition(T, P, data, functionals=data_sets.keys(), filename='all_compositions.pdf')
    plot_mu_functionals(data, T, P, filename='mu_functionals.pdf', compact=False, functionals=('LDA','PBEsol','B3LYP','PBE0','PBE0_scaled'))  

    plot_mu_contributions(T, P, data, functionals=['PBE0_scaled'], filename='mu_contributions.pdf', figsize=(17.2/2.54, 10/2.54))
    # "Experimentalist-friendly" plot around annealing conditions in degrees C
    plot_mu_contributions(T,P,data,functionals=['PBE0_scaled'],filename='mu_for_annealing.pdf', figsize=(17.2/2.54, 10/2.43), T_range=(100,600), T_units='C', T_increment=100, mu_range=(-100,50))
                
if __name__ == '__main__':
    main()
