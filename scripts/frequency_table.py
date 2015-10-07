#! /usr/bin/env python

import ase
import ase.db
import sys
import os.path
import itertools
script_directory = os.path.dirname(__file__)

# Append a trailing slash to make coherent directory name - this would select the
# root directory in the case of no prefix, so we need to check
if script_directory:
    script_directory  = script_directory + '/'
module_directory = os.path.abspath(script_directory + '..')
data_directory = os.path.abspath(script_directory +  '../data')
sys.path.insert(0,module_directory)

ordered_species = ['S2','S3_ring','S3_bent','S4_buckled','S4_eclipsed','S5_ring','S6_stack_S3','S6_branched','S6_buckled','S6_chain_63','S7_ring','S7_branched','S8']

ordered_data_sets = ['LDA','PBEsol','B3LYP','PBE0','PBE0_scaled']

data_sets = {'LDA':'sulfur_lda.json', 'PBEsol':'sulfur_pbesol.json', 'PBE0':'sulfur_pbe0.json', 'PBE0_scaled':'sulfur_pbe0_96.json', 'B3LYP':'sulfur_b3lyp.json'}

db_dict = {
    label: ase.db.connect(data_directory + '/' + db_file) for label, db_file in data_sets.iteritems()
    }

def frequencies(species,data_set):
    """
    Return list of vibrational frequencies in cm-1
    """
    db = db_dict[data_set]
    try:
        frequencies = db.get_dict(species)['data']['frequencies']
    except KeyError:
        return None
        
    return list(frequencies)

def print_species_frequencies(species,formatting='{0:10.2f}'+5*' '):
    """Given a species name, print a table of frequencies corresponding
    to data sets in variable "ordered_data_sets"
    """
    print "".join('{0:15s}'.format(d) for d in ordered_data_sets)
    for row in itertools.izip_longest(*(frequencies(species,d) for d in ordered_data_sets)):
        print "".join(formatting.format(f) for f in row)

def pretty_print_name(species):
    for _, db in db_dict.iteritems():
        try:
            sx_dict = db.get_dict(species)
            break
        except ValueError:
            pass
    else:
        return None

    pointgroup = sx_dict['data']['pointgroup']
    N = len(sx_dict['positions'])

    return "S_{0:d} ({1:s})".format(N, pointgroup)

def main():
    for species in ordered_species:
        print pretty_print_name(species)
        print_species_frequencies(species)
        print ""

if __name__ == '__main__':
    main()
