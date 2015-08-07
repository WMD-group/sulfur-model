from collections import namedtuple
import os
import ase.io
import ase.db

homedir = os.path.expanduser('~')
materials_dir = homedir + '/Documents/Data/materials'
data_dir = homedir + '/Documents/Data/datasets/sulfur_clusters'

pointgroups = {'S8':'D4d', 'S7_branched':'Cs', 'S7_ring':'Cs', 'S6_branched':'Cs',
                   'S6_buckled':'C2v', 'S6_stack_S3':'D3h', 'S6_chain_63':'C1',
                   'S5_ring':'Cs', 'S4_eclipsed':'C2v', 'S4_buckled':'D2d', 'S3_ring':'D3h',
                   'S3_bent':'C2v','S2':'Dinfh'}
# pointgroup  'S4_C2h':'C2h' removed from dict as this species is unstable (negative frequencies)
rot_sym = {'C1':['nonlinear',1], 'Cs':['nonlinear',1], 'C2v':['nonlinear',2], 'C2h':['nonlinear',2],
               'D3h':['nonlinear',6], 'D2d':['nonlinear',4],'D4d':['nonlinear',8],
               'Dinfh':['linear',2]}


"""Generate a JSON database for computed sulfur allotropes for thermochemical modelling.

The database format is that used by ASE, with the following custom data fields:
frequencies: List of vibrational mode frequencies, including zero-frequency modes, in cm-1
geometry: takes the values 'linear', 'nonlinear', 'monatomic'. Needed for calculation of rotational energy.
symmetry: symmetry number; integer number of equivalent rotations.
pointgroup: string indicating point group of molecule.

"""

Species = namedtuple('Species', 'id structure_path frequencies geometry symmetry pointgroup')


def vibs_from(path, abs_path=False):
    if abs_path:
        full_path=path
    else:
        full_path = materials_dir + '/' + path
    vib_list=[]
    with open(full_path,'r') as f:
        for line in f:
            if line[0:13] == "  Mode number":
                break
        for line in f:
            if len(line.split()) == 4:
                vib_list.append(float(line.split()[1]))
            else:
                break
    return vib_list


def main():

    for functional in 'PBEsol','PBE0', 'LDA', 'B3LYP':
        c = ase.db.connect('sulfur_' + functional.lower() + '.json')
        calc_dir = data_dir + '/' + functional
        for species, pointgroup in pointgroups.iteritems():
            try:
                atoms = ase.io.read(calc_dir + '/' + species + '/vibs/basic.central.out')
                
                vibs = vibs_from(calc_dir + '/' + species + '/vibs/basic.vib.out', abs_path=True)
                c.write(species, atoms, data={'frequencies':vibs,
                                          'geometry':rot_sym[pointgroup][0],
                                          'symmetry':rot_sym[pointgroup][1],
                                          'pointgroup':pointgroup
                                          })
            except IOError:
                pass


    c = ase.db.connect('sulfur_pbe0.json')
    c_96 = ase.db.connect('sulfur_pbe0_96.json')


    for allotrope in c.select():
        atoms = c.get_atoms(allotrope.id)
        data = allotrope.data
        data.update({'frequencies':[v * 0.96 for v in data['frequencies']]})

        c_96.write(allotrope.id, atoms, data=data)

    
if __name__ == '__main__':
    main()
