from collections import namedtuple
import os
import ase.io
import ase.db

homedir = os.path.expanduser('~')
materials_dir = homedir + '/Documents/Data/materials'

"""Generate a JSON database for computed sulfur polytypes for thermochemical modelling.

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
    polytypes = [
        Species('S8', 'S/S8_g/vibrations/317/basic.central.out',
                vibs_from('S/S8_g/vibrations/317/basic.vib.out'),
                'nonlinear', 8, 'D4d'),
        Species('S7_branched', 'S/S7/branched/vibrations/317/basic.central.out',
                vibs_from('S/S7/branched/vibrations/317/basic.vib.out'),
                'nonlinear', 1, 'Cs'),
        Species('S7_ring','S/S7/ring/vibrations/317/basic.central.out',
                vibs_from('S/S7/ring/vibrations/317/basic.vib.out'),
                'nonlinear', 1, 'Cs'),
        Species('S6_branched','S/S6/branch/vibrations/346/basic.central.out',
                vibs_from('S/S6/branch/vibrations/346/basic.vib.out'),
                'nonlinear', 1, 'Cs'),
        Species('S6_buckled','S/S6/buckled/vibrations/346/basic.central.out',
                vibs_from('S/S6/buckled/vibrations/346/basic.vib.out'),
                'nonlinear', 2, 'C2v'),
        # Species('S6_crown','S/S6/crown/vibrations/346/basic.central.out',
        #         vibs_from('S/S6/crown/vibrations/346/basic.vib.out'),
        #         'nonlinear', 2, 'C2v'),
        Species('S6_stack_S3','S/S6/s3_stack/vibrations/346/basic.central.out',
                vibs_from('S/S6/s3_stack/vibrations/346/basic.vib.out'),
                'nonlinear', 6, 'D3h'),
        Species('S6_chain_63','S/S6/chains/63/vibrations/346/basic.central.out',
                vibs_from('S/S6/chains/63/vibrations/346/basic.vib.out'),
                'nonlinear', 1, 'C1'),
        Species('S5_ring','S/S5/vibrations/322/basic.central.out',
                vibs_from('S/S5/vibrations/322/basic.vib.out'),
                'nonlinear', 1, 'Cs'),
        Species('S4_eclipsed', 'S/S4/eclipsed/vibrations/315/basic.central.out',
                vibs_from('S/S4/eclipsed/vibrations/315/basic.vib.out'),
                'nonlinear', 2, 'C2v'),
        Species('S4_buckled', 'S/S4/square/vibrations/315/basic.central.out',
                vibs_from('S/S4/square/vibrations/315/basic.vib.out'),
                'nonlinear', 4, 'D2d'),
        Species('S3_ring', 'S/S3/cyclic/vibrations/322/basic.central.out',
                vibs_from('S/S3/cyclic/vibrations/322/basic.vib.out'),
                'nonlinear', 6, 'D3h'),
        Species('S3_bent', 'S/S3/bent/vibrations/322/basic.central.out',
                vibs_from('S/S3/bent/vibrations/322/basic.vib.out'),
                'nonlinear', 2, 'C2v'),
        Species('S2', 'S/S2/vibrations/144/0.0025.central.out',
                vibs_from('S/S2/vibrations/144/0.0025.vib.out'),
                'linear', 2, 'Dinfh')
        ]

    c = ase.db.connect('sulfur_pbesol.json')
    for species in polytypes:
        atoms = ase.io.read(materials_dir + '/' + species.structure_path)
        id = species.id
        c.write(id,atoms, data={'frequencies': species.frequencies,
                                'geometry': species.geometry,
                                'symmetry': species.symmetry,
                                'pointgroup': species.pointgroup})

    ### LDA ###
    pointgroups = {'S8':'D4d', 'S7_branched':'Cs', 'S7_ring':'Cs', 'S6_branched':'Cs',
                   'S6_buckled':'C2v', 'S6_stack_S3':'D3h', 'S6_chain_63':'C1',
                   'S5_ring':'Cs', 'S4_eclipsed':'C2v', 'S4_buckled':'D2d', 'S3_ring':'D3h',
                   'S3_bent':'C2v','S2':'Dinfh'}
    rot_sym = {'C1':['nonlinear',1], 'Cs':['nonlinear',1], 'C2v':['nonlinear',2],
               'D3h':['nonlinear',6], 'D2d':['nonlinear',4],'D4d':['nonlinear',8],
               'Dinfh':['linear',2]}
    
    c = ase.db.connect('sulfur_lda.json')
    lda_calc_dir='/Users/adamjackson/runs/czts/403'
    for species, pointgroup in pointgroups.iteritems():
        atoms = ase.io.read(lda_calc_dir + '/' + species + '/basic.central.out')
        vibs = vibs_from(lda_calc_dir + '/' + species + '/basic.vib.out', abs_path=True)
        c.write(species,atoms, data={'frequencies':vibs,
                                     'geometry':rot_sym[pointgroup][0],
                                     'symmetry':rot_sym[pointgroup][1],
                                     'pointgroup':pointgroup})

    ### PBE0 ###
    c = ase.db.connect('sulfur_pbe0.json')
    c_96 = ase.db.connect('sulfur_pbe0_96.json')
    pbe0_calc_dir='/Users/adamjackson/runs/czts/407'
    for species, pointgroup in pointgroups.iteritems():
        atoms = ase.io.read(pbe0_calc_dir + '/' + species + '/basic.central.out')
        vibs = vibs_from(pbe0_calc_dir + '/' + species + '/basic.vib.out', abs_path=True)
        c.write(species,atoms, data={'frequencies':vibs,
                                     'geometry':rot_sym[pointgroup][0],
                                     'symmetry':rot_sym[pointgroup][1],
                                     'pointgroup':pointgroup})
        c_96.write(species,atoms, data={'frequencies':[v * 0.96 for v in vibs],
                                     'geometry':rot_sym[pointgroup][0],
                                     'symmetry':rot_sym[pointgroup][1],
                                     'pointgroup':pointgroup})



if (__name__ == '__main__'):
    main()
