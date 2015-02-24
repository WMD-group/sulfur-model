from collections import namedtuple
import os
import ase.io
import ase.db

homedir = os.path.expanduser('~')
materials_dir = homedir + '/Documents/Data/materials'

Species = namedtuple('Species', 'id structure_path frequencies geometry symmetry')


def vibs_from(path):
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
        Species('S8', 'S/S8_g/best_structure/clean/geometry.in', vibs_from('S/S8_g/vibrations/317/basic.vib.out')),
        Species('S7_branched', 'S/S7/branched/vibrations/317/geometry.in.basic', vibs_from('S/S7/branched/vibrations/317/basic.vib.out')),
        Species('S7_ring','S/S7/ring/vibrations/317/geometry.in.basic', vibs_from('S/S7/ring/vibrations/317/basic.vib.out')),
        Species('S6_branched','S/S6/branch/vibrations/346/geometry.in.basic', vibs_from('S/S6/branch/vibrations/346/basic.vib.out')),
        Species('S6_buckled','S/S6/buckled/vibrations/346/geometry.in.basic', vibs_from('S/S6/buckled/vibrations/346/basic.vib.out')),
        Species('S6_crown','S/S6/crown/vibrations/346/geometry.in.basic', vibs_from('S/S6/crown/vibrations/346/basic.vib.out')),
        Species('S6_stack_S3','S/S6/s3_stack/vibrations/346/geometry.in.basic', vibs_from('S/S6/s3_stack/vibrations/346/basic.vib.out')),
        Species('S6_chain_63','S/S6/chains/63/vibrations/346/geometry.in.basic', vibs_from('S/S6/chains/63/vibrations/346/basic.vib.out')),
        Species('S5_ring','S/S5/vibrations/322/geometry.in.basic', vibs_from('S/S5/vibrations/322/basic.vib.out')),
        Species('S4_eclipsed', 'S/S4/eclipsed/vibrations/315/geometry.in.basic', vibs_from('S/S4/eclipsed/vibrations/315/basic.vib.out')),
        Species('S4_buckled', 'S/S4/square/vibrations/315/geometry.in.basic', vibs_from('S/S4/square/vibrations/315/basic.vib.out')),
        Species('S3_ring', 'S/S3/cyclic/vibrations/322/geometry.in.basic', vibs_from('S/S3/cyclic/vibrations/322/basic.vib.out')),
        Species('S3_bent', 'S/S3/bent/vibrations/322/geometry.in.basic', vibs_from('S/S3/bent/vibrations/322/basic.vib.out')),
        Species('S2', 'S/S2/vibrations/147/geometry.in', vibs_from('S/S2/vibrations/147/0.001.vib.out'), 'linear', 2)
        ]

    c = ase.db.connect('sulfur.json')
    for species in polytypes:
        atoms = ase.io.read(materials_dir + '/' + species.structure_path, format='aims')
        id = species.id
        c.write(id,atoms, data={'frequencies': species.frequencies})
        
if (__name__ == '__main__'):
    main()
