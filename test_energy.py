from ann import ATOMModel, Ann
from ase.io import read

p1 = read('au55.xyz',index=0,format='xyz')
model = ATOMModel(restore='atom_model', n=55)    
p1.set_calculator(Ann(atoms=p1, ann_model=model))
print p1.get_potential_energy() 
print p1.get_forces()
