from ann import ATOMModel, Ann
from ase.io import read
from ase.optimize.fire import FIRE

p1 = read('au55.xyz',index=0,format='xyz')
model = ATOMModel(restore='atom_model', n=55)    
p1.set_calculator(Ann(atoms=p1, ann_model=model))
opt= FIRE(atoms=p1, maxmove = 1.0, dt = 0.2, dtmax = 1.0, logfile='geo_opt.dat', trajectory='geo_opt.traj')
opt.run(fmax=0.05, steps=1000)
