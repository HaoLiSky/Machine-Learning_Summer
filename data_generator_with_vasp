import sys
from ase.io import read
from ase.optimize.fire import FIRE
from ase.calculators.vasp import Vasp
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase.units import kB, fs

"""
Data generator:
Run molecular dynamic simulations to get high-E structures;
Then optimize high-E structures to local minimum and record certain number of structures (both energy and force) during optimization.
Example of input file:
===================================
input_file       =  au55.xyz             # initial structure
format           =  xyz                  # file format
output_file      =  data_structures.xyz  # currently in xyz format
md_temperature   =  1000                 # unit Kelvin
md_step_size     =  2                    # unit fs
md_steps         =  10000                # maximum md steps
md_interval      =  100                  # frequecy to record md structures
record_frequency =  10                   # frequency to record structures from optimization trajectory
===================================
"""

def readinputs(filename):
    f=open(filename, 'r')
    parameters = {}
    lines=f.readlines()
    for line in lines:
      if line.startswith('#') or len(line.split()) == 0:
         continue
      fields = line.partition('#')[0].split('=')
      if fields[1].replace("\n","").strip() in  ['True', 'true']:
         parameters[fields[0].strip()] = True
         continue
      if fields[1].replace("\n","").strip() in ['False', 'false']:
         parameters[fields[0].strip()] = False
         continue
      parameters[fields[0].strip()]=fields[1].replace("\n","").strip()
    return parameters

def run_md(atoms, md_temperature = 1000*kB, md_step_size = 1*fs, md_steps=100, md_interval=100, friction=0.02):
    natoms=atoms.get_number_of_atoms()
    #atoms.set_calculator(self.opt_calculator)
    atoms_md = []
    e_log = []
    atoms_md.append(atoms.copy())
    MaxwellBoltzmannDistribution(atoms=atoms, temp=md_temperature)
    dyn = Langevin(atoms, md_step_size, md_temperature, friction)
    def md_log(atoms=atoms):
        atoms_md.append(atoms.copy())
        epot=atoms.get_potential_energy()
        ekin=atoms.get_kinetic_energy()
        temp = ekin / (1.5 * kB * natoms)
        e_log.append([epot, ekin, temp])
    traj = Trajectory('Au_md.traj', 'w', atoms)      # Output MD trajectory before it is treatment
    dyn.attach(traj.write, interval=10)
    dyn.attach(md_log, interval=md_interval)
    dyn.run(md_steps)
    return atoms_md, e_log

def opt_structure(atoms, interval, opt_calculator, optimizer):
    atoms.set_calculator(opt_calculator)
    opt= optimizer(atoms=atoms, maxmove = 1.0, dt = 0.2, dtmax = 1.0, logfile=None, trajectory=None)

    opt_traj = []
    e_log = []
    forces = []
    def log_traj(atoms=atoms):
        opt_traj.append(atoms.copy())
        epot=atoms.get_potential_energy()
        f = atoms.get_forces()
        e_log.append(epot)
        forces.append(f)
    opt.attach(log_traj, interval=1)
    opt.run(fmax=0.03, steps=300)               #maxmium of force for vasp?  The fmax TAG should be transferred as a parameter?
    return opt_traj, e_log, forces

def main():
    arg = sys.argv
    paras = readinputs(arg[1])

    p1 = read('POSCAR')

    #set calculator
    #TODO: generate calculator with user-given parameters (user-friendly but less flexible)
    #or keep both features
    vasp1  = Vasp(xc='PBE',       # for MD, coarse prec
                 prec='Med',
                 ediff=1e-3,
                 lplane=True,
                 npar=4,
                 lwave=True,
                 lcharg=True,
                 encut=200,
                 algo='Fast',
                 lreal='Auto',
                 istart=1,
                 icharg=1,
                 )
    vasp2  = Vasp(xc='PBE',     # for scf, prec normal prec
		 ediff=float(paras['EDIFF']),
		 lplane=paras['LPLANE'],
                 npar=int(paras['NPAR']),
		 lwave=paras['LWAVE'],
	         lcharg=paras['LCHARG'],
                 encut=float(paras['ENCUT']),
		 algo=paras['ALGO'],
                 lreal=paras['LREAL'],
                 istart=1,
                 icharg=1
                 )
    #end of calculator

    p1.set_calculator(vasp1)
    md_structures, e_log = run_md(atoms=p1, 
                                  md_temperature = float(paras['md_temperature']) * kB,
                                  md_step_size = float(paras['md_step_size']),
                                  md_steps= int(paras['md_steps']),
                                  md_interval= int(paras['md_interval'])
                                  )
    logfile = open(str(paras['output_file']), 'a+')
    numb_structure = int(paras['record_frequency'])
    for atoms in md_structures:
        opt_traj, e_log, fs = opt_structure(atoms=atoms, interval=1, opt_calculator=vasp2, optimizer=FIRE)
        if len(opt_traj) < numb_structure:
           interval = 1
        else:
           interval = numb_structure                  #Chaged by Wanglai Cen at 16-02-2018
        for i in range (len(opt_traj)):
            if i % interval == 0 or i == len(opt_traj)-1:
               if i == len(opt_traj)-1:
                  label = 'local minimum'
               else:
                  label = 'nonstable'
               logfile.write("%d\n"%(len(atoms)))
               logfile.write("%s  %s: %15.6f\n"%(label, 'Energy',e_log[i]))
               for atom in opt_traj[i]:
                   j = atom.index
                   logfile.write("%s %15.6f %15.6f %15.6f %15.6f %15.6f %15.6f\n"%(atom.symbol, atom.x,
                                 atom.y, atom.z, fs[i][j][0], fs[i][j][1], fs[i][j][2]))
        logfile.flush()                            #Added by Wanglai Cen at 16-02-2018
    logfile.close()
if __name__ == '__main__':
    main()
