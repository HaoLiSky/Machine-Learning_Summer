import sys
from ase.io import read
from ase.optimize.fire import FIRE
from tsase.calculators.lmplib import LAMMPSlib
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
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
      if line.startswith('#'):
         continue
      fields = line.partition('#')[0].split('=')
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
    dyn.attach(md_log, interval=md_interval)
    dyn.run(md_steps)
    return atoms_md, e_log
    #for debug
    #dump_atoms(atoms_md, 'md.xyz')
    #log_e = open('md.log', 'w')
    #i = 0
    #for e in e_log:
    #    log_e.write("%d %15.6f %15.6f\n" %(i, e[0], e[1]))
    #    i+=1
    #log_e.close()


def opt_structure(atoms, interval, opt_calculator, optimizer):
    atoms.set_calculator(opt_calculator)
    opt= optimizer(atoms=atoms, maxmove = 1.0, dt = 0.2, dtmax = 1.0, logfile='geo_opt.dat', trajectory='geo_opt.traj')

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
    opt.run(fmax=0.0001, steps=4000)
    return opt_traj, e_log, forces

def main():
    arg = sys.argv
    paras = readinputs(arg[1])

    p1 = read(paras['input_file'],index=0,format=paras['format'])
    p1.set_cell([[80,0,0],[0,80,0],[0,0,80]],scale_atoms=False)
    p1.set_pbc(True)

    #set calculator
    #TODO: generate calculator with user-given parameters (user-friendly but less flexible)
    #or keep both features
    cmds = ["pair_style eam/fs",
            "pair_coeff * * PdAu_sc.eam.fs Au"]
    lammps = LAMMPSlib(lmpcmds = cmds, 
                       atoms=p1,
                       lammps_header=['units metal',
                                      'atom_style charge',
                                      'atom_modify map array sort 0 0.0'])
    #end of calculator

    p1.set_calculator(lammps)
    md_structures, e_log = run_md(atoms=p1, 
                                  md_temperature = float(paras['md_temperature']) * kB,
                                  md_step_size = float(paras['md_step_size']),
                                  md_steps= int(float(paras['md_steps'])),
                                  md_interval= int(float(paras['md_interval']))
                                  )
    logfile = open(paras['output_file'], 'w')
    numb_structure = int(float(paras['record_frequency']))
    for atoms in md_structures:
        opt_traj, e_log, fs = opt_structure(atoms=atoms, interval=1, opt_calculator=lammps, optimizer=FIRE)
        if len(opt_traj) < numb_structure:
           interval = 1
        else:
           interval = int(len(opt_traj)/numb_structure)
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
    logfile.close()
if __name__ == '__main__':
    main()
