#!/usr/bin/python

import numpy as np
from ase.build import *
from ase import *
from ase.constraints import FixAtoms
from ase.io import write
from ase.build import sort
from ase.calculators.vasp import Vasp
import os, sys

calculation_start = sys.argv[2] # without any input calculation is not started
job_name = sys.argv[1] # input job name


def random_alloy_structure(num_structure=10, host='Au', impurity='Pd', a_host=4.0853, a_impurity=3.8907,
                           concentration_impurity=0.5): #TODO list : controlled by output
    random_st = []
    for i in range(num_structure):
       seed=3001223+i*100 # should be gently controlled TODO list : finding best number
       latticeconstant = a_host*(1-concentration_impurity)+a_impurity*concentration_impurity
       model=fcc111(host, size=(3,3,4), a=latticeconstant, vacuum=6.0, orthogonal=False) #TODO list : surface structure controlled by output
       c = FixAtoms(mask=[x >2   for x in model.get_tags()])
       model.set_constraint(c)

       elements=model.get_chemical_symbols()
       num_atom=model.get_number_of_atoms()

       num_impurity=np.round(num_atom*concentration_impurity)
       np.random.seed(seed)

       j=0
       while j < int(num_impurity):
             r=np.random.rand()
             n=int(np.round(r*num_atom))
             if elements[n]==host:
                elements[n]=impurity
                j=j+1

       model.set_chemical_symbols(elements)
       model=sort(model)
       write('POSCAR_'+str(i),model,format='vasp',direct=True)

def main():
    pwd = os.getcwd()
    os.chdir(pwd)
    random_alloy_structure()
    list = os.popen('ls POSCAR_*').read().split('\n')
    list.pop()

    for i in list:
       print "folder name - %s - will be made" % (str(i[7:]))
       try:
          os.mkdir(i[7:])
       except:
          print "folder name - %s - exists" % (str(i[7:]))
       cmd = 'cp exe INCAR KPOINTS POTCAR '+i+' '+i[7:] # TODO LIST: making INCAR, KPOINTS, POTCAR, exe(job script) from the scratch
       os.system(cmd)
       os.chdir(i[7:])
       os.system('cp '+i+' POSCAR')
       try:
          sys.argv[2]
          os.system('sbatch -J '+sys.argv[1]+'_'+i[7:]+'  exe') # could be changed with local environment
       except IndexError:
          print "Folder made / No job submitted"

       os.chdir(pwd)

main()
