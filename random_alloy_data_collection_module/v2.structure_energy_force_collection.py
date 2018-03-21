#!/usr/bin/python

import os, sys

interval = 2 # changing the number of data set want to get

pwd = os.getcwd()
os.chdir(pwd)

os.mkdir('output')
out_energy = open('output/result_force_energy', 'w')

dirs = os.listdir(pwd)
dirs.sort()

for dir in dirs:
   try:
      os.chdir(os.path.join(pwd,dir))
      sys_total = os.popen('grep "F=" OSZICAR').readlines()
      sys_st = open('XDATCAR','r').readlines()
      for i in range(0,len(sys_total),interval): # force and energy gathering
          enf = sys_total[i].split()[0:5:2] # Sequencee - number : force : energy
          num = str(dir)+'_'+str(enf[0])
          out_energy.write("%4s %20s %20s\n"%(num, enf[1], enf[2]))

          st = sys_st[0:7]
          #st.append('Selective Dynamics\n')
          st.append('Direct\n')
          st.extend(sys_st[8+i*37:8+36*(i+1)])
          out_st = open('../output/POSCAR_'+num,'w')
          for j in st:
              out_st.write(j)
          out_st.close()

   except:
      pass

   os.chdir(pwd)

out_energy.close()
