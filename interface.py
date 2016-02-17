from tempfile import mkstemp
from shutil import move,copyfile
from os import remove, close
import subprocess

from dndzExplorer import dndzHand
from scipy.integrate import quad
from scipy.interpolate import interp1d

import numpy as np
import sys, os

from mmUtils import Plotter


from mpi4py import MPI

from math import ceil


import sn
from cmbUtils import smartCls

import glob

class CAMB:
    def __init__(self,templateIni,cambCall="./camb",seed=0):
        self._template = "cambInterface_temp_"+str(seed)+"_.ini"
        copyfile(templateIni, self._template)
        self._callpath = cambCall
        
    def setParam(self,paramName,newVal):
        self._replace(self._template,paramName,subst=paramName+"="+str(newVal))

    def call(self,suppress=True):
        if suppress:
            with open(os.devnull, "w") as f:
                subprocess.call([self._callpath,self._template],stdout=f)
        else:
                subprocess.call([self._callpath,self._template])

    def done(self):
        remove(self._template)

    def getCls(self):
        raise NotImplementedError

    def _replace(self,file_path, pattern, subst):

        fh, abs_path = mkstemp()
        with open(abs_path,'w') as new_file:
            with open(file_path) as old_file:
                for line in old_file:
                    if pattern in line:
                        line = subst+"\n"
                    new_file.write(line)
        close(fh)
        remove(file_path)
        move(abs_path, file_path)




# Some hard-coded parameters


MPI_RES_TAG = 88

# Where am I in the MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()

print "Started job ", rank


# The stuff to do

brange = np.arange(0.7*7.810,1.3*7.810,0.1)
npoints = len(brange)



num_each = int(ceil(float(npoints) / float(numcores)))
myBs = brange[rank*num_each:(rank+1)*num_each]
numBs = len(myBs)

print "Gotta do ", numBs

        



a = 0.531
#b = 7.810
c = 0.517
A = 0.688       
zlist = np.arange(0.,3.5,0.001)

pl = Plotter()

myCls = []

for b in myBs:


    myCamb = CAMB("/astro/u/msyriac/software/cambRed/params_testbed.ini",seed=rank)
    myCamb.setParam("accuracy_boost",1)
    myCamb.setParam("redshift_file(2)","../../repos/cmb-lensing-projections/data/dndz/cmass_dndz.csv")

    Hand = dndzHand(A,a,b,c)
    norm = quad(Hand.dndz,0.,3.5)[0]
    #print norm

    distro = Hand.dndz(zlist)/norm
    savemat = np.vstack((zlist,distro)).transpose()
    redFile = "temp_dndz_"+str(rank)+"_.csv"
    np.savetxt(redFile,savemat,delimiter=' ')

    pl.add(zlist,distro)

    outRoot = "dndzTest"+str(rank)
    myCamb.setParam("output_root",outRoot)
    myCamb.setParam("redshift_file(1)",redFile)
    myCamb.call(suppress=False)
    remove(redFile)

    theoryFile = outRoot+"_scalCovCls.dat"
    colnum = sn.getColNum(0,1,2)
    norm="lsq"
    transpower=[2.,0.5]
    readCls = smartCls(theoryFile)
    Cls = np.array(readCls.getCol(colnum=colnum,norm=norm,transpower=transpower))
    ells = readCls.ells

    #print Cls

    myCls.append(Cls)

    test = outRoot+'*.dat'
    r = glob.glob(test)
    for i in r:
       os.remove(i)
    os.remove(outRoot+"_params.ini")
       
    myCamb.done()

    

myCls = np.array(myCls,dtype=np.float64)
print myCls.shape
rec_exp = np.empty(shape=(1,1),dtype=np.float64)
send_exp = np.empty(shape=(1,1),dtype=np.float64)
send_exp[0,0] = numBs

# Send and collect the results

    

if rank!=0:
    MPI.COMM_WORLD.Send(myCls,dest=0,tag=MPI_RES_TAG)
    MPI.COMM_WORLD.Send(send_exp,dest=0,tag=MPI_RES_TAG*2)


else:

    print "postprocessing..."
    for index in range(1,numcores):
        

        MPI.COMM_WORLD.Recv(rec_exp,source=index,tag=MPI_RES_TAG*2)
        num_exp = rec_exp[0,0]
        if num_exp<1: continue
        
        sCls = np.empty((num_exp,len(ells)), dtype=np.float64)
        MPI.COMM_WORLD.Recv(sCls,source=index,tag=MPI_RES_TAG)
        myCls = np.vstack((myCls,sCls))

        pl.add

        print index+1 , " / " , numcores
        

    print myCls
    print myCls.shape

    pl = Plotter()

    # for b,Cls in zip(brange,myCls):
    #     pl.add(ells,ells*Cls,label=str(b))

    # pl.legendOn(labsize=8)

    pl.add(brange,myCls[:,500])
    pl.done("brange500.png")

    pl = Plotter()
    pl.add(brange,myCls[:,100])
    pl.done("brange100.png")

    pl = Plotter()
    pl.add(brange,myCls[:,1000])
    pl.done("brange1000.png")


    pl = Plotter()
    pl.add(brange,myCls[:,2000])
    pl.done("brange2000.png")
