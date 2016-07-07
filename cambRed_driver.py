import os,sys
import numpy as np
from cambRedCall import cambRed
import matplotlib.pyplot as plt

outName = 'test2_lensing'
templateIni = os.environ['CAMBREDNN_DIR']+'params_lensing.ini'
log = True

cRed = cambRed(outName,templateIni,cambRoot=os.environ['CAMBREDNN_DIR'],seed=0)
cRed.call(suppress=False)
Cls = cRed.getCls()
print Cls
n = float(len(Cls[0,:]))
a = int(round(np.sqrt(n),0))
b = int(np.ceil(n/a))
fig, ax = plt.subplots(a,b,figsize=(20,10))
winNum = n
for i in range(1,int(n)):
    winNum = winNum-i
    if winNum == i+1:
        winNum-=1
        break
x = np.arange(float(len(Cls)))
i = 0
for xpos in range(a):
    for ypos in range(b):
        #ax[xpos,ypos].plot(x,Cls[:,i])
        #ax[xpos,ypos].plot(x,data[:,i]*2*np.pi/(x*(x+1.0)))
        
        if i == 0:
            ax[xpos,ypos].plot(x,Cls[:,i]*(4./2./np.pi)/(x*(x+1.0)))
        elif (i!=0) and (i<= winNum):
            ax[xpos,ypos].plot(x,Cls[:,i]*(2./2./np.pi))
        else:
            ax[xpos,ypos].plot(x,Cls[:,i]/2./np.pi*(x*(x+1.0)))
        if log:
            ax[xpos,ypos].set_xscale('log')
            #ax[xpos,ypos].set_yscale('log')
            prefix = '(log)'
        i+=1
        if i==n:
            break
#plt.show()
fileName = outName+prefix+'.png'
plt.savefig(fileName,format='png')
print "Saved figure ",fileName
