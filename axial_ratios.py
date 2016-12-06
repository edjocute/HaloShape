#useFOF = True -> use FOF particles
#useFOF = False -> use central subhalos
#solid = True -> use ellipsoidal volume instead of shell

import numpy as np
#import loadhalo
import snapshot
import readhaloHDF5
import ellipsoid
import utils
#import readhaloHDF5

def axial(cat,dir,snapnum,group,nbins,rmin=1e-2,rmax=1.,useFOF=False,solid=False,NR=False,binwidth=0.1):

    subnum=cat.GroupFirstSub[group]
    rvir=cat.Group_R_Crit200[group]
    if NR and not useFOF:
        pos=readhaloHDF5.readhalo(dir+'output/', "snap", snapnum, "POS ",1,0,subnum)
        pos=utils.image(cat.SubhaloPos[subnum],pos,75000)-cat.SubhaloPos[subnum]
    elif NR and useFOF:
        pos=readhaloHDF5.readhalo(dir+'output/', "snap", snapnum, "POS ",1,group,-1)
        pos=utils.image(cat.GroupPos[group],pos,75000)-cat.GroupPos[group]
    elif not NR and useFOF:
        pos=snapshot.loadhalo(dir+"/output",135,group,1,["Coordinates"])
        pos=utils.image(cat.GroupPos[group],pos,75000)-cat.GroupPos[group]
    else: #DEFAULT: not NR and use cantral subhalo only
        pos=snapshot.loadSubhalo(dir+"/output",135,subnum,1,["Coordinates"])
        pos=utils.image(cat.SubhaloPos[subnum],pos,75000)-cat.SubhaloPos[subnum]
    pos/=rvir

    q,s=np.zeros(nbins),np.zeros(nbins)
    n=np.zeros(nbins)
    axes=np.zeros((nbins,3,3))
    logr=np.linspace(np.log10(rmin),np.log10(rmax),nbins)
    rin=10**(logr-binwidth/2.)
    rout=10**(logr+binwidth/2.)

    #print "Fitting ellipsoids with",nbins,"bins"
    for i in np.arange(nbins):
        #print "Bin : ", i
        if solid:
            q[i],s[i],n[i],axes[i]=ellipsoid.ellipsoidfit(pos,rvir,0,10**logr[i],weighted=True)
        else:
            q[i],s[i],n[i],axes[i]=ellipsoid.ellipsoidfit(pos,rvir,rin[i],rout[i])
    return q,s,n,axes

def axial2(group,cat,dir,snapnum,nbins,rmin=1e-2,rmax=1.,useFOF=False,solid=False,NR=False,\
        binwidth=0.1):
    return axial(cat,dir,snapnum,group,nbins,rmin,rmax,useFOF,solid,NR,binwidth)


def func(x,p):
    return p[0]*(np.tanh(p[1]*np.log10(x/p[2]))-1)

def residuals(p,y,x):
    return y - func(x,p)
