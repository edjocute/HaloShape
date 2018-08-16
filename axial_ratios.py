#useFOF = True -> use FOF particles
#useFOF = False -> use central subhalos
#solid = True -> use ellipsoidal volume instead of shell

import numpy as np
#import loadhalo
import snapshot
import readsubfHDF5
import ellipsoid
import utils
#import readhaloHDF5

class AxialRatio:
    def __init__(self,catdir,snapnum,nbins,rmin=1e-2,rmax=1.,useFOF=False,solid=False,NR=False,binwidth=0.1):
            #raise Exception('Invalid specification of nbins. Must be float, numpy array or int')
        #assert nbins.__class__  in [int, float, np.ndarray]
        assert nbins >= 0

        if catdir.find('/output')<0:
            self.snapdir=catdir+'/output/'
        else:
            self.snapdir=catdir
        print self.snapdir
        self.cat=readsubfHDF5.subfind_catalog(self.snapdir,snapnum,\
                keysel=["Group_R_Crit200","GroupFirstSub", "SubhaloPos","Group_M_Crit200","GroupPos","SubhaloMass"])

        self.setparams(catdir,snapnum,nbins,rmin,rmax,useFOF,solid,NR,binwidth);

    def setparams(self,catdir,snapnum,nbins,rmin=1e-2,rmax=1.,useFOF=False,solid=False,NR=False,binwidth=0.1):
        #if nbins.__class__ == float: #single radius calc
        #    assert nbins <= 1.
        if nbins == 1:
            print 'nbin=1: calculating for single r', rmin
            assert rmin.__class__ == float
            self.logr=np.array(np.log10(rmin))
            self.nbins=1
            print 'Calculating for single r', nbins
        #elif nbins.__class__ == np.ndarray: #specify radii
        elif nbins == 0: #specify radii
            print 'nbins=-1 specified. Using the specified radii',rmin
            self.logr=np.log10(rmin)
            self.nbins=len(rmin)
        #elif nbins.__class__ == int: #use log interval
        elif nbins > 1:
            self.logr=np.linspace(np.log10(rmin),np.log10(rmax),nbins)
            self.nbins=nbins
            print 'Number of bins specified. Using rmin,rmax=',rmin,rmax,'with nbins=', nbins
        else:
            raise Exception('Invalid specification of nbins. Must be float, numpy array or int')

        self.snapnum=snapnum
        self.useFOF=useFOF
        self.solid=solid
        self.NR=NR
        self.binwidth=binwidth
        self.rin=10**(self.logr-self.binwidth/2.);self.rout=10**(self.logr+self.binwidth/2.)

        if useFOF==True:
            print 'Note: Using all particles in FOF group!'
        else:
            print 'Note: Using only particles in central subhalo!'
        if solid==True:
            print 'NOte: Using ellipsoidal volumes!'
        else:
            print 'Note: Using ellipsoidal shells!'

    def readhalo(self,groupid,parttype):
        snapdir=self.snapdir
        snapnum=self.snapnum
        cat=self.cat
        if self.useFOF:
            pos=snapshot.loadhalo(snapdir,snapnum,group,parttype,["Coordinates"])
            pos=utils.image(cat.GroupPos[group],pos,75000)-cat.GroupPos[group]
        else: #DEFAULT: not NR and use central subhalo only
            subnum=cat.GroupFirstSub[groupid]
            pos=snapshot.loadSubhalo(snapdir,snapnum,subnum,parttype,["Coordinates"])
            pos=utils.image(cat.SubhaloPos[subnum],pos,75000)-cat.SubhaloPos[subnum]
        return pos

    def getshape(self,group,parttype):
        logr=self.logr
        nbins=self.nbins
        rvir=self.cat.Group_R_Crit200[group]
        pos=self.readhalo(group,parttype)/rvir

        q,s=np.zeros(nbins),np.zeros(nbins)
        n=np.zeros(nbins,dtype=int)
        axes=np.zeros((nbins,3,3))

        #print "Fitting ellipsoids with",nbins,"bins"
        for i in np.arange(nbins):
            if self.solid:
                if parttype==1:
                    tempout=ellipsoid.ellipsoidfit(pos,rvir,0,10**logr[i],weighted=True)
                else:
                    tempout=ellipsoid.ellipsoidfit(pos,rvir,0,10**logr[i],mass=mass,weighted=True)
            else:
                if parttype==1:
                    tempout=ellipsoid.ellipsoidfit(pos,rvir,self.rin[i],self.rout[i])
                else:
                    tempout=ellipsoid.ellipsoidfit(pos,rvir,self.rin[i],self.rout[i],mass=mass)
            q[i],s[i],n[i],axes[i]=tempout
        return q,s,n,axes

    def DM(self,group):
        return self.getshape(group,1)

    def Stars(self,group):
        return self.getshape(group,4)

    def get2Dshape(self,group,parttype):
        logr=self.logr
        nbins=self.nbins
        rvir=self.cat.Group_R_Crit200[group]
        pos=self.readhalo(group,parttype)[:,1:]/rvir

        q=np.zeros(nbins)
        n=np.zeros(nbins,dtype=int)
        axes=np.zeros((nbins,2,2))

        #print "Fitting ellipsoids with",nbins,"bins"
        for i in np.arange(nbins):
            if self.solid:
                if parttype==1:
                    tempout=ellipsoid.ellipsoidfit2D(pos,rvir,0,10**logr[i])
                else:
                    tempout=ellipsoid.ellipsoidfit2D(pos,rvir,0,10**logr[i],mass=mass)
            else:
                if parttype==1:
                    tempout=ellipsoid.ellipsoidfit2D(pos,rvir,self.rin[i],self.rout[i])
                else:
                    tempout=ellipsoid.ellipsoidfit2D(pos,rvir,self.rin[i],self.rout[i],mass=mass)
            q[i],n[i],axes[i]=tempout
        return q,n,axes

    def SubhaloShape(self,subnum,parttype,useR200=True):
        if self.useFOF==True:
            print ('You have specified useFOF=True, which is incompatible with SubhaloShape.')
            print ('Setting useFOF=False')
            self.useFOF=False
        rhoc=2.7754997454196346e-08*1e10
        rvir= (cat.SubhaloMass[subnum]*1e10/(800./3.*np.pi*rhoc))**(1./3.)/0.704
        pos=self.readhalo(group,parttype)

        q,s=np.zeros(nbins),np.zeros(nbins)
        n=np.zeros(nbins)
        axes=np.zeros((nbins,3,3))

        if useR200==True:
            pos/=rvir
        else:
            rin=self.rin*300; rout=self.rout*300
            rvir=1.

        #print "Fitting ellipsoids with",nbins,"bins"
        for i in np.arange(nbins):
            if solid:
                q[i],s[i],n[i],axes[i]=ellipsoid.ellipsoidfit(pos,rvir,0,10**self.logr[i],weighted=True)
            else:
                q[i],s[i],n[i],axes[i]=ellipsoid.ellipsoidfit(pos,rvir,rin[i],rout[i])
        return q,s,n,axes

def axial(cat,catdir,snapnum,group,nbins,rmin=1e-2,rmax=1.,useFOF=False,solid=False,NR=False,binwidth=0.1):

    subnum=cat.GroupFirstSub[group]
    rvir=cat.Group_R_Crit200[group]

    if catdir.find('/output')<0:
        snapdir=catdir+'/output/'
    else:
        snapdir=catdir

    if NR and not useFOF:
        pos=readhaloHDF5.readhalo(snapdir, "snap", snapnum, "POS ",1,0,subnum)
        pos=utils.image(cat.SubhaloPos[subnum],pos,75000)-cat.SubhaloPos[subnum]
    elif NR and useFOF:
        pos=readhaloHDF5.readhalo(snapdir, "snap", snapnum, "POS ",1,group,-1)
        pos=utils.image(cat.GroupPos[group],pos,75000)-cat.GroupPos[group]
    elif not NR and useFOF:
        pos=snapshot.loadhalo(snapdir,snapnum,group,1,["Coordinates"])
        pos=utils.image(cat.GroupPos[group],pos,75000)-cat.GroupPos[group]
    else: #DEFAULT: not NR and use cantral subhalo only
        pos=snapshot.loadSubhalo(snapdir,snapnum,subnum,1,["Coordinates"])
        pos=utils.image(cat.SubhaloPos[subnum],pos,75000)-cat.SubhaloPos[subnum]
    pos/=rvir

    if nbins.__class__ == float:
        assert nbins <= 1.
        logr=np.array(np.log10(nbins))
        nbins=1
    elif nbins.__class__ == np.ndarray:
        logr=np.log10(nbins)
        nbins=len(logr)
    elif nbins.__class__ == int:
        logr=np.linspace(np.log10(rmin),np.log10(rmax),nbins)
        print 'Number of bins specified. Using rmin,rmax=',rmin,rmax
    else:
        raise Exception('Invalid specification of nbins. Must be float, numpy array or int')

    q,s=np.zeros(nbins),np.zeros(nbins)
    n=np.zeros(nbins)
    axes=np.zeros((nbins,3,3))

    #print "Fitting ellipsoids with",nbins,"bins"
    for i in np.arange(nbins):
        #print "Bin : ", i
        if solid:
            q[i],s[i],n[i],axes[i]=ellipsoid.ellipsoidfit(pos,rvir,0,10**logr[i],weighted=True)
        else:
            rin=10**(logr-binwidth/2.);rout=10**(logr+binwidth/2.)
            q[i],s[i],n[i],axes[i]=ellipsoid.ellipsoidfit(pos,rvir,rin[i],rout[i])
    return q,s,n,axes

def axialstars(cat,catdir,snapnum,group,nbins,rmin=1e-2,rmax=1.,useFOF=False,solid=False,NR=False,binwidth=0.1):

    subnum=cat.GroupFirstSub[group]
    rvir=cat.Group_R_Crit200[group]
    if catdir.find('/output')<0:
        snapdir=catdir+'/output/'
    else:
        snapdir=catdir

    if NR and not useFOF:
        pos=readhaloHDF5.readhalo(dir+'output/', "snap", snapnum, "POS ",4,0,subnum)
        pos=utils.image(cat.SubhaloPos[subnum],pos,75000)-cat.SubhaloPos[subnum]
    elif NR and useFOF:
        pos=readhaloHDF5.readhalo(dir+'output/', "snap", snapnum, "POS ",4,group,-1)
        pos=utils.image(cat.GroupPos[group],pos,75000)-cat.GroupPos[group]
    elif not NR and useFOF:
        pos=snapshot.loadhalo(snapdir,snapnum,group,4,["Coordinates"])
        pos=utils.image(cat.GroupPos[group],pos,75000)-cat.GroupPos[group]
        mass=snapshot.loadhalo(snapdir,snapnum,subnum,4,["Masses"])
    else:
        pos=snapshot.loadSubhalo(snapdir,snapnum,subnum,4,["Coordinates"])
        pos=utils.image(cat.SubhaloPos[subnum],pos,75000)-cat.SubhaloPos[subnum]
        mass=snapshot.loadSubhalo(snapdir,snapnum,subnum,4,["Masses"])
    pos/=rvir

    if nbins.__class__ == float:
        assert nbins <= 1.
        logr=np.array(np.log10(nbins))
        nbins=1
    elif nbins.__class__ == np.ndarray:
        logr=np.log10(nbins)
        nbins=len(logr)
    elif nbins.__class__ == int:
        logr=np.linspace(np.log10(rmin),np.log10(rmax),nbins)
        print 'Number of bins specified. Using rmin,rmax=',rmin,rmax
    else:
        raise Exception('Invalid specification of nbins. Must be float, numpy array or int')

    q,s=np.zeros(nbins),np.zeros(nbins)
    n=np.zeros(nbins)
    axes=np.zeros((nbins,3,3))

    #print "Fitting ellipsoids with",nbins,"bins"
    for i in np.arange(nbins):
        #print "Bin : ", i
        if solid:
            q[i],s[i],n[i],axes[i]=ellipsoid.ellipsoidfit(pos,rvir,0,10**logr[i],mass=mass,weighted=True)
        else:
            rin=10**(logr-binwidth/2.);rout=10**(logr+binwidth/2.)
            q[i],s[i],n[i],axes[i]=ellipsoid.ellipsoidfit(pos,rvir,rin[i],rout[i],mass=mass)
    return q,s,n,axes

def axialsubhalos(cat,catdir,snapnum,subnum,nbins,rmin=1e-2,rmax=1.,solid=False,NR=False,binwidth=0.1,useR200=True):

    rhoc=2.7754997454196346e-08*1e10
    rvir= (cat.SubhaloMass[subnum]*1e10/(800./3.*np.pi*rhoc))**(1./3.)/0.704

    if catdir.find('/output')<0:
        snapdir=catdir+'/output/'
    else:
        snapdir=catdir

    if NR and not useFOF:
        pos=readhaloHDF5.readhalo(snapdir, "snap", snapnum, "POS ",1,0,subnum)
        pos=utils.image(cat.SubhaloPos[subnum],pos,75000)-cat.SubhaloPos[subnum]
    else: #DEFAULT: not NR and use cantral subhalo only
        pos=snapshot.loadSubhalo(snapdir,snapnum,subnum,1,["Coordinates"])
        pos=utils.image(cat.SubhaloPos[subnum],pos,75000)-cat.SubhaloPos[subnum]
    q,s=np.zeros(nbins),np.zeros(nbins)
    n=np.zeros(nbins)
    axes=np.zeros((nbins,3,3))

    logr=np.linspace(np.log10(rmin),np.log10(rmax),nbins)
    rin=10**(logr-binwidth/2.)
    rout=10**(logr+binwidth/2.)
    if useR200==True:
        pos/=rvir
    else:
        rin=rin*300
        rout=rout*300
        rvir=1.

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

def axial2D(cat,catdir,snapnum,group,nbins,useFOF=False,solid=False,NR=False,rmin=1e-2,rmax=1.,binwidth=0.1):

    subnum=cat.GroupFirstSub[group]
    rvir=cat.Group_R_Crit200[group]

    if catdir.find('/output')<0:
        snapdir=catdir+'/output/'
    else:
        snapdir=catdir

    if useFOF:
        pos=snapshot.loadhalo(snapdir,snapnum,group,1,["Coordinates"])
        pos=utils.image(cat.GroupPos[group],pos,75000)-cat.GroupPos[group]
    else: #DEFAULT: not NR and use central subhalo only
        pos=snapshot.loadSubhalo(snapdir,snapnum,subnum,1,["Coordinates"])
        pos=(utils.image(cat.SubhaloPos[subnum],pos,75000)-cat.SubhaloPos[subnum])[:,1:]
    pos/=rvir

    if nbins.__class__ == float:
        assert nbins <= 1.
        logr=np.array(np.log10(nbins))
        nbins=1
    elif nbins.__class__ == np.ndarray:
        logr=np.log10(nbins)
        nbins=len(logr)
    elif nbins.__class__ == int:
        logr=np.linspace(np.log10(rmin),np.log10(rmax),nbins)
        print 'Number of bins specified. Using rmin,rmax=',rmin,rmax
    else:
        raise Exception('Invalid specification of nbins. Must be float, numpy array or int')

    q=np.zeros(nbins)
    n=np.zeros(nbins)
    axes=np.zeros((nbins,2,2))

    #print "Fitting ellipsoids with",nbins,"bins"
    for i in np.arange(nbins):
        #print "Bin : ", i
        if solid:
            q[i],n[i],axes[i]=ellipsoid.ellipsoidfit2D(pos,rvir,0,10**logr[i])
        else:
            rin=10**(logr-binwidth/2.); rout=10**(logr+binwidth/2.)
            q[i],n[i],axes[i]=ellipsoid.ellipsoidfit2D(pos,rvir,rin[i],rout[i])
    return q,n,axes

def func(x,p):
    return p[0]*(np.tanh(p[1]*np.log10(x/p[2]))-1)

def residuals(p,y,x):
    return y - func(x,p)
