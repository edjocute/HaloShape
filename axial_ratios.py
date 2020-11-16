#useFOF = True -> use FOF particles
#useFOF = False -> use central subhalos
#solid = True -> use ellipsoidal volume instead of shell

import power as power
import numpy as np
#import loadhalo
import snapshot
import readsubfHDF5
import readsnapHDF5
import ellipsoid_numba as ellipsoid
import utils
#import readhaloHDF5

class AxialRatio:
    def __init__(self,catdir,snapnum,nbins,rmin=1e-2,rmax=1.,useSubhaloID=False,\
            useFOF=False, useStellarhalfmassRad=None, useReduced=False, radinkpc=False,\
            NR=False, binwidth=0.1, debug=False, useSubhaloes=False, testconvergence=False,):

        assert type(nbins) is int and nbins >= 0, 'parameter nbins must be int'

        # Set main parameters
        if catdir.find('/output') < 0:
            self.snapdir = catdir+'/output/'
        else:
            self.snapdir = catdir
        self.snapnum = snapnum
        self.useSubhaloID = useSubhaloID
        self.useStellarhalfmassRad = False
        self.radinkpc = radinkpc
        print '\n\tAxialRatio: ',self.snapdir

        # Read snapshot header for boxsize
        snapstr=str(snapnum).zfill(3)
        self.header=readsnapHDF5.snapshot_header(self.snapdir+'/snapdir_'+snapstr+'/snap_'+snapstr)
        self.boxsize = self.header.boxsize
        print '\tAxialRatio: Boxsize =', self.boxsize

        self.parttypes = [1]
        if self.header.cooling == 1:
            self.parttypes.append(0)
        if self.header.sfr == 1:
            self.parttypes.append(4)
        print '\tAvailable parttypes =',self.parttypes

        # Set other parameters
        self.setparams(nbins,rmin,rmax,useFOF,useStellarhalfmassRad,useReduced,NR,\
                binwidth, debug, testconvergence);

        # Read SUBFIND catalogue
        keysel = ["Group_R_Crit200","GroupFirstSub", "Group_M_Crit200","GroupPos"]
        if useSubhaloes is True:
            print 'adding keysel for subhaloes'
            keysel += ["GroupNsubs","SubhaloPos","SubhaloMass"]
        if useSubhaloID is True:
            print 'adding keysel for using SubhaloIDs'
            assert useStellarhalfmassRad is True
            assert useFOF is False
        if useStellarhalfmassRad is True:
            assert self.header.sfr == 1, 'Simulation does not have stars'
            print 'adding keysel for StellarhalfmassRad'
            keysel += ["SubhaloPos","SubhaloMassInRadType","SubhaloHalfmassRadType"]
        self.cat=readsubfHDF5.subfind_catalog(self.snapdir,snapnum, keysel=keysel)

    def setparams(self,nbins,rmin=1e-2,rmax=1., useFOF=False,\
            useStellarhalfmassRad=None, useReduced=False, NR=False, binwidth=0.1, debug=False, \
            testconvergence=False):

        # Choice of radii depends on parameter nbins:
        if nbins == 1:
            print '\n\tnbin=1: calculating for single r', rmin
            assert type(rmin) is float, '\trmin must be float for nbin=1!'
            self.logr=np.array(np.log10([rmin]))
            self.nbins=1
            print '\n\tCalculating for single r', nbins
        elif nbins == 0: #specify radii
            print '\n\tnbins=0 specified. Using the specified radii', rmin
            assert type(rmin) is list, '\trmin must be a list for nbin=0!'
            self.logr=np.log10(rmin)
            self.nbins=len(rmin)
        elif nbins > 1:
            print '\n\tAxialRatio: Number of bins specified. Using rmin,rmax=',rmin,rmax,'with nbins=', nbins
            self.logr=np.linspace(np.log10(rmin),np.log10(rmax),nbins)
            self.nbins=nbins
        else:
            raise Exception('Invalid specification of nbins. Must be integer > 0')

        self.debug = debug
        self.useFOF = useFOF
        if useStellarhalfmassRad is not None:
            self.useStellarhalfmassRad = useStellarhalfmassRad
        if self.useStellarhalfmassRad is True:
            assert self.header.sfr == 1, 'Simulation does not have stars'
        if self.useSubhaloID is True:
            assert self.useStellarhalfmassRad is True, 'Cannot use Group_M_Crit200 if using SubhaloID'
        self.useReduced = useReduced
        self.NR=NR
        self.binwidth=binwidth
        self.rin  = 10**(self.logr-self.binwidth/2.)
        self.rout = 10**(self.logr+self.binwidth/2.)

        self.testconvergence = testconvergence

        if self.useSubhaloID is True:
            print '\tAxialRatio Note: Using SubhaloIDs!'
        else:
            print '\tAxialRatio Note: Using GroupIDs!'
        if self.useStellarhalfmassRad is True:
            print '\tAxialRatio Note: Calculating shapes wrt Stellar half-mass radii!'
        else:
            print '\tAxialRatio Note: Calculating shapes wrt Group_M_Crit200!'
        if self.useFOF is True:
            print '\tAxialRatio Note: Using all particles in FOF group!'
        else:
            print '\tAxialRatio Note: Using only particles in subhalo only!'
        if self.useReduced is True:
            print '\tAxialRatio Note: Using ellipsoidal volumes with reduced shape tensor!'
        else:
            print '\tAxialRatio Note: Using ellipsoidal shells with unweighted shape tensor!'
        if self.testconvergence:
            print '\tAxialRatio Note: Testing convergence by using {}% of particles'.format(self.testconvergence*100)

        print '\tAxialRatio Note: Specified radii:', 10**self.logr
        print '\tAxialRatio Note: Log. bin width = ',self.binwidth
        print '\tAxialRatio Note: R_inner = ',self.rin
        print '\tAxialRatio Note: R_outer = ',self.rout



    def readhalo(self,groupid,parttype=1):
        assert parttype in self.parttypes,'Specified parttype is not available in this run'
        snapdir=self.snapdir
        snapnum=self.snapnum
        cat=self.cat
        centre = cat.GroupPos[groupid]
        if self.useFOF:
            pos = snapshot.loadHalo(snapdir,snapnum,groupid,parttype,["Coordinates"])
            if parttype != 1:
                mass = snapshot.loadHalo(snapdir,snapnum,groupid,parttype,["Masses"])
        else: #DEFAULT: not NR and use central subhalo only
            subnum = cat.GroupFirstSub[groupid]
            pos = snapshot.loadSubhalo(snapdir,snapnum,subnum,parttype,["Coordinates"])
            if parttype != 1:
                mass = snapshot.loadSubhalo(snapdir,snapnum,subnum,parttype,["Masses"])

        npart = len(pos)
        try:
            pos = utils.image(pos-centre,None,self.boxsize)
        except:
            print 'readhalo failed:',groupid, pos.__class__, centre
            return -1,None
        assert npart == len(pos),'Readhalo error! pos={}'.format(pos)

        if parttype == 1:
            return pos,None
        else:
            return pos,mass

    def readsubhalo(self,subid,parttype=1):
        snapdir=self.snapdir
        snapnum=self.snapnum
        cat=self.cat
        centre = cat.SubhaloPos[subid]
        if self.useFOF:
            raise Exception, 'readsubhalo is incompatible with option useFOF'
        else: #use central subhalo only
            pos = snapshot.loadSubhalo(snapdir,snapnum,subid,parttype,["Coordinates"])
            if parttype != 1:
                mass = snapshot.loadSubhalo(snapdir,snapnum,subid,parttype,["Masses"])

        if type(pos) == dict:
            assert pos['count'] == 0
            #print 'No particles of type {} in subhalo {}'.format(parttype,subid)
            return 0,None

        npart = len(pos)
        try:
            pos = utils.image(pos-centre,None,self.boxsize)
        except:
            print 'readsubhalo failed:',subid, pos.__class__, centre
            return -1,None
        #assert npart == len(pos),'Readhalo error! pos={}'.format(pos)

        if parttype == 1:
            return pos,None
        else:
            return pos,mass

    def getshape(self,objid,parttype):
        logr=self.logr
        nbins=self.nbins

        # Check if we're specifying groupids or subids
        if self.useSubhaloID:
            readpart = self.readsubhalo
            subid = objid
        else:
            readpart = self.readhalo
            groupid = objid
            subid = self.cat.GroupFirstSub[groupid]

        # Normalize by R200 or Stellar half mass radius.
        if self.useStellarhalfmassRad:
            rad = self.cat.SubhaloHalfmassRadType[subid,4]
        else:
            assert not self.useSubhaloID
            rad = self.cat.Group_R_Crit200[groupid]

        # Read positions (for all) and masses (for stars and gas)
        #if parttype == 1:
        #    pos = readpart(objid,parttype)
        #else:
        pos,mass = readpart(objid,parttype)
        if type(pos) is int:
            assert pos in [0,-1]
            if pos == -1:
                print 'getshape: error reading particles '
                return None
            elif pos == 0:
                return None

        if self.radinkpc is True:
            pos/=self.header.hubble
        else:
            # Normalize positions by chosen radius
            pos/=rad

        if self.testconvergence is not False:
            assert type(self.testconvergence) is float

            sel = np.zeros(len(pos),dtype=bool)
            sel[:int(len(pos)*self.testconvergence)] = True
            np.random.shuffle(sel)
            pos = pos[sel]
            if parttype != 1:
                mass = mass[sel]

        q,s=np.zeros(nbins),np.zeros(nbins)
        n=np.zeros((nbins,2),dtype=int)
        axes=np.zeros((nbins,3,3))

        #print "Fitting ellipsoids with",nbins,"bins"
        for i in np.arange(nbins):
            if self.useReduced:
                #if parttype==1:
                    #tempout=ellipsoid.ellipsoidfit(pos,rad,0,10**logr[i],weighted=True)
                    #if self.debug: print 'Using reduced tensor for DM'
                #else:
                    tempout=ellipsoid.ellipsoidfit(pos,rad,0,10**logr[i],mass=mass,weighted=True)
                    #if self.debug: print 'Using reduced tensor for stars'
            else:
                #if parttype==1:
                #    tempout=ellipsoid.ellipsoidfit(pos,rad,self.rin[i],self.rout[i])
                #    if self.debug: print 'Using normal tensor for DM'
                #else:
                    tempout=ellipsoid.ellipsoidfit(pos,rad,self.rin[i],self.rout[i],mass=mass)
                 #   if self.debug: print 'Using normal tensor for stars'
            q[i],s[i],n[i,0],axes[i],n[i,1]=tempout
        return q,s,n,axes

    def DM(self,group):
        return self.getshape(group,1)

    def Stars(self,group):
        return self.getshape(group,4)

    def Gas(self,group):
        return self.getshape(group,0)

    def getPowerK(self,objid,normrad=True):
        # Check if we're specifying groupids or subids
        if self.useSubhaloID:
            readpart = self.readsubhalo
            subid = objid
        else:
            readpart = self.readhalo
            groupid = objid
            subid = self.cat.GroupFirstSub[groupid]

        # Normalize by R200 or Stellar half mass radius.
        if self.useStellarhalfmassRad:
            rad = self.cat.SubhaloHalfmassRadType[subid,4]
        else:
            assert not self.useSubhaloID
            rad = self.cat.Group_R_Crit200[groupid]

        r = 10**self.logr
        if normrad == True:
            r *= rad

        # Read positions (for all) and masses (for stars and gas)
        lpos, lmass = [],[]
        for parttype in self.parttypes:
            pos,mass = readpart(objid,parttype)
            if type(pos) is not int:
                if parttype == 1:
                    assert mass is None
                    mass = np.ones(len(pos)) * self.header.massarr[1]
                lpos.append(pos)
                lmass.append(mass)
        if lpos is not []:
            pos = np.vstack((lpos))
            mass = np.hstack((lmass))
            return power.powerrad(pos, mass, r, self.header.redshift, self.header.omega0, self.header.omegaL)
        else:
            return -1

    def getShapefromSubhalo(self,group,parttype=1,minmass=10**9):
        logr=self.logr
        nbins=self.nbins
        rvir=self.cat.Group_R_Crit200[group]
        pos,mass=self.getSubhaloes(group,parttype,minmass)
        print group,len(mass)
        print np.count_nonzero(np.linalg.norm(pos,axis=1) < rvir)

        q,s=np.zeros(nbins),np.zeros(nbins)
        n=np.zeros((nbins,2),dtype=int)
        axes=np.zeros((nbins,3,3))

        if pos is None:
            print 'getshape: pos is None'
            return None
        pos/=rvir

        for i in np.arange(nbins):
            if self.solid:
                tempout=ellipsoid.ellipsoidfit(pos,rvir,0,10**logr[i],mass=None,weighted=True,verbose=True)
            else:
                tempout=ellipsoid.ellipsoidfit(pos,rvir,0,10**logr[i],mass=None,verbose=True)
            q[i],s[i],n[i,0],axes[i],n[i,1]=tempout
        return q,s,n,axes

    def getSubhaloes(self,group,parttype,minmass):
        s=slice(self.cat.GroupFirstSub[group],self.cat.GroupFirstSub[group]+self.cat.GroupNsubs[group])

        pos=self.cat.SubhaloPos[s]
        mass=self.cat.SubhaloMass[s]

        centre = self.cat.GroupPos[group]
        npart = len(pos)
        try:
            pos = utils.image(pos-centre,None,self.boxsize)
        except:
            print 'getSubhaloes failed:',groupid, pos.__class__, centre
            return None

        return pos[mass>minmass/1e10],mass[mass>minmass/1e10]


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

'''
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
'''


if __name__ == "__main__":
    import tables
    mass=[12,12.01]
    hubble=0.6774
    fbase = "/n/hernquistfs3/IllustrisTNG/Runs/"

    fname = "L75n910TNG_DM"
    a=AxialRatio(fbase+fname,99,0,[0.15,1.])
    groupmass=a.cat.Group_M_Crit200/hubble*1e10
    groups=np.nonzero((groupmass>10**mass[0]) & (groupmass<10**mass[1]))[0]

    with tables.open_file('/n/hernquistfs3/kchua/Shape/101-AllRedshifts/'+fname+'/'+fname+'_DMShape_099.hdf5','r') as f:
        out = [a.DM(t) for t in groups]
        assert np.allclose(np.array([t[0] for t in out]), f.root.shell.q[:][groups][:,[1,-1]])
        assert np.allclose(np.array([t[1] for t in out]), f.root.shell.s[:][groups][:,[1,-1]])

        a.setparams(0,[0.15,1.],useReduced=True)
        out = [a.DM(t) for t in groups]
        assert np.allclose(np.array([t[0] for t in out]), f.root.reduced.q[:][groups][:,[1,-1]])
        assert np.allclose(np.array([t[1] for t in out]), f.root.reduced.s[:][groups][:,[1,-1]])

        try:
            a.setparams(0,[0.15,1.],useStellarhalfmassRad=True)
        except AssertionError:
            print '\n\tDMO Tests passed!'
    print '\t***************************************************************************'

    fname = "L75n910TNG"
    a=AxialRatio(fbase+fname,99,0,[0.15,1.])
    groupmass=a.cat.Group_M_Crit200/hubble*1e10
    groups=np.nonzero((groupmass>10**mass[0]) & (groupmass<10**mass[1]))[0]

    with tables.open_file('/n/hernquistfs3/kchua/Shape/101-AllRedshifts/'+fname+'/'+fname+'_DMShape_099.hdf5','r') as f:
        out = [a.DM(t) for t in groups]
        assert np.allclose(np.array([t[0] for t in out]), f.root.shell.q[:][groups][:,[1,-1]])
        assert np.allclose(np.array([t[1] for t in out]), f.root.shell.s[:][groups][:,[1,-1]])

        a.setparams(0,[0.15,1.],useReduced=True)
        out = [a.DM(t) for t in groups]
        assert np.allclose(np.array([t[0] for t in out]), f.root.reduced.q[:][groups][:,[1,-1]])
        assert np.allclose(np.array([t[1] for t in out]), f.root.reduced.s[:][groups][:,[1,-1]])

        a.setparams(0,[0.15,1.],useStellarhalfmassRad=True)

    print '\n\tFP Tests passed!'
    print '\t***************************************************************************'

    mass=[11,11.01]
    a=AxialRatio(fbase+fname,99,0,[1.,2.],useStellarhalfmassRad=True,useSubhaloID=True, debug=True)
    stellarmass=a.cat.SubhaloMassInRadType[:,4]/hubble*1e10
    subs=np.nonzero((stellarmass>10**mass[0]) & (stellarmass<10**mass[1]))[0]
    print 'nsubs=',len(subs)

    out = [a.DM(t) for t in subs]
    out = [a.Stars(t) for t in subs]

    a.setparams(0,[1.,2.],useReduced=True,debug=True)
    out = [a.DM(t) for t in subs]
    out = [a.Stars(t) for t in subs]

    try:
        a.setparams(0,[1.,2.],useStellarhalfmassRad=False)
    except AssertionError:
        print '\n\tStellar Tests passed!'
