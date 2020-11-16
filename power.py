import numpy as np
import conversions as conv
#from numba import jit
import numexpr as ne

def powerrad(pos, mass, rad, z=0, OmegaM=0.27, OmegaL=0.73, h=1.,comoving=True):
    '''
    If h == 1, pos, rad should have units [kpc/h]and mass should have units [10**10 M_sun/h]
    '''

    # Convert to physical if comoving coordinates are provided for pos and rad
    a = 1./(1.+z) if comoving else 1

    if len(pos) == 0:
        return np.zeros_like(rad)

    edges = np.append(0.,rad*a)
    Vol = 4./3.*np.pi*edges[1:]**3

    dist = ne.evaluate("sum(pos**2,axis=1)")
    dist = ne.evaluate("sqrt(dist) * a")
    #dist = np.sqrt(np.sum(pos**2,axis=1))


    hist,edgesout=np.histogram(dist,bins=edges)
    totmass,edgesout=np.histogram(dist,bins=edges,weights=mass)
    hist = hist.cumsum()
    totmass = totmass.cumsum()

    rhocrit = conv.GetRhoCrit(z, OmegaM, OmegaL)*h**2 #Physical critical density
    avgden = totmass/Vol/rhocrit #Normalized density profile
    f = np.sqrt(200)/8. * hist/ np.log(hist) / np.sqrt(avgden) #Power et al. 2003

    return f

def massprofile(grp,cat,fdir,fract,rmin=1e-3,rmax=1.2,nbins=50):
    partmass=0.00052946428432085776
    if cat.filebase.find('910') > 0:
        partmass*=8
        print 'Using partmass for 910'
    elif cat.filebase.find('455') > 0:
        partmass*=64
        print 'Using partmass for 455'
    elif cat.filebase.find('1820') > 0:
        print 'Using partmass for 1820'
    #print "t_relax/t_200 = ", fract

    rvir=cat.Group_R_Crit200[grp]
    #part=snapshot.loadHalo(dir,135,grp,1,fields)
    part=snapshot.loadSubhalo(fdir,135,cat.GroupFirstSub[grp],1,["Coordinates"])
    dist=utils.shortestdist(cat.GroupPos[grp],part,75000)/rvir

    edges=10**np.linspace(np.log10(rmin),np.log10(rmax),nbins)
    #print edges
    hist=np.zeros(len(edges))
    #y=(10**edges[1:]+10**edges[:-1])/2
    #dV=4*np.pi/3*((10**edges[1:])**3  -(10**edges[:-1])**3)
    Vol=4*np.pi/3*(edges*rvir)**3

    hist[1:],edgesout=np.histogram(dist,bins=edges)
    hist[0]=(dist<=edges[0]).sum()
    hist=hist.cumsum() #Number profile
    avgden= hist*partmass /Vol/conv.GetRhoCrit() #Normalized density profile

    f=np.sqrt(200)/8. * hist/ np.log(hist) / np.sqrt(avgden) #Power et al. 2003
    f[np.isinf(f)]=0

    if np.count_nonzero(f>fract) > 1:
        N=np.nonzero(f>fract)[0][0]
        rconv=edges[N]*rvir
    else:
        print '\tnot found 1',cat.Group_M_Crit200[grp]
        return np.nan,-1,-1

    edges1=np.arange(edges[N-1]*rvir,edges[N+1]*rvir,0.002*rconv)/rvir
    Vol1=4*np.pi/3*(edges1*rvir)**3
    hist1=np.zeros(len(edges1))
    hist1[1:],edgesout=np.histogram(dist,bins=edges1)
    hist1[0]=(dist<=edges1[0]).sum()
    hist1=hist1.cumsum() #Cumulative Number profile
    avgden1= hist1*partmass /Vol1/conv.GetRhoCrit() # Normalized average density profile

    f1=np.sqrt(200)/8. * hist1/ np.log(hist1) / np.sqrt(avgden1) #Power et al. 2003
    f1[np.isinf(f1)]=0
    #print edges1*rvir
    if np.count_nonzero(f1>fract) > 0:
        bin1=np.nonzero(f1>fract)[0][0]
        rconv=edges1[np.nonzero(f1>fract)[0][0]]*rvir
        return rconv,hist1[bin1],avgden1[bin1]
    else:
        #print N,rconv,edges1*rvir,f1
        print 'not found 2',cat.Group_M_Crit200[grp]
        return np.nan,-1,-1

    #den=hist[1:]/dV/cat.Group_R_Crit200[grp]**3

if __name__ == "__main__":
    import readsubfHDF5
    import readsnapHDF5
    import readhaloHDF5

    root="/n/mvogelsfs01/mvogelsberger/projects/INSIDM/cosmo/"
    model='Model_A_elastic'#, 'Model_A_inelastic', 'Model_B_elastic', 'Model_B_inelastic']
    level = 13
    num = 24

    path=root+model+'/level_'+str(level)+'/output/'
    base=path
    fname = base + "/snapdir_" + str(num).zfill(3) + "/snap_"+str(num).zfill(3)

    cat = readsubfHDF5.subfind_catalog(path, num)
    rvir = cat.Group_R_Crit200[0]
    head = readsnapHDF5.snapshot_header(fname)

    pos=readhaloHDF5.readhalo(base, "snap", num, "POS ", 1, 0, 0, long_ids=True, double_output=False)
    mass=readhaloHDF5.readhalo(base, "snap", num, "MASS", 1, 0, 0, long_ids=True, double_output=False)

    pos = pos-cat.GroupPos[0]
