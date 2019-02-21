###########################################################################
# Script to calculate halo shapes for a single snapshot.
# Example: run calc.py 135 DM
# Example: run calc.py 130 FP -b /n/ghernquist/Illustris/Runs/L75n910
# Min halo mass = 10^12 M_sun
# #########################################################################


import numpy as np
import readsubfHDF5
import readsnapHDF5
import os,sys
import argparse
import tables
import sharedmem
from functools import partial
#sys.path.append(os.path.join(os.path.dirname(__file__),"../code"))
#sys.path.append('/n/home04/kchua/CODES/Shape/')
import axial_ratios as ar

filters = tables.Filters(complevel=1, complib='zlib', shuffle=True)

#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("snap", type=int, help="snapshot number")
parser.add_argument("t", help="Sim type: FP, DM or NR")
parser.add_argument("-c","--chunksize", type=int, default=100, help="Chunksize for parallel calculation")
parser.add_argument("--minmass", type=float, default=10, help="Min halo mass to be considered")
parser.add_argument("--maxmass", type=float, default=100, help="Min halo mass to be considered")
parser.add_argument("--nthreads", type=int, default=4, help="Number of threads for numexpr")
parser.add_argument("--test", action="store_true", help="Turn on testing mode")
parser.add_argument("--subhaloes", action="store_true", help="Turn on shapes from subhaloes")
parser.add_argument("--TNG", action="store_true", help="Use TNG dir? If this option is selected, supercedes --base")
parser.add_argument("-b","--base", default='/n/ghernquist/Illustris/Runs/', help="Base directory. Default = /n/ghernquist/Illustris/Runs/")
parser.add_argument("-e","--extra", default='', help="addtion to save filename")
parser.add_argument("--binwidth", type=float, default=0.2, help="Log. bin width")
#parser.add_argument("-hu","--hubble", default=0.704, help="Hubble parameter H_0=100h")
args = parser.parse_args()

print '######################################################'
print 'Calculating stellar shapes!\n'
#dir='/n/ghernquist/Illustris/Runs/L75n1820DM/'
fbase = args.base
if (args.t.find('TNG')>0) or (args.TNG):
    print args.t
    print args.TNG
    fbase = "/n/hernquistfs3/IllustrisTNG/Runs/"
fdir = fbase+args.t+'/'
assert os.path.isdir(fdir),fdir+" does not exist!"

snap = args.snap
snapstr=str(snap).zfill(3)
if len(args.extra) == 0:
    fout = args.t+'_StellarShape_'+snapstr+'.hdf5'
else:
    fout = args.t+'_StellarShape_'+snapstr+'_'+args.extra+'.hdf5'


header=readsnapHDF5.snapshot_header(fdir+'/output/snapdir_'+snapstr+'/snap_'+snapstr)
boxsize = header.boxsize
hubble = header.hubble

chunksize = args.chunksize
#ar.ellipsoid.ne.set_num_threads(args.nthreads)
mass=[args.minmass,args.maxmass]
if args.test:
    mass=[10,10.5]
    chunksize=5
binwidth = args.binwidth

print 'Directory: ',fdir
print 'Snapshot: ', snap
print 'Boxsize: ', boxsize
print 'Hubble parameter: ', hubble
print 'Halo mass range for calculation: 10^'+str(mass),'M_sun'
print 'Number of cores for sharedmem: ',sharedmem.cpu_count()
print 'Number of threads for numexpr: ',ar.ellipsoid.numba.config.NUMBA_NUM_THREADS
print ' '

#nbins=18; a=ar.AxialRatio(fdir,snap,nbins,rmin=10**-2.42857143,useSubhaloes=args.subhaloes)
nbins=2
a=ar.AxialRatio(fdir,snap,0,[1.,2.],useStellarhalfmassRad=True, useSubhaloID=True, binwidth=binwidth)
submass=a.cat.SubhaloMassInRadType[:,4]/hubble*1e10
subhalos=np.nonzero((submass>10**mass[0]) & (submass<10**mass[1]))[0]
nsubs=a.cat.nsubs
ngroups=a.cat.ngroups
N=len(subhalos)
print 'Subfind catalogue loaded'
print 'Total no. of subhalos =', nsubs
print 'Total no. of groups =', ngroups
print 'Subhalos to be calculated =', N
print '######################################################'
print ' '

def createandsavefields(f,g,nsubs,nbins,data,subhalos,chunksize,done=False):
    print 'Saving results!'
    r=f.create_carray(g,"r",tables.Float32Col(),(nbins,))
    q=f.create_carray(g,"q",tables.Float32Col(),(nsubs,nbins))
    s=f.create_carray(g,"s",tables.Float32Col(),(nsubs,nbins))
    T=f.create_carray(g,"T",tables.Float32Col(),(nsubs,nbins))
    n=f.create_carray(g,"n",tables.Int32Col(),(nsubs,nbins,2))
    rotmat=f.create_carray(g,"RotMat",tables.Float32Col(),(nsubs,nbins,3,3))

    N = len(subhalos)
    for i in range(0,N,chunksize):
        sl = slice (i, i + chunksize)
        start, end, step = sl.indices(N)
        for m in np.arange(start,end,step):
            grpt = subhalos[m]
            assert data[m/chunksize][0]==i
            if data[m/chunksize][1][m%chunksize] is None:
                #print 'Saving none for subhalo',grpt
                continue
            else:
                q[grpt],s[grpt],n[grpt],rotmat[grpt]=data[m/chunksize][1][m%chunksize]
                T[grpt]=(1-q[grpt]**2)/(1-s[grpt]**2)
                if done:
                    f.root.SubhaloDone[grpt]=1
    ## Be careful when saving triaxiality parameter, when s or q==-1, set T=-1 instead of np.nan
    wherenan = np.where( np.isnan(T[:]))
    for x,y in zip(wherenan[0],wherenan[1]):
        T[x,y] = -1.

    print 'no. failed =', (s[:][subhalos] < 0.).sum(axis = 0), 'out of', len(subhalos)
    return r,q,s,T,n,rotmat

def createandsavepower(f,nsubs,nbins,data,subhalos,chunksize):
    k = f.create_carray("/","k",tables.Float32Col(),(nsubs,nbins))
    N = len(subhalos)
    for i in range(0,N,chunksize):
        sl = slice (i, i + chunksize)
        start, end, step = sl.indices(N)
        for m in np.arange(start,end,step):
            grpt = subhalos[m]
            assert data[m/chunksize][0]==i
            if data[m/chunksize][1][m%chunksize] is None:
                print 'Saving none for subhalo',grpt
            else:
                k[grpt]=data[m/chunksize][1][m%chunksize]


#1: Create and write useful information
with tables.open_file(fout,'w') as f:
    f.create_carray("/","SubhaloDone",tables.Int16Col(),(nsubs,))
    f.create_carray("/","SubhaloMassInRadType",tables.Float32Col(),(nsubs,6))
    f.create_carray("/","SubhaloHalfmassRadType",tables.Float32Col(),(nsubs,6))
    f.root.SubhaloMassInRadType[:] = a.cat.SubhaloMassInRadType
    f.root.SubhaloHalfmassRadType[:] = a.cat.SubhaloHalfmassRadType
    f.flush()


print '#######################################################################################'
save_power_k = True
if save_power_k:
    print 'Calculating parameter for estimating convergence'
    a.setparams(0, [1.,2.],useReduced=False, binwidth=binwidth)

    def workpower(i):
        sl = slice (i, i + chunksize)
        start, end, step = sl.indices(N)
        return i,[a.getPowerK(t) for t in subhalos[np.arange(start,end,step)]]
    with sharedmem.MapReduce() as pool:
        outk = pool.map(workpower, range(0,N,chunksize))
        print 'CONVERGENCE DONE'
    with tables.open_file(fout,'r+') as f:
        createandsavepower(f,nsubs,nbins,outk,subhalos,chunksize)


#2: Do ellipsoidal shells at 1 and 2 stellar half-mass rad
print '#######################################################################################'
if not args.subhaloes:
    print '\n STARTING SHELLS'
    a.setparams(0, [1.,2.],useReduced=False, binwidth=binwidth)

    def workdm(i):
        sl = slice (i, i + chunksize)
        start, end, step = sl.indices(N)
        return i,[a.DM(t) for t in subhalos[np.arange(start,end,step)]]

    def workstars(i):
        sl = slice (i, i + chunksize)
        start, end, step = sl.indices(N)
        return i,[a.Stars(t) for t in subhalos[np.arange(start,end,step)]]

    def workgas(i):
        sl = slice (i, i + chunksize)
        start, end, step = sl.indices(N)
        return i,[a.Gas(t) for t in subhalos[np.arange(start,end,step)]]

    with sharedmem.MapReduce() as pool:
        outdm    = pool.map(workdm, range(0,N,chunksize))
        print 'DM DONE'
        outstars = pool.map(workstars, range(0,N,chunksize))
        print 'STARS DONE'
        #outgas   = pool.map(workgas, range(0,N,chunksize))
        #print 'GAS DONE'

    with tables.open_file(fout,'r+') as f:
        g = f.create_group("/", "dm_shell", "Local DM shapes from ellipsoidal shells")
        t = createandsavefields(f,g,nsubs,nbins,outdm,subhalos,chunksize)
        t[0][:] = 10**a.logr

        g = f.create_group("/", "stars_shell", "Local stellar shapes in ellipsoidal shells")
        t = createandsavefields(f,g,nsubs,nbins,outstars,subhalos,chunksize)
        t[0][:] = 10**a.logr

        #g = f.create_group("/", "gas_shell", "Local stellar shapes in ellipsoidal shells")
        #t = createandsavefields(f,g,nsubs,nbins,outgas,subhalos,chunksize)
        #t[0][:] = 10**a.logr

        f.flush()

#3: Do ellipsoidal volumes
    print '\n STARTING REDUCED'
    a.setparams(0, [1.,2.],useReduced=True, binwidth=binwidth)

    with sharedmem.MapReduce() as pool:
        outdm    = pool.map(workdm, range(0,N,chunksize))
        print 'DM DONE'
        outstars = pool.map(workstars, range(0,N,chunksize))
        print 'STARS DONE'
        #outgas   = pool.map(workgas, range(0,N,chunksize))

    with tables.open_file(fout,'r+') as f:
        g = f.create_group("/", "dm_reduced", "Local DM shapes from ellipsoidal shells")
        t = createandsavefields(f,g,nsubs,nbins,outdm,subhalos,chunksize)
        t[0][:] = 10**a.logr

        g = f.create_group("/", "stars_reduced", "Local stellar shapes in ellipsoidal shells")
        t = createandsavefields(f,g,nsubs,nbins,outstars,subhalos,chunksize)
        t[0][:] = 10**a.logr

        #g = f.create_group("/", "gas_reduced", "Local stellar shapes in ellipsoidal shells")
        #t = createandsavefields(f,g,nsubs,nbins,outgas,subhalos,chunksize,done=True)
        #t[0][:] = 10**a.logr

        f.flush()

    print '\n DONE!: Calculations for '+fout
