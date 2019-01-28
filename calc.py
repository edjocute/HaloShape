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

#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("snap", type=int, help="snapshot number")
parser.add_argument("t", help="Sim type: FP, DM or NR")
parser.add_argument("-c","--chunksize", type=int, default=100, help="Chunksize for parallel calculation")
parser.add_argument("--minmass", type=float, default=11, help="Min halo mass to be considered")
parser.add_argument("--maxmass", type=float, default=100, help="Min halo mass to be considered")
parser.add_argument("--nthreads", type=int, default=4, help="Number of threads for numexpr")
parser.add_argument("--test", action="store_true", help="Turn on testing mode")
parser.add_argument("--TNG", action="store_true", help="Use TNG dir? If this option is selected, supercedes --base")
parser.add_argument("-b","--base", default='/n/ghernquist/Illustris/Runs/', help="Base directory. Default = /n/ghernquist/Illustris/Runs/L75n1820")
#parser.add_argument("-hu","--hubble", default=0.704, help="Hubble parameter H_0=100h")
args = parser.parse_args()

print '######################################################'
print 'Calculating halo shapes!\n'
#dir='/n/ghernquist/Illustris/Runs/L75n1820DM/'
fbase = args.base
hubble = args.hubble
if (args.t.find('TNG')>0) or (args.TNG):
    print args.t
    print args.TNG
    fbase = "/n/hernquistfs3/IllustrisTNG/Runs/"
fdir = fbase+args.t+'/'
assert os.path.isdir(fdir),fdir+" does not exist!"

snap = args.snap
snapstr=str(snap).zfill(3)
fout = args.t+'_DMShape_'+snapstr+'.hdf5'

header=readsnapHDF5.snapshot_header(fdir+'/output/snapdir_'+snapstr+'/snap_'+snapstr)
boxsize = header.boxsize
hubble = header.hubble

chunksize = args.chunksize
ar.ellipsoid.ne.set_num_threads(args.nthreads)
mass=[args.minmass,args.maxmass]
if args.test:
    mass=[12,12.01]
    chunksize=5

print 'Directory: ',fdir
print 'Snapshot: ', snap
print 'Boxsize: ', boxsize
print 'Hubble parameter: ', hubble
print 'Halo mass range for calculation: 10^'+str(mass),'M_sun'
print 'Number of cores for sharedmem: ',sharedmem.cpu_count()
print 'Number of threads for numexpr: ',args.nthreads
print ' '

nbins=18; a=ar.AxialRatio(fdir,snap,nbins,rmin=10**-2.42857143)
#nbins=15; a=ar.AxialRatio(fdir,snap,nbins,rmin=10**-2)
groupmass=a.cat.Group_M_Crit200/hubble*1e10
groups=np.nonzero((groupmass>10**mass[0]) & (groupmass<10**mass[1]))[0]
nsubs=a.cat.nsubs
ngroups=a.cat.ngroups
N=len(groups)
print 'Subfind catalogue loaded'
print 'Total no. of subhalos =', nsubs
print 'Total no. of groups =', ngroups
print 'Groups to be calculated =', N
print '######################################################'
print ' '


def createandsavefields(f,g,ngroups,nbins,data,groups,chunksize,done=False):
    r=f.create_carray(g,"r",tables.Float32Col(),(nbins,))
    q=f.create_carray(g,"q",tables.Float32Col(),(ngroups,nbins))
    s=f.create_carray(g,"s",tables.Float32Col(),(ngroups,nbins))
    T=f.create_carray(g,"T",tables.Float32Col(),(ngroups,nbins))
    n=f.create_carray(g,"n",tables.Int32Col(),(ngroups,nbins,2))
    rotmat=f.create_carray(g,"RotMat",tables.Float32Col(),(ngroups,nbins,3,3))

    N = len(groups)
    for i in range(0,N,chunksize):
        sl = slice (i, i + chunksize)
        start, end, step = sl.indices(N)
        for m in np.arange(start,end,step):
            #print i,m
            #print m/chunksize
            grpt = groups[m]
            assert data[m/chunksize][0]==i
            if data[m/chunksize][1][m%chunksize] is None:
                print 'Saving none for group',grpt
            else:
                q[grpt],s[grpt],n[grpt],rotmat[grpt]=data[m/chunksize][1][m%chunksize]
                T[grpt]=(1-q[grpt]**2)/(1-s[grpt]**2)
                if done:
                    f.root.GroupDone[grpt]=1
    ## Be careful when saving triaxiality parameter, when s or q==-1, set T=-1 instead of np.nan
    wherenan = np.where( np.isnan(T[:]))
    for x,y in zip(wherenan[0],wherenan[1]):
        T[x,y] = -1.

    return r,q,s,T,n,rotmat


#1: Create and write useful information
with tables.open_file(fout,'w') as f:
    f.create_carray("/","GroupDone",tables.Int16Col(),(ngroups,))
    f.create_carray("/","Group_R_Crit200",tables.Float32Col(),(ngroups,))
    f.create_carray("/","Group_M_Crit200",tables.Float32Col(),(ngroups,))
    f.root.Group_M_Crit200[:] = a.cat.Group_M_Crit200
    f.root.Group_R_Crit200[:] = a.cat.Group_R_Crit200
    f.flush()

#2: Start with 20 ellipsoidal shells
print '#######################################################################################'
with sharedmem.MapReduce() as pool:
    def work(i):
        sl = slice (i, i + chunksize)
        start, end, step = sl.indices(N)
        return i,[a.DM(t) for t in groups[np.arange(start,end,step)]]
    out = pool.map(work, range(0,N,chunksize))

print 'Saving file'
with tables.open_file(fout,'r+') as f:
    g = f.create_group("/", "profile", "Local shapes from ellipsoidal shells with radius")
    createandsavefields(f,g,ngroups,nbins,out,groups,chunksize)
    f.root.profile.r[:] = 10**a.logr
    f.flush()

#3: Do 3 particular ellipsoidal shells next: 10%, 15%, 20%, 50%, 100% R_200
print ' '
print '#######################################################################################'
nbins = 5
a.setparams(fdir,snap, 0, [0.1,0.15,0.3,0.5,1.0])
with sharedmem.MapReduce() as pool:
    def work(i):
        sl = slice (i, i + chunksize)
        start, end, step = sl.indices(N)
        return i,[a.DM(t) for t in groups[np.arange(start,end,step)]]
    out = pool.map(work, range(0,N,chunksize))

with tables.open_file(fout,'r+') as f:
    g = f.create_group("/", "shell", "Local shapes from ellipsoidal shells at chosen radii")
    createandsavefields(f,g,ngroups,nbins,out,groups,chunksize)
    f.root.shell.r[:] = 10**a.logr
    f.flush()


#4: Finally, do ellipsoidal volumes next: 10%, 15%, 20%, 50%, 100% R_200
print ' '
print '#######################################################################################'
nbins = 5
a.setparams(fdir,snap, 0, [0.1,0.15,0.3,0.5,1.0],solid=True)
with sharedmem.MapReduce() as pool:
    def work(i):
        sl = slice (i, i + chunksize)
        start, end, step = sl.indices(N)
        return i,[a.DM(t) for t in groups[np.arange(start,end,step)]]
    out = pool.map(work, range(0,N,chunksize))

with tables.open_file(fout,'r+') as f:
    g = f.create_group("/", "reduced", "Shapes from ellipsoidal volumes with reduced inertia tensor at chosen radii")
    createandsavefields(f,g,ngroups,nbins,out,groups,chunksize,done=True)
    f.root.reduced.r[:] = 10**a.logr
    f.flush()

print 'Calculations done for:'+fout
