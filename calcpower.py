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
filters = tables.Filters(complevel=1, complib='zlib',shuffle=True)


#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("snap", type=int, help="snapshot number")
parser.add_argument("t", help="Sim type: FP, DM or NR")
parser.add_argument("-c","--chunksize", type=int, default=100, help="Chunksize for parallel calculation")
parser.add_argument("--minmass", type=float, default=11, help="Min halo mass to be considered")
parser.add_argument("--maxmass", type=float, default=100, help="Min halo mass to be considered")
parser.add_argument("--nthreads", type=int, default=4, help="Number of threads for numexpr")
parser.add_argument("--test", action="store_true", help="Turn on testing mode")
parser.add_argument("--subhaloes", action="store_true", help="Turn on shapes from subhaloes")
parser.add_argument("--TNG", action="store_true", help="Use TNG dir? If this option is selected, supercedes --base")
parser.add_argument("-b","--base", type=str, default='/n/ghernquist/Illustris/Runs/', help="Base directory. Default = /n/ghernquist/Illustris/Runs/")
#parser.add_argument("-hu","--hubble", default=0.704, help="Hubble parameter H_0=100h")
args = parser.parse_args()

print '######################################################'
print 'Calculating halo shapes!\n'
#dir='/n/ghernquist/Illustris/Runs/L75n1820DM/'
fbase = args.base
if (args.t.find('TNG')>0) or (args.TNG):
    print args.t
    print args.TNG
    fbase = "/n/hernquistfs3/IllustrisTNG/Runs/"
fbase = fbase if fbase[-1] == '/' else fbase+'/'
fdir = fbase+args.t+'/'
assert os.path.isdir(fdir),fdir+" does not exist!"

snap = args.snap
snapstr=str(snap).zfill(3)
fout = args.t+'_Power_'+snapstr+'.hdf5'

header=readsnapHDF5.snapshot_header(fdir+'/output/snapdir_'+snapstr+'/snap_'+snapstr)
boxsize = header.boxsize
hubble = header.hubble

chunksize = args.chunksize
#ar.ellipsoid.ne.set_num_threads(args.nthreads)
ar.ellipsoid.numba.config.NUMBA_NUM_THREADS = args.nthreads
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

nbins=18; a=ar.AxialRatio(fdir,snap,nbins,rmin=10**-2.42857143,useSubhaloes=args.subhaloes)
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
    print 'Saving results!'
    k = f.create_carray(g,"k",tables.Float32Col(),(ngroups,nbins))

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
                k[grpt]=data[m/chunksize][1][m%chunksize]
                f.root.GroupDone[grpt] = 1

    return k


#2: Start with 20 ellipsoidal shells
print '#######################################################################################'
if not args.subhaloes:
    with sharedmem.MapReduce() as pool:
        def work(i):
            sl = slice (i, i + chunksize)
            start, end, step = sl.indices(N)
            return i,[a.getPowerK(t) for t in groups[np.arange(start,end,step)]]
        out = pool.map(work, range(0,N,chunksize))

    with tables.open_file(fout,'w') as f:
        f.create_carray("/","GroupDone",tables.Int16Col(),(ngroups,),filters=filters)
        f.create_carray("/","r",tables.Float32Col(),(nbins,))

        k = createandsavefields(f,"/",ngroups,nbins,out,groups,chunksize)
        f.root.r[:] = 10**a.logr
        f.flush()


    print 'Calculations using all particles done for '+fout

