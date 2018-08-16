###########################################################################
# Script to calculate halo shapes for a single snapshot.
# Example: run calc.py 135 DM
# Example: run calc.py 130 FP -b /n/ghernquist/Illustris/Runs/L75n910
# Min halo mass = 10^12 M_sun
# #########################################################################


import numpy as np
import readsubfHDF5
import os,sys
import argparse
import tables
import sharedmem
from functools import partial
#sys.path.append(os.path.join(os.path.dirname(__file__),"code"))
#sys.path.append('')
import axial_ratios as ar

mass=[12,12.01]

#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("snap", type=int, help="snapshot number")
parser.add_argument("t", help="Sim type: FP, DM or NR")
parser.add_argument("-b","--base", default='/n/ghernquist/Illustris/Runs/L75n1820', help="Base directory. Default = /n/ghernquist/Illustris/Runs/L75n1820")
parser.add_argument("-c","--chunksize", type=int, default=100, help="Chunksize for parallel calculation")
parser.add_argument("--minmass", type=int, default=11, help="Min halo mass to be considered")
parser.add_argument("--nthreads", type=int, default=4, help="Number of threads for numexpr")
args = parser.parse_args()

#dir='/n/ghernquist/Illustris/Runs/L75n1820DM/'
fdir = args.base+args.t+'/'
snap = args.snap
fout = str(snap).zfill(4)

chunksize = args.chunksize
ar.ellipsoid.ne.threads=args.nthreads

print 'Directory: ',fdir
print 'Snapshot: ', snap
print 'Number of threads for numexpr: ', args.nthreads
nbins=15

a=ar.AxialRatio(fdir,snap,nbins)
#subhalomass=a.cat.SubhaloMass/0.704*1e10
#subhalos=np.nonzero((subhalomass>10**(12)) & subhalomass<10**12.1))[0]
groupmass=a.cat.Group_M_Crit200/0.704*1e10
groups=np.nonzero((groupmass>10**mass[0]) & (groupmass<10**mass[1]))[0]
nsubs=a.cat.nsubs
ngroups=a.cat.ngroups
N=len(groups)
print 'Subfind catalogue loaded'
print 'Total no. of subhalos =', nsubs
print 'Total no. of groups =', ngroups
print 'Groups to be calculated =', N
print ''

def createandsavefields(f,g,ngroups,nbins,data,groups,chunksize,done=False):
    r=f.create_carray(g,"r",tables.Float32Col(),(nbins,))
    q=f.create_carray(g,"q",tables.Float32Col(),(ngroups,nbins))
    s=f.create_carray(g,"s",tables.Float32Col(),(ngroups,nbins))
    T=f.create_carray(g,"T",tables.Float32Col(),(ngroups,nbins))
    n=f.create_carray(g,"n",tables.Int32Col(),(ngroups,nbins))
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
            q[grpt],s[grpt],n[grpt],rotmat[grpt]=data[m/chunksize][1][m%chunksize]
            T[grpt]=(1-q[i]**2)/(1-s[i]**2)
            if done:
                f.root.GroupDone[grpt]=1
       #for grpt,i in zip(groups,xrange(len(groups))):
       # q[grpt],s[grpt],n[grpt],rotmat[grpt]=out[i]
       # T[grpt]=(1-q[i]**2)/(1-s[i]**2)
    return r,q,s,T,n,rotmat


#1: Create and write useful information
with tables.open_file('L75n1820DM.hdf5','w') as f:
    #f.create_carray("/","SubhaloID",tables.Int32Col(),(nsub,))
    #f.create_carray("/","SubhaloDone",tables.Int16Col(),(nsub,))
    #f.create_carray("/","SubhaloMass",tables.Float32Col(),(nsub,))
    #f.root.SubhaloMass[:]=subhalomass
    f.create_carray("/","GroupDone",tables.Int16Col(),(ngroups,))
    f.create_carray("/","Group_R_Crit200",tables.Float32Col(),(ngroups,))
    f.create_carray("/","Group_M_Crit200",tables.Float32Col(),(ngroups,))
    f.root.Group_M_Crit200[:] = a.cat.Group_M_Crit200
    f.root.Group_R_Crit200[:] = a.cat.Group_R_Crit200
    f.flush()

#2: Start with 15 ellipsoidal shellsa
print '#######################################################################################'
with sharedmem.MapReduce() as pool:
    def work(i):
        sl = slice (i, i + chunksize)
        start, end, step = sl.indices(N)
        return i,[a.DM(t) for t in groups[np.arange(start,end,step)]]
    out = pool.map(work, range(0,N,chunksize))

with tables.open_file('L75n1820DM.hdf5','r+') as f:
    g = f.create_group("/", "shell", "Shapes from ellipsoidal shells with radius")
    createandsavefields(f,g,ngroups,nbins,out,groups,chunksize)
    f.root.shell.r[:] = 10**a.logr
    f.flush()

#3: Do 3 particular ellipsoidal shells next: 10%, 15%, 20%, 50%, 100% R_200
print ' '
print '#######################################################################################'
nbins = 5
a.setparams(fdir,snap, 0, [0.1,0.15,0.2,0.5,1.0])
with sharedmem.MapReduce() as pool:
    def work(i):
        sl = slice (i, i + chunksize)
        start, end, step = sl.indices(N)
        return i,[a.DM(t) for t in groups[np.arange(start,end,step)]]
    out = pool.map(work, range(0,N,chunksize))

with tables.open_file('L75n1820DM.hdf5','r+') as f:
    g = f.create_group("/", "fixedRshell", "Shapes from ellipsoidal shells at chosen radii")
    createandsavefields(f,g,ngroups,nbins,out,groups,chunksize)
    f.root.fixedRshell.r[:] = 10**a.logr
    f.flush()


#4: Finally, do ellipsoidal volumes next: 10%, 15%, 20%, 50%, 100% R_200
print ' '
print '#######################################################################################'
nbins = 5
a.setparams(fdir,snap, 0, [0.1,0.15,0.2,0.5,1.0],solid=True)
with sharedmem.MapReduce() as pool:
    def work(i):
        sl = slice (i, i + chunksize)
        start, end, step = sl.indices(N)
        return i,[a.DM(t) for t in groups[np.arange(start,end,step)]]
    out = pool.map(work, range(0,N,chunksize))

with tables.open_file('L75n1820DM.hdf5','r+') as f:
    g = f.create_group("/", "fixedRvol", "Shapes from ellipsoidal volumes at chosen radii")
    createandsavefields(f,g,ngroups,nbins,out,groups,chunksize,done=True)
    f.root.fixedRvol.r[:] = 10**a.logr
    f.flush()

print 'Calculations done!'
