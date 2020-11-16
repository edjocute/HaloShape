import numpy as np
import readsubfHDF5
import os,sys
import argparse
import tables
import snapshot,readsnapHDF5,utils
import numexpr as ne


parser = argparse.ArgumentParser()
parser.add_argument("snap", type=int, help="snapshot number")
parser.add_argument("t", help="Sim type: FP, DM or NR")
parser.add_argument("-b","--base", default='/n/ghernquist/Illustris/Runs/', help="Base directory. Default = /n/ghernquist/Illustris/Runs/L75n1820/")
parser.add_argument("--minmass", type=float, default=11, help="Min halo mass to be considered")
parser.add_argument("--maxmass", type=float, default=100, help="Min halo mass to be considered")

args = parser.parse_args()


print '######################################################'
print 'Saving Stellar Mass!'
#dir='/n/ghernquist/Illustris/Runs/L75n1820DM/'
fbase = args.base
if (args.t.find('TNG')>0):
    print args.t
    fbase = "/n/hernquistfs3/IllustrisTNG/Runs/"
fdir = fbase+args.t+'/output/'
assert os.path.isdir(fdir),fdir+" does not exist!"
snap = args.snap
snapstr = str(snap).zfill(3)
fout = '{}_StellarMass_{:03d}.hdf5'.format(args.t,snap)

header = readsnapHDF5.snapshot_header('{0}/snapdir_{1:03d}/snap_{1:03d}'.format(fdir,snap))
hubble = header.hubble

box=header.boxsize


print 'Directory: ',fdir
print 'Snapshot: ', snap
print 'Redshift: ', header.redshift
print 'Hubble: ', header.hubble
print 'Massarr ', header.massarr
print 'Halo mass range for calculation: 10^[{},{}] M_sun'.format(args.minmass,args.maxmass)



#1: Create and write useful information
cat=readsubfHDF5.subfind_catalog(fdir,snap, keysel=["Group_R_Crit200","GroupFirstSub","Group_M_Crit200","SubhaloMassInRadType","SubhaloHalfmassRadType","SubhaloMassInHalfRadType","SubhaloPos"])
first = cat.GroupFirstSub[:]
#mgal=cat.SubhaloMassInRadType[cat.GroupFirstSub[:]]
ngroups = cat.ngroups
groupmass = cat.Group_M_Crit200*1e10/hubble
groups = np.nonzero((groupmass>10**args.minmass) & (groupmass<10**args.maxmass))[0]

print 'To calculate {} groups'.format(len(groups))

with tables.open_file(fout,'w') as f:
    f.create_carray("/","Group_M_Crit200",tables.Float32Col(),(ngroups,))
    f.create_carray("/","GroupMassInRadType",tables.Float32Col(),(ngroups,6))
    f.create_carray("/","GroupMassInHalfRadType",tables.Float32Col(),(ngroups,6))
    f.create_carray("/","GroupHalfmassRadType",tables.Float32Col(),(ngroups,6))
    masses = f.create_carray("/","GroupMassType",tables.Float32Col(), (ngroups,4,6))

    f.root.Group_M_Crit200[:] = cat.Group_M_Crit200
    f.root.GroupMassInRadType[:] = cat.SubhaloMassInRadType[first]
    f.root.GroupMassInHalfRadType[:] = cat.SubhaloMassInHalfRadType[first]
    f.root.GroupHalfmassRadType[:] = cat.SubhaloHalfmassRadType[first]

    f.flush()


    for grp in groups:
        for parttype in [0,4]:
            subid = cat.GroupFirstSub[grp]
            pos = snapshot.loadSubhalo(fdir,snap,subid,parttype,["Coordinates"]) #kpc/h
            mass = snapshot.loadSubhalo(fdir,snap,subid,parttype,["Masses"]) #1e10 msun/h

            if type(pos) is dict:
                assert pos['count'] == 0

            else:
                pos = utils.image(pos-cat.SubhaloPos[subid],None,header.boxsize) #kpc
                #pos = ((pos - cat.SubhaloPos[subid])-box/2.)%box - box/2.
                dist = np.linalg.norm(pos,axis=1)/hubble

                masses[grp,0,parttype] = mass[dist < 5].sum()
                masses[grp,1,parttype] = mass[dist < 10].sum()
                masses[grp,2,parttype] = mass[dist < 30].sum()
                masses[grp,3,parttype] = mass[dist < 100].sum()

    f.flush()
