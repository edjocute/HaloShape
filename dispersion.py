import snapshot
import readsubfHDF5
import numpy as np
import tables

which='DM'

dir='/n/hernquistfs1/Illustris/Runs/L75n1820'+which+'/output/'
snapnum=135
massrange=[11,15]
nbins=20

cat=readsubfHDF5.subfind_catalog(dir,135,\
        keysel=["Group_M_Crit200","Group_R_Crit200","GroupVel","GroupPos","GroupFirstSub"])

massg=cat.Group_M_Crit200/0.704*1e10
N=len(massg)
gallist=np.nonzero((massg>10**massrange[0]) & (massg < 10**massrange[1]))[0]

edges=10**np.linspace(np.log10(1e-2),np.log10(1),nbins+1)
y=(edges[1:]+edges[:-1])/2

with tables.open_file('1820'+which+'VelDisp.hdf5','w') as f:
    s2=f.create_carray('/','Veldisp2',tables.Float32Col(),(N,nbins))
    rs2=f.create_carray('/','RadialVeldisp2',tables.Float32Col(),(N,nbins))
    ts2=f.create_carray('/','TanVeldisp2',tables.Float32Col(),(N,nbins))
    B=f.create_carray('/','VelAniso',tables.Float32Col(),(N,nbins))
    r=f.create_carray('/','Radius',tables.Float32Col(),(nbins,))

    for gal in gallist:
        part=snapshot.loadSubhalo(dir,snapnum,cat.GroupFirstSub[gal],1,["Coordinates","Velocities"])
        pos=part["Coordinates"]-cat.GroupPos[gal]
        vel=part["Velocities"]-cat.GroupVel[gal]

        dist=np.sqrt((pos**2).sum(axis=1))/cat.Group_R_Crit200[gal]
        for i in np.arange(nbins):
            v=vel[(dist<=edges[i+1]) & (dist > edges[i])]
            binpos = pos[(dist<=edges[i+1]) & (dist > edges[i])]
            vr=binpos*((v*binpos).sum(axis=1)/(binpos**2).sum(axis=1))[:,np.newaxis]

    	#s2[gal,i]=np.mean((v**2).sum(axis=1))-((np.mean(v,axis=0))**2).sum()
    	    s2[gal,i]=np.mean((v**2).sum(axis=1))
            rs2[gal,i]=np.mean((  vr**2    ).sum(axis=1))
            ts2[gal,i]=np.mean(( (v-vr)**2 ).sum(axis=1))
        B[gal]=1-ts2[gal]/(2*rs2[gal])

        r[:]=y
