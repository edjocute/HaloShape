## Read File ##
## Returns a list which has a length of the number of subhaloes in the file
## Which entry contains:
## 1. Group index
## 2. Number of subhaloes in the group, excluding the parent halo
## 3. Array of subhalo index
import numpy as np
def readdat(filename):
	f=open(filename,'rb')
	nsub=np.fromfile(f,dtype='uint32',count=1)

	List=[]
	for i in np.arange(nsub):
	        gal=np.fromfile(f,dtype='uint32',count=1)
        	count=np.fromfile(f,dtype='uint32',count=1)
	        sub=np.fromfile(f,dtype='uint32',count=count)
	        List.append([gal[0],count[0],sub])

	f.close()
	return List

def mean(a):
	b=np.zeros(len(a[0,:]))
	for i in np.arange(len(a[0,:])):
		b[i]=a[:,i][a[:,i]>0].mean()
	return b

#Finds shortest distance in periodic box	
def shortestdistold(a,b,boxsize):#a and b are 3-d vectors
	distmin=np.sqrt(((a-b)**2).sum())
	if distmin <= boxsize/2.:
		return distmin
	else:
		for x in np.linspace(-boxsize,boxsize,3):
		  for y in np.linspace(-boxsize,boxsize,3):
		    for z in np.linspace(-boxsize,boxsize,3):
			btemp=np.array([b[0]+x,b[1]+y,b[2]+z])
			disttemp=np.sqrt(((a-btemp)**2).sum())
			if disttemp <=boxsize/2.:
				distmin=disttemp
				return disttemp
			if disttemp<distmin:
			  distmin=disttemp
		return distmin

#Returns image of point b closest to point a in a periodic box
def image(a,b,boxsize):#a and b are 3-d vectors
	diff=b-a
	b[(b-a)>(boxsize/2.)]-=boxsize
	b[(b-a)<(-boxsize/2.)]+=boxsize
	return b

def shortestdist(a,b,boxsize):#a and b are 3-d vectors
	if len(b.shape)==1:
		return np.sqrt(((image(a,b,boxsize)-a)**2).sum())
	else:
		return np.sqrt(((image(a,b,boxsize)-a)**2).sum(axis=1))

def readcat(cat,mass,boxsize):
	import numpy as np
	massg=cat.Group_M_Crit200
	mass=mass-10
	gal=np.nonzero((massg>10**mass) & (massg < 10**(mass+0.5)))[0]
	List=[]
	for i in gal:
		galpos=cat.GroupPos[i]
		count=np.array([],dtype='uint32')
		for j in np.arange(cat.GroupNsubs[i]-1):
			j=j+cat.GroupFirstSub[i]+1#skip main halo
			if cat.SubhaloLen[j] >=100:
				dist=shortestdist(galpos,cat.SubhaloPos[j],boxsize)
				if (dist <= cat.Group_R_Crit200[i]):
					if cat.SubhaloMass[j] > massg[i]:
						print 'Error: submass larger than parent halo'
					else:
						count=np.append(count,j)
				#else:
					#print 'Error, outside of Rvir'
		List.append([i,len(count),count])
	return List	


def readcatall(cat,mass,boxsize,which=0):
        import numpy as np
        massg=cat.Group_M_Crit200
        mass=mass-10
        gal=np.nonzero((massg>10**mass) & (massg < 10**(mass+0.5)))[0]
        List=[]
	if which == 0:
		rad=np.inf*np.ones(cat.ngroups)
	elif which == 1:
		rad=cat.Group_R_Crit200
	elif which == 2 :
		rad=cat.Group_R_TopHat200
        for i in gal:
                galpos=cat.GroupPos[i]
                count=np.array([],dtype='uint32')
                for j in np.arange(cat.GroupNsubs[i]-1):
                        j=j+cat.GroupFirstSub[i]+1#skip main halo
                        if cat.SubhaloLen[j] >=100:
				dist=shortestdist(galpos,cat.SubhaloPos[j],boxsize)
                                if (dist <= rad[i]):
                                        if cat.SubhaloMass[j] > massg[i]:
                                                print 'Error: submass larger than parent halo'
                                        else:
                                                count=np.append(count,j)
                List.append([i,len(count),count])
        return List

def vir(dist,mass,D=200):
        import numpy as np
        import conversions as conv
        import sys

        #if isinstance(mass[0],int):
        #        massc=mass[1]
        #        distc=dist[1]
        #else:
        #        massc=np.concatenate((mass[0],mass[1],mass[2]))
        #        distc=np.concatenate((dist[0],dist[1],dist[2]))
        #stack=np.column_stack((dist,mass))
        arg=np.argsort(dist)
	distsort=dist[arg]
	masssort=mass[arg]
        rhocrit=conv.GetRhoCrit()
	rho=masssort.cumsum()/(4./3.*np.pi*distsort**3)
	if (rho <= D*rhocrit).sum()!=0:
		n=np.nanmin(np.nonzero(rho <= D*rhocrit))
	else:
		n=-1
	return distsort[n],(masssort.cumsum())[n]
	
def virold(dist,mass,D=200):
        import numpy as np
        import conversions as conv
        import sys

        #if isinstance(mass[0],int):
        #        massc=mass[1]
        #        distc=dist[1]
        #else:
        #        massc=np.concatenate((mass[0],mass[1],mass[2]))
        #        distc=np.concatenate((dist[0],dist[1],dist[2]))
        #stack=np.column_stack((dist,mass))
        arg=np.argsort(dist)
        distsort=dist[arg]
        masssort=mass[arg]
        #stack=stack[arg,:]
        rhocrit=conv.GetRhoCrit()
        M=0.
        rad=0.
        for i in np.arange(len(distsort)):
                Mnew=M+masssort[i]
                if distsort[i] != 0:
                        rho=Mnew/(4./3. * np.pi * (distsort[i]) **3)
                        if rho <= D*rhocrit:
                               #if (stack[i,0]-stack[i-1,0])/stack[i-1,0] > 0.01:
                               #        print 'Virial radius inaccurate'
                                return distsort[i],Mnew
                                break
                        else:
                                M=Mnew
