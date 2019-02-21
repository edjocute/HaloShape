## Kunting Chua (2012)
## This function finds the axial ratios q=b/a and s=c/a
## given a set of points (pos) which should be an array of 3d vectors.
## This uses the method described in Dubinski and Carlberg (1995)
## using the eigenvalues of the modified inertia tensor
##
## Usage:
## ellipsoidfit(posold,rvir,rin,rout,mass=False,weighted=False,convcrit=1e-2)
## 1) posold = positions dim=(N,3)
## 2) rvir = rvir in kpc
## 3) rin = inner r_ell of shell. rin = 0 -> ellipsoidal solids
## 4) rout = outer r_ell of shell. Units should be same as that of the input positions.
## 5) mass = False: no mass weightage e.g. DM-only runs
##         =  array of size N: weight particles by mass e.g. for FP runs
## 6) convcrit = Convergence criteria
##
## Output:
## 1) q (size N array)
## 2) s (size N array)
## 3) n (size N array): no. of particles in shell
## 4) M (size Nx3x3): columns indicate direction of principal axes \
##                      e.g. M[:,0]= direction of principal major axis

import numpy as np
import numexpr as ne
import newdot
#from scipy.linalg import blas as FP

def ellipsoidfit(posold, rvir, rin, rout, mass=None, weighted=False,\
        convcrit=1e-2, maxiter=150, verbose=False, returnall=False):

    if mass is not None:
        assert len(posold) == len(mass), 'Error: mass array seems wrong!!'

    ###Initialize values###
    #pos = posold.copy()
    p0,p1,p2 = posold[:,0],posold[:,1],posold[:,2]
    pos = posold[ne.evaluate("(p0**2 + p1**2 + p2**2) < rout**2")]

    qall = np.zeros(maxiter+1)
    sall = np.zeros(maxiter+1)
    qall[0], sall[0]= 1.,1.

    axes = np.zeros([maxiter+1,3,3])
    axes[0] = np.eye(3)
    #center = np.zeros(3)

    a2 = 1
    massbin = 1
    for it in xrange(maxiter):

        q,s = qall[it],sall[it]
        ### Restrict to particles of required radii
        p0,p1,p2 = pos[:,0],pos[:,1],pos[:,2]
        #dist2=pos[:,0]**2 + (pos[:,1]/q)**2 + (pos[:,2]/s)**2
        d2  = ne.evaluate("p0**2 + (p1/q)**2 + (p2/s) **2")
        sel = ne.evaluate("(d2>rin**2) & (d2<rout**2)")
        posbin = pos[sel]
        if mass is not None:
            massbin = mass[sel]
        lenbin = len(posbin)

        #meanpos = np.average(posbin,axis=0,weights=massbin)
        #center += meanpos
        #posbin = ne.evaluate("posbin - meanpos")

        #print 'it, nparts, q, s =',it,len(posbin),q,s
        #print 'meanpos = ',posbin.mean(axis=0)
        ### If few particles in volume, abort
        if lenbin <= 10:
            if verbose: print "Too few particles in bin: iter,N,rin,rout=",it,lenbin,rin,rout,q,s
            return -2.,-2.,lenbin, np.zeros((3,3)),np.count_nonzero(d2< (rin**2))

        ### Do we want to use a weighted shape tensor? ##
        if weighted:
            #pb0,pb1,pb2 = posbin[:,0],posbin[:,1],posbin[:,2]
            #a2 = ne.evaluate("pb0**2 + (pb1/q)**2 + (pb2/s) **2")
            a2 = d2[sel]

        ### Calculate shape tensor
        M=np.zeros([3,3])
        for i in np.arange(3):
            pi=posbin[:,i]
            M[i,i]+= ne.evaluate("sum(massbin*pi**2/a2)")
            for j in np.arange(i):
                pj=posbin[:,j]
                temp= ne.evaluate("sum(massbin*pi* pj/a2)")
                M[i,j]+=temp
                M[j,i]+=temp
        if mass is None:
            M /= len(posbin)
        else:
            M /= ne.evaluate("sum(massbin)")
        #print 'S_ij = ',M

        ## Get eigenvalues and eigenvectors of M[i,j]
        ## The COLUMNs of eigvec are the principal axes according to numpy documentation
        eigval,eigvec =np.linalg.eigh(M)

        ## Sort them from largest to smallest eigenvalue##
        arg = np.argsort(eigval)[::-1]
        E   = eigval[arg]
        V   = eigvec[:,arg]

        #print 'Eigenvalues = ',E
        #print 'Eigenvectors = ,',V,'\n'
        rotmat=V.T
        ## The actual rotation matrix is the transpose of rotmat
        ## because the transformation should be:
        ## I'= R I R.T

        ## Check rotation matrix using similarity transformation
        ## The convention here is M' = V.T M V
        ## which should correspond to the eigenvalues on the diagonal
        if not np.allclose( np.linalg.multi_dot((V.T,M,V)),np.diag(E),atol=1e-10):
            print "Error in similarity transformation!!"
            print np.linalg.multi_dot(V.T,M,V)
            print E
            return -1.,-1.,0, np.zeros((3,3)),np.count_nonzero(d2< (rin**2))


        ## Stop iterations if eigenvalues are too small, else imaginary values can be produced
        if (E < 1e-6).any():
            print 'eigenvalues are too close to zero, stopping iteration'
            print '\trvir =', rvir
            print '\titer, eigenvalues =',it, eigval
            print '\tS_ij =',M
            return -1.,-1.,0, np.zeros((3,3)),np.count_nonzero(d2< (rin**2))

        ## Now we can obtain q and s from the eigenvalues
        qall[it+1], sall[it+1] = np.sqrt(E[1:]/E[0])

        if (qall[it+1]>1 or sall[it+1]>1):
            print "Ellipsoidfit ERROR: q/s greater than 1!"
            return -1.,-1.,0, np.zeros((3,3)),np.count_nonzero(d2< (rin**2))

        ## Rotate postions into principal frame##
        ## R_kj X_ij = X_ki = R X.T
        ## Xik = (X_ki).T = (R X.T).T = X R.T
        #pos=np.dot(pos,rotmat) #this step is slow and is equivalent to np.dot(rotmat,pos[i])
        pos=newdot.dot(rotmat,pos.T).T    ## This is done using routine in scipy which is much faster
        #pos = newdot.dot(pos,rotmat.T)         ## than the one in numpy.
                                            ## We want x' = dot(V,x)

        ## We also carry the identity matrix so that we can obtain
        ## the overall rotation from the original frame x to the principal frame x'
        ## The rows of rotmat give the original axes in the principal frame
        ## The rows of axes give the final positions of the original axes in the principal frame
        ## The columns of rotmat give the principal axes in the orignal frame
        ## Don't mess this up!
        axes[it+1] = np.dot(axes[it],rotmat.T)

        conv = max( abs((qall[it+1]-q)/q), abs((sall[it+1]-s)/s) )
        if conv < convcrit:
            #if not np.allclose(newdot.dot(axes[it+1].T,posold.T).T,pos):
            #    print 'Ellipsoidfit warning: old and new positions do not match!!'
            enc=np.count_nonzero(ne.evaluate("d2 < rin**2"))

            if returnall:
                return qall,sall,len(posbin), axes, enc

            #print 'center = ',center
            #return q,s,len(posbin),axes[it],enc
            return qall[it+1],sall[it+1],len(posbin),axes[it+1],enc

    #print posorg[0],pos[0],np.dot(posorg[0],axes)

    if verbose:
        print 'Not converging after {} iterations at rin,rout = {},{} with {} particles'.format(\
                maxiter, rin,rout, len(posbin))
    return -1.,-1.,len(posbin), np.zeros((3,3)),np.count_nonzero(d2< (rin**2))


def ellipsoidfit2D(posold,rvir,rin,rout,mass=False,weighted=False):

    ###Initialize values###
    ndims=2
    pos=posold.copy()
    q=1.
    conv=10
    count=0
    exit=0
    axes=np.diag((1.,1.))

    if mass.__class__== bool:
        assert mass == False
    elif mass.__class__ == np.ndarray:
        assert len(posold) == len(mass)
    else:
        print 'Warning: mass array seems wrong!!'

    while (conv > 1e-2 and exit!=1): ##r=1e-3 is the convergence criterion###
        count+=1

        ### Restrict to particles of required radii
        p0,p1 = pos[:,0],pos[:,1]
        dist2 = ne.evaluate("p0**2 + (p1/q)**2")
        slice= ne.evaluate("(dist2>rin**2) & (dist2<rout**2)")
        posbin=pos[slice]
        if type(mass) == np.ndarray:
            massbin=mass[slice]

        if len(posbin)==0:
            print "no particles in bin"
            return -1.,0,np.zeros((2,2))
            exit=1

        ## Do we want to use a weighted shape tensor? ##
        if weighted:
            pb0,pb1 = posbin[:,0],posbin[:,1]
            a2 = ne.evaluate("pb0**2 + (pb1/q)**2")
        else:
            a2 = 1

        # Calculate inertia tensor
        M=np.zeros([ndims,ndims])
        if mass.__class__== bool:
            for i in np.arange(ndims):
                pi=posbin[:,i]
                M[i,i]+= ne.evaluate("sum(pi**2/a2)")
                for j in np.arange(i):
                    pj=posbin[:,j]
                    temp= ne.evaluate("sum(pi* pj/a2)")
                    M[i,j]+=temp
                    M[j,i]+=temp
            M=M*rvir**2/len(posbin)
        else:
            for i in np.arange(ndims):
                pi=posbin[:,i]
                M[i,i]+= ne.evaluate("sum(massbin*pi**2/a2)")
                for j in np.arange(i):
                    pj=posbin[:,j]
                    temp= ne.evaluate("sum(massbin* pi* pj/a2)")
                    M[i,j]+=temp
                    M[j,i]+=temp
            M=M*rvir**2/ne.evaluate("sum(massbin)")

###Get eigenvalues and eigenvectors of M[i,j] and sort them from largest to smallest eigenvalue###
        eigval,eigvec=np.linalg.eig(M)      ## get eigenvalues and normalized eigenvectors
                                            ## The COLUMNs of eigvec are the principal axes according to numpy documentation

        arg=np.argsort(eigval)[::-1]
        newM=eigval[arg]                    ## sort eigenvalues from largest to smallest
        rotmat=eigvec[:,arg]                ## sort the columns of the eigenvectors in the same mannor
                                            ## The actual rotation matrix is the transpose of rotmat
                                            ## Because the transformation should be I'= R I R.T

        ## Check rotation matrix using similarity transformation
        ## The convention here is M' = V.T M V
        ## which should correspond to the eigenvalues on the diagonal
        if not np.allclose( np.dot(rotmat.T,np.dot(M,rotmat)),np.diag(newM)):
            #print "Difference : ",abs(np.dot(rotmat.T,np.dot(M,rotmat))-np.diag(newM))
            print "Error in similarity transformation!!"
            print eigval,eigvec
            return -1.,len(posbin), np.zeros((ndims,ndims))
            exit=1

        if (newM < 1e-5).any():             ## Check that eigenvalues are not too small, else the next iteration can
                                            ## return imaginary values
            print 'eigenvalues too close to zero, stopping iteration'
            print count,eigval
            print eigvec
            return -1.,len(posbin), np.zeros((ndims,ndims))
            exit=1

        ## Now we can obtain q and s from the eigenvalues
        q_new = np.sqrt(newM[1]/newM[0])

        ## Rotate postions into principal ##
        #pos=np.dot(pos,rotmat) #this step is slow and is equivalent to np.dot(rotmat,pos[i])
        pos=newdot.dot(rotmat.T,pos.T).T    ## This is done using routine in scipy which is much faster
                                            ## than the one in numpy.
                                            ## We want x' = dot(V.T,x)

        axes=np.dot(axes,rotmat)            ## We also carry the identity matrix so that we can obtain
                                            ## the overall rotation from the original frame x
                                            ## to the principal frame x'
                                            ## The rows of rotmat give the original axes in the principal frame
                                            ## The rows of axes give the final positions of the original axes in the principal frame
                                            ## The columns of rotmat give the principal axes in the orignal frame
                                            ## Don't mess this up!
        conv = abs((q_new-q)/q)
        q=q_new

        ###Other checks###
        if (count == 100):
            print 'Not converging at rin,rout = ',rin,rout, 'with', len(posbin),'in bin'
            exit=1
        if (q>1):
            print "q greater than 1!"
            exit=1
            print "lenbin = ",len(posbin)
    #if not np.allclose(newdot.dot(axes.T,posold.T).T,pos):
    if not np.allclose(np.dot(posold,axes),pos):
        print 'Old and new positions do not match!!'
        #print (newdot.dot(axes.T,posold.T).T)[-2:]
        #print axes
        #print posold[-2:]
        #print pos[-2:]
    #print posorg[0],pos[0],np.dot(posorg[0],axes)
    #if count < 100:
    #    print "It took ", count, "iterations to converge"
    return q,len(posbin),axes

def test(N=1000,h=0.1,R=1.,q=1.,s=1.,phi=0.,theta=0.):
	from mpl_toolkits.mplot3d import Axes3D
	import matplotlib.pyplot as plt

	#Generate random points in an ellipsoidal shell
	allphi=np.random.rand(N)*2.*np.pi
	costheta = 2.*np.random.rand(N)-1.
	alltheta=np.arccos(costheta)

	r=np.zeros(N)
	tot=0
	while tot < N:
		u = np.random.rand(1)
		t = R* u**(1./3.)
        	if t>(1-h)*R:
			r[tot]=t
			tot+=1

	print "Run some tests:"
	x=r*np.sin(alltheta)*np.cos(allphi)
	y=q*r*np.sin(alltheta)*np.sin(allphi)
	z=s*r*np.cos(alltheta)

	#Rotate ellipsoid
	phi=np.deg2rad(phi)
	theta=np.deg2rad(theta)
	Rp=np.array([[np.cos(phi),-np.sin(phi),0.],[np.sin(phi),np.cos(phi),0.],[0.,0.,1.]])
	Rt=np.array([[1.,0.,0.],[0,np.cos(theta),-np.sin(theta)],[0.,np.sin(theta),np.cos(theta)]])
	Rtot=Rt.dot(Rp)

	#fig = plt.figure(figsize=(6,6))
	#ax = fig.add_subplot(111, projection='3d')
	#ax.scatter(x,y,z)

	pos=np.vstack([x,y,z]).T
	pos=np.dot(Rtot,pos.T).T

	out=ellipsoidfit(pos,1.,0,1e10,convcrit=1e-3)
	#out=ellipsoidfit(pos,R,1.0,1.2,convcrit=1e-2)
	print 'Results:'
	print 'q=',out[0],'expected=',q
	print 's=',out[1],'expected=',s
	print 'n=',out[2]
	print 'R=',out[3]
	print Rtot
        rdot = [out[3][:,i].dot(Rtot[:,i]) for i in range(3)]
        print rdot
	print ''

if __name__=="__main__":
	N=5000
	print "1) q=1, s=1 :"
	test(N,q=1.,s=1.)

	print "2) q=0.9,s=0.7:"
	test(N,q=0.9,s=0.7)

	print "3) q=0.5,s=0.25:"
	test(N,q=0.5,s=0.25)

	print "4) q=0.5,s=0.25,phi=45,theta=0"
	test(N,q=0.5,s=0.25,phi=45.,theta=0.)

	print "5) q=0.5,s=0.25,phi=45,theta=60"
	test(N,q=0.5,s=0.25,phi=45.,theta=60.)

	print "6) q=0.5,s=0.25,phi=0,theta=60"
	test(N,q=0.5,s=0.25,phi=0.,theta=60.)

	print "7) q=0.5,s=0.25,phi=30,theta=30"
	test(N,q=0.5,s=0.25,phi=30.,theta=30.)
	test(100000,q=0.8,s=0.5,phi=30.,theta=30.,h=0.9,R=2)
