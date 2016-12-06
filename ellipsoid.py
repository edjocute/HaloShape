## Kunting Chua (2012)
## This function finds the axial ratios q=b/a and s=c/a
## given a set of points (pos) which should be an array of 3d vectors.
## This uses the method described in Dubinski and Carlberg (1995)
## using the eigenvalues of the modified inertia tensor

import numpy as np
import numexpr as ne
import newdot
#from scipy.linalg import blas as FP

def ellipsoidfit(posold,rvir,rin,rout,weighted=False):

    ###Initialize values###
    pos=posold.copy()
    q,s=1.,1.
    conv=10
    count=0
    exit=0
    axes=np.diag((1,1,1))
    while (conv > 1e-2 and exit!=1): ##r=1e-3 is the convergence criterion###
        count+=1

        ### Restrict to particles of required radii
        p0,p1,p2 = pos[:,0],pos[:,1],pos[:,2]
        #dist2=pos[:,0]**2 + (pos[:,1]/q)**2 + (pos[:,2]/s)**2
        dist2 = ne.evaluate("p0**2 + (p1/q)**2 + (p2/s) **2")
        slice= ne.evaluate("(dist2>rin**2) & (dist2<rout**2)")
        posbin=pos[slice]
        if len(posbin)==0:
            print "no particles in bin"
            return -1.,-1.,0, np.zeros((3,3))
            exit=1
        #print (len(posbin))


## THIS IS SLOW! DONT USE THIS ##
## USE NEW ROUTINE BELOW ##
        #for k in np.arange(len(posbin)):
            #a2=posbin[k,0]**2+ (posbin[k,1]/q)**2 + (posbin[k,2]/s)**2
        #    for i in np.arange(3):
        #        M[i,i]+=posbin[k,i]**2#/(a2)
        #        for j in np.arange(i):
        #            temp=posbin[k,i]*posbin[k,j]#/(a2)
        #            M[i,j]+=temp
        #            M[j,i]+=temp

## Do we want to use weights? ##
        if weighted:
            pb0,pb1,pb2 = posbin[:,0],posbin[:,1],posbin[:,2]
            a2 = ne.evaluate("pb0**2 + (pb1/q)**2 + (pb2/s) **2")
        else:
            a2 = 1

## Calculate inertia tensor
        M=np.zeros([3,3])
        for i in np.arange(3):
            pi=posbin[:,i]
            M[i,i]+= ne.evaluate("sum(pi**2/a2)")
            for j in np.arange(i):
                pj=posbin[:,j]
                temp= ne.evaluate("sum(pi* pj/a2)")
                M[i,j]+=temp
                M[j,i]+=temp
        M*=rvir**2
        M/=float(len(posbin))

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
            return -1.,-1.,len(posbin), np.zeros((3,3))
            exit=1

        if (newM < 1e-5).any():             ## Check that eigenvalues are not too small, else the next iteration can
                                            ## return imaginary values
            print 'eigenvalues too close to zero, stopping iteration'
            return -1.,-1.,len(posbin), np.zeros((3,3))
            exit=1

        ## Now we can obtain q and s from the eigenvalues
        q_new = np.sqrt(newM[1]/newM[0])
        s_new = np.sqrt(newM[2]/newM[0])

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
        conv = max( abs((q_new-q)/q), abs((s_new-s)/s) )
        q,s=q_new,s_new

        ###Other checks###
        if (count == 100):
            print 'Not converging at rin,rout = ',rin,rout, 'with', len(posbin),'in bin'
            exit=1
        elif (q>1 or s>1):
            print "q/s greater than 1!"
            exit=1
            print "lenbin = ",len(posbin)
    if not np.allclose(newdot.dot(axes.T,posold.T).T,pos):
        print 'Old and new positions do not match!!'
    #print posorg[0],pos[0],np.dot(posorg[0],axes)
    #if count < 100:
    #    print "It took ", count, "iterations to converge"
    return q,s,len(posbin),axes
