import numpy as np

def par2mean(L,dim,tZL):
    
    num_param = 0
    cum_param = np.zeros((2*L+1,))
    for iL in np.arange(0,L):
        num_param = num_param + dim[iL] * dim[iL+1]
        cum_param[2*iL+1] = num_param
        num_param = num_param + dim[iL+1]
        cum_param[2*iL+2] = num_param

    num_param = num_param.astype(int)
    cum_param = cum_param.astype(int)
    
    meang = np.zeros((num_param,))

    meang[cum_param[2*L-1]:cum_param[2*L]] = tZL

    return meang






def par2cov(L,dim,phi,ttphi,tPhi1,tPhiL,PHI1,PSI1,PHIL,ttPHIL):
    
    num_param = 0
    cum_param = np.zeros((2*L+1,))
    for iL in np.arange(0,L):
        num_param = num_param + dim[iL] * dim[iL+1]
        cum_param[2*iL+1] = num_param
        num_param = num_param + dim[iL+1]
        cum_param[2*iL+2] = num_param

    num_param = num_param.astype(int)
    cum_param = cum_param.astype(int)
    
    covg = np.zeros((num_param,num_param))

    # covariance

    # block-diagonal
    covg[cum_param[0]:cum_param[1],cum_param[0]:cum_param[1]] = np.kron(PHI1,np.eye(dim[1]))
    covg[cum_param[1]:cum_param[2],cum_param[1]:cum_param[2]] = ttphi[0] * np.eye(dim[1])
    for iL in np.arange(1,L-1):
        covg[cum_param[2*iL]:cum_param[2*iL+1],cum_param[2*iL]:cum_param[2*iL+1]] =  phi[iL-1] * np.eye(dim[iL+1]*dim[iL])
        covg[cum_param[2*iL+1]:cum_param[2*iL+2],cum_param[2*iL+1]:cum_param[2*iL+2]] =  ttphi[iL] * np.eye(dim[iL+1])
    covg[cum_param[2*(L-1)]:cum_param[2*L-1],cum_param[2*(L-1)]:cum_param[2*L-1]] = np.kron(np.eye(dim[L-1]),PHIL)
    covg[cum_param[2*L-1]:cum_param[2*L],cum_param[2*L-1]:cum_param[2*L]] = ttPHIL

    # block-tridiagonal
    covg[cum_param[1]:cum_param[2],cum_param[0]:cum_param[1]] = np.kron(tPhi1.T,np.eye(dim[1]))
    covg[cum_param[0]:cum_param[1],cum_param[1]:cum_param[2]] = covg[cum_param[1]:cum_param[2],cum_param[0]:cum_param[1]].T
    covg[cum_param[2*L-3]:cum_param[2*L-2],cum_param[2*L-2]:cum_param[2*L-1]] = np.kron(np.eye(dim[L-1]),tPhiL.T)
    covg[cum_param[2*L-2]:cum_param[2*L-1],cum_param[2*L-3]:cum_param[2*L-2]] = covg[cum_param[2*L-3]:cum_param[2*L-2],cum_param[2*L-2]:cum_param[2*L-1]].T
    
    if L==2:
    	covg[cum_param[0]:cum_param[1],cum_param[2]:cum_param[3]] = np.kron(PSI1,np.eye(dim[1])).reshape([dim[1],dim[0],dim[1],dim[2]],order='F').swapaxes(2,3).reshape([dim[1]*dim[0],dim[2]*dim[1]],order='F')
    	covg[cum_param[2]:cum_param[3],cum_param[0]:cum_param[1]] = covg[cum_param[0]:cum_param[1],cum_param[2]:cum_param[3]] .T
    
    return covg






def mean2par(L,dim,meang):

    num_param = 0
    cum_param = np.zeros((2*L+1,))
    for iL in np.arange(0,L):
        num_param = num_param + dim[iL] * dim[iL+1]
        cum_param[2*iL+1] = num_param
        num_param = num_param + dim[iL+1]
        cum_param[2*iL+2] = num_param

    num_param = num_param.astype(int)
    cum_param = cum_param.astype(int)
    
    tZL = meang[cum_param[2*L-1]:cum_param[2*L]]
             
    return tZL








def cov2par(L,dim,covg):
    
    num_param = 0
    cum_param = np.zeros((2*L+1,))
    for iL in np.arange(0,L):
        num_param = num_param + dim[iL] * dim[iL+1]
        cum_param[2*iL+1] = num_param
        num_param = num_param + dim[iL+1]
        cum_param[2*iL+2] = num_param

    num_param = num_param.astype(int)
    cum_param = cum_param.astype(int)

    dum = np.reshape( covg[ cum_param[0]:cum_param[1] , cum_param[0]:cum_param[1] ] , (dim[1],dim[0],dim[1],dim[0]) , order = 'F' )
    PHI1 = np.einsum('ijil->jl',dum) / dim[1]
    PHI1 = (PHI1 + PHI1.T)/2
                
    phi = np.zeros((L-2,))
    for iL in np.arange(1,L-1):
        phi[iL-1] = np.einsum('ii->',covg[ cum_param[2*iL]:cum_param[2*iL+1] , cum_param[2*iL]:cum_param[2*iL+1] ]) / dim[iL] / dim[iL+1]
        
    dum = np.reshape( covg[ cum_param[2*(L-1)]:cum_param[2*L-1] , cum_param[2*(L-1)]:cum_param[2*L-1] ] , (dim[L],dim[L-1],dim[L],dim[L-1]) , order = 'F' )
    PHIL = np.einsum('ijkj->ik',dum) / dim[L-1]          
    PHIL = (PHIL + PHIL.T)/2
                
    ttphi = np.zeros((L-1,))
    for iL in np.arange(0,L-1):
        ttphi[iL] = np.einsum('ii->',covg[ cum_param[2*iL+1]:cum_param[2*iL+2] , cum_param[2*iL+1]:cum_param[2*iL+2] ]) / dim[iL+1]
        
    ttPHIL = covg[cum_param[2*L-1]:cum_param[2*L],cum_param[2*L-1]:cum_param[2*L]]
    ttPHIL = (ttPHIL + ttPHIL.T)/2
    
    tPhi1 = np.zeros((dim[0],1))
    dum = np.reshape( covg[ cum_param[1]:cum_param[2] , cum_param[0]:cum_param[1] ] , (dim[1],dim[1],dim[0]) , order = 'F' )
    tPhi1 = tPhi1 + np.einsum('iik->k',dum).reshape((dim[0],1)) / dim[1]
    dum = np.reshape( covg[ cum_param[0]:cum_param[1] , cum_param[1]:cum_param[2] ] , (dim[1],dim[0],dim[1]) , order = 'F' )    
    tPhi1 = tPhi1 + np.einsum('iki->k',dum).reshape((dim[0],1)) / dim[1]
    tPhi1 = tPhi1 / 2
    
    tPhiL = np.zeros((dim[L],1))
    dum = np.reshape( covg[ cum_param[2*L-3]:cum_param[2*L-2] , cum_param[2*L-2]:cum_param[2*L-1] ] , (dim[L-1],dim[L],dim[L-1]) , order = 'F' )
    tPhiL = tPhiL + np.einsum('iki->k',dum).reshape((dim[L],1)) / dim[L-1]
    dum = np.reshape( covg[ cum_param[2*L-2]:cum_param[2*L-1], cum_param[2*L-3]:cum_param[2*L-2] ] , (dim[L],dim[L-1],dim[L-1]) , order = 'F' )
    tPhiL = tPhiL + np.einsum('kii->k',dum).reshape((dim[L],1)) / dim[L-1]
    tPhiL = tPhiL / 2
    
    PSI1=[]
    if L==2:
    	PSI1 = np.zeros([dim[0],dim[2]])
    	dum = np.reshape( covg[ cum_param[0]:cum_param[1] , cum_param[2]:cum_param[3] ] , (dim[1],dim[0],dim[2],dim[1]) , order = 'F' )
    	PSI1 = PSI1 + np.einsum('ijli->jl',dum) / dim[1]
    	dum = np.reshape( covg[ cum_param[2]:cum_param[3] , cum_param[0]:cum_param[1] ] , (dim[2],dim[1],dim[1],dim[0]) , order = 'F' )
    	PSI1 = PSI1 + np.einsum('liij->jl',dum) / dim[1]
    	PSI1 = PSI1 / 2

                 
    return phi,ttphi,tPhi1,tPhiL,PHI1,PSI1,PHIL,ttPHIL







def grad2par(L,dim,grad):
    
    num_param = 0
    cum_param = np.zeros((2*L+1,))
    for iL in np.arange(0,L):
        num_param = num_param + dim[iL] * dim[iL+1]
        cum_param[2*iL+1] = num_param
        num_param = num_param + dim[iL+1]
        cum_param[2*iL+2] = num_param

    num_param = num_param.astype(int)
    cum_param = cum_param.astype(int)
    
    R = np.shape(grad)[1]
    
    grad_res = np.reshape( grad[ cum_param[0]:cum_param[1] , : ] , (dim[1],dim[0],R) , order = 'F' )
    PHI1 = np.einsum('ijr,ilr->jl',grad_res,grad_res) / dim[1] / R
    
    phi = np.zeros((L-2,))    
    for iL in np.arange(1,L-1):
        phi[iL-1] = np.sum(grad[ cum_param[2*iL]:cum_param[2*iL+1] , : ]**2) / dim[iL+1] / dim[iL] / R
    
    grad_res = np.reshape( grad[ cum_param[2*(L-1)]:cum_param[2*L-1] , : ] , (dim[L],dim[L-1],R) , order = 'F' )
    PHIL = np.einsum('ijr,kjr->ik',grad_res,grad_res) / dim[L-1] / R
    
    ttphi = np.zeros((L-1,))
    for iL in np.arange(0,L-1):
        ttphi[iL] = np.sum(grad[ cum_param[2*iL+1]:cum_param[2*iL+2] , : ]**2) / dim[iL+1] / R

    grad_res = np.reshape( grad[ cum_param[0]:cum_param[1] , : ] , (dim[1],dim[0],R) , order = 'F' )
    tPhi1 = np.einsum('ir,ikr->k', grad[ cum_param[1]:cum_param[2] , : ] , grad_res ) / dim[1] / R
    
    grad_res = np.reshape( grad[ cum_param[2*(L-1)]:cum_param[2*L-1] , : ] , (dim[L],dim[L-1],R) , order = 'F' )
    tPhiL = np.einsum('jr,ijr->i', grad[ cum_param[2*L-3]:cum_param[2*L-2] , : ] , grad_res ) / dim[L-1] / R
    
    PSI1 = []
    if L==2:
    	grad_res1 = np.reshape( grad[ cum_param[0]:cum_param[1] , : ] , (dim[1],dim[0],R) , order = 'F' )
    	grad_res2 = np.reshape( grad[ cum_param[2]:cum_param[3] , : ] , (dim[2],dim[1],R) , order = 'F' )
    	PSI1 = np.einsum('ijr,lir->jl',grad_res1,grad_res2) / dim[1] / R
 
    grad_res = grad[ cum_param[2*L-1]:cum_param[2*L] , : ]
    ttPHIL = np.einsum('ir,jr->ij',grad_res,grad_res) / R
    tMuL = np.einsum('ir->i',grad_res) / R
    
    return phi,ttphi,tPhi1,tPhiL,PHI1,PSI1,PHIL,ttPHIL,tMuL










def matmat(L,dim,phi_1,ttphi_1,tPhi1_1,tPhiL_1,PHI1_1,PSI1_1,PHIL_1,ttPHIL_1,phi_2,ttphi_2,tPhi1_2,tPhiL_2,PHI1_2,PSI1_2,PHIL_2,ttPHIL_2):
    
    dim0 = np.copy(dim)
    for iL in np.arange(1,L):
        dim0[iL] = 1
    
    mat1 = par2cov(L,dim0,phi_1,ttphi_1,tPhi1_1,tPhiL_1,PHI1_1,PSI1_1,PHIL_1,ttPHIL_1)
    mat2 = par2cov(L,dim0,phi_2,ttphi_2,tPhi1_2,tPhiL_2,PHI1_2,PSI1_2,PHIL_2,ttPHIL_2)
    
    mat0 = mat1 @ mat2
    
    phi_0,ttphi_0,tPhi1_0,tPhiL_0,PHI1_0,PSI1_0,PHIL_0,ttPHIL_0 = cov2par(L,dim0,mat0)
    
    return phi_0,ttphi_0,tPhi1_0,tPhiL_0,PHI1_0,PSI1_0,PHIL_0,ttPHIL_0











def inverse(L,dim,phi,ttphi,tPhi1,tPhiL,PHI1,PSI1,PHIL,ttPHIL):
    
    dim0 = np.copy(dim)
    for iL in np.arange(1,L):
        dim0[iL] = 1

    mat = par2cov(L,dim0,phi,ttphi,tPhi1,tPhiL,PHI1,PSI1,PHIL,ttPHIL)
    imat = np.linalg.pinv(mat)
    phi_inv,ttphi_inv,tPhi1_inv,tPhiL_inv,PHI1_inv,PSI1_inv,PHIL_inv,ttPHIL_inv = cov2par(L,dim0,imat)

    return phi_inv,ttphi_inv,tPhi1_inv,tPhiL_inv,PHI1_inv,PSI1_inv,PHIL_inv,ttPHIL_inv








def squareroot(L,dim,phi,ttphi,tPhi1,tPhiL,PHI1,PSI1,PHIL,ttPHIL):
    
    dim0 = np.copy(dim)
    for iL in np.arange(1,L):
        dim0[iL] = 1

    mat = par2cov(L,dim0,phi,ttphi,tPhi1,tPhiL,PHI1,PSI1,PHIL,ttPHIL)
    evl, evc = np.linalg.eigh(mat)
    sqrtmat = (evc * np.sqrt(evl*np.heaviside(evl,0))) @ evc.T
    phi_sqrt,ttphi_sqrt,tPhi1_sqrt,tPhiL_sqrt,PHI1_sqrt,PSI1_sqrt,PHIL_sqrt,ttPHIL_sqrt= cov2par(L,dim0,sqrtmat)
    
    return phi_sqrt,ttphi_sqrt,tPhi1_sqrt,tPhiL_sqrt,PHI1_sqrt,PSI1_sqrt,PHIL_sqrt,ttPHIL_sqrt









def eig_unique(L,dim,phi,ttphi,tPhi1,tPhiL,PHI1,PSI1,PHIL,ttPHIL):
    
    dim0 = np.copy(dim)
    for iL in np.arange(1,L):
        dim0[iL] = 1

    mat = par2cov(L,dim0,phi,ttphi,tPhi1,tPhiL,PHI1,PSI1,PHIL,ttPHIL)
    evl, evc = np.linalg.eigh(mat)
    
    return evl






def matvec(L,dim,phi,ttphi,tPhi1,tPhiL,PHI1,PSI1,PHIL,ttPHIL,vec0):
    
    num_param = 0
    cum_param = np.zeros((2*L+1,))
    for iL in np.arange(0,L):
        num_param = num_param + dim[iL] * dim[iL+1]
        cum_param[2*iL+1] = num_param
        num_param = num_param + dim[iL+1]
        cum_param[2*iL+2] = num_param

    num_param = num_param.astype(int)
    cum_param = cum_param.astype(int)

    vec1 = np.zeros((num_param,))
    
    mat = vec0[cum_param[0]:cum_param[1]].reshape( (dim[1],dim[0]) , order='F' )
    vec1[cum_param[0]:cum_param[1]] = np.reshape( mat @ PHI1 , (dim[1]*dim[0],) , order='F' ) + np.squeeze(np.kron(tPhi1,vec0[cum_param[1]:cum_param[2],np.newaxis]))
    vec1[cum_param[1]:cum_param[2]] = np.squeeze( mat @ tPhi1 + ttphi[0]*vec0[cum_param[1]:cum_param[2],np.newaxis] )
    if L==2:
    	vec1[cum_param[2]:cum_param[3]] = np.reshape( PSI1.T @ mat.T , (dim[2]*dim[1],) , order = 'F' )
    for iL in np.arange(1,L-2):
        vec1[cum_param[2*iL]:cum_param[2*iL+1]] = phi[iL-1] * vec0[cum_param[2*iL]:cum_param[2*iL+1]]
        vec1[cum_param[2*iL+1]:cum_param[2*iL+2]] = ttphi[iL] * vec0[cum_param[2*iL+1]:cum_param[2*iL+2]]
    mat = vec0[cum_param[2*L-2]:cum_param[2*L-1]].reshape( (dim[L],dim[L-1]) , order='F' )    
    if L>2:    
    	vec1[cum_param[2*L-4]:cum_param[2*L-3]] = phi[L-3] * vec0[cum_param[2*L-4]:cum_param[2*L-3]]
    	vec1[cum_param[2*L-3]:cum_param[2*L-2]] = vec1[cum_param[2*L-3]:cum_param[2*L-2]] + ttphi[L-2] * vec0[cum_param[2*L-3]:cum_param[2*L-2]]
    vec1[cum_param[2*L-3]:cum_param[2*L-2]] = vec1[cum_param[2*L-3]:cum_param[2*L-2]] + np.reshape( tPhiL.T @ mat , (dim[L-1],) , order='F' )
    vec1[cum_param[2*L-2]:cum_param[2*L-1]] = vec1[cum_param[2*L-2]:cum_param[2*L-1]] + np.squeeze(np.kron(vec0[cum_param[2*L-3]:cum_param[2*L-2],np.newaxis],tPhiL)) + np.reshape( PHIL @ mat , (dim[L]*dim[L-1],) , order='F' )
    vec1[cum_param[2*L-1]:cum_param[2*L]] = np.squeeze(ttPHIL @ vec0[cum_param[2*L-1]:cum_param[2*L]])
    if L==2:
    	vec1[cum_param[0]:cum_param[1]] = vec1[cum_param[0]:cum_param[1]] + np.reshape( mat.T @ PSI1.T , (dim[1]*dim[0],) , order = 'F' )
        
    return vec1    








   




