import numpy as np


def par2cov(L,dim,PHI1i,PHI1ii,phi2i,phi2ii,phi2iii,phi2iv,PHILi,PHILii,Psi1i,Psi1ii,PsiLi,PsiLii,Ome13i,Ome13ii):
    
    num_param = 0
    cum_param = np.zeros((L+1,))
    for iL in np.arange(0,L):
        num_param = num_param + dim[iL] * dim[iL+1]
        cum_param[iL+1] = num_param

    num_param = num_param.astype(int)
    cum_param = cum_param.astype(int)
    
    covg = np.zeros((num_param,num_param))

    # covariance

    # block-diagonal
    covg[cum_param[0]:cum_param[1],cum_param[0]:cum_param[1]] = np.kron(PHI1i,np.eye(dim[1])) + np.kron(PHI1ii,np.ones((dim[1],dim[1]))) / dim[1]
    covg[cum_param[1]:cum_param[2],cum_param[1]:cum_param[2]] = phi2i * np.eye(dim[1]*dim[2]) + phi2ii / dim[2] * np.kron( np.eye(dim[1]) , np.ones((dim[2],dim[2])) ) + phi2iii / dim[1] * np.kron( np.ones((dim[1],dim[1])) , np.eye(dim[2]) ) + phi2iv / dim[1] / dim[2]
    covg[cum_param[2]:cum_param[3],cum_param[2]:cum_param[3]] = np.kron(np.eye(dim[2]),PHILi) + np.kron(np.ones((dim[2],dim[2])),PHILii) / dim[2]
    
    covg[cum_param[0]:cum_param[1],cum_param[1]:cum_param[2]] = np.kron( np.kron( Psi1i , np.eye(dim[1]) ) , np.ones((1,dim[2])) ) / np.sqrt(dim[2]) + np.kron( np.kron( Psi1ii , np.ones((dim[1],dim[1])) ) , np.ones((1,dim[2])) ) / np.sqrt(dim[2]) / dim[1]
    covg[cum_param[1]:cum_param[2],cum_param[0]:cum_param[1]] = covg[cum_param[0]:cum_param[1],cum_param[1]:cum_param[2]].T
    
    covg[cum_param[1]:cum_param[2],cum_param[2]:cum_param[3]] = np.kron( np.kron( np.ones((dim[1],1)) , np.eye(dim[2]) ) , PsiLi.T ) / np.sqrt(dim[1]) + np.kron( np.kron( np.ones((dim[1],1)) , np.ones((dim[2],dim[2])) ) , PsiLii.T ) / np.sqrt(dim[1]) / dim[2]
    covg[cum_param[2]:cum_param[3],cum_param[1]:cum_param[2]] = covg[cum_param[1]:cum_param[2],cum_param[2]:cum_param[3]].T
    
    covg[cum_param[0]:cum_param[1],cum_param[2]:cum_param[3]] = np.kron( np.kron( Ome13i , np.ones((dim[1],dim[2])) ) , np.ones((1,dim[3])) ) / np.sqrt(dim[1]*dim[2]*dim[3]) + np.kron( np.kron( np.ones((dim[0],1)) , np.ones((dim[1],dim[2])) ) , Ome13ii.T ) / np.sqrt(dim[0]*dim[1]*dim[2])
    covg[cum_param[2]:cum_param[3],cum_param[0]:cum_param[1]] = covg[cum_param[0]:cum_param[1],cum_param[2]:cum_param[3]].T
    
    return covg



   




