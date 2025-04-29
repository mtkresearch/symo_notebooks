import numpy as np

def par2mean(L,dim,wm):
    
    num_param = 0
    cum_param = np.zeros((L+1,))
    for iL in np.arange(0,L):
        num_param = num_param + dim[iL] * dim[iL+1]
        cum_param[iL+1] = num_param

    num_param = num_param.astype(int)
    cum_param = cum_param.astype(int)
    
    meang = np.zeros((num_param,))

    # mean
    for iL in np.arange(0,L):
        meang[cum_param[iL]:cum_param[iL+1]] = wm[iL]

    return meang






def par2cov(L,dim,wd,wt,wo):
    
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
    for iL in np.arange(0,L):
        covg[cum_param[iL]:cum_param[iL+1],cum_param[iL]:cum_param[iL+1]] =  wd[0,iL] * np.eye(dim[iL]*dim[iL+1]) + wd[1,iL] / dim[iL+1] * np.kron( np.eye(dim[iL]) , np.ones((dim[iL+1],dim[iL+1])) ) + wd[2,iL] / dim[iL] * np.kron( np.ones((dim[iL],dim[iL])) , np.eye(dim[iL+1]) ) + wd[3,iL] / dim[iL] / dim[iL+1]

    # block-tridiagonal
    for iL in np.arange(0,L-1):
        covg[cum_param[iL]:cum_param[iL+1],cum_param[iL+1]:cum_param[iL+2]] = wt[1,iL] / dim[iL+1] / np.sqrt(dim[iL]*dim[iL+2]) + wt[0,iL] / np.sqrt(dim[iL]*dim[iL+2]) * np.kron( np.kron( np.ones((dim[iL],1)) , np.eye(dim[iL+1]) ) , np.ones((1,dim[iL+2])) ) 
        covg[cum_param[iL+1]:cum_param[iL+2],cum_param[iL]:cum_param[iL+1]] = covg[cum_param[iL]:cum_param[iL+1],cum_param[iL+1]:cum_param[iL+2]].T

    # all other blocks
    iwo = 0
    for iL in np.arange(0,L-2):
        for jL in np.arange(0,iL+1):
            covg[cum_param[iL+2]:cum_param[iL+3],cum_param[jL]:cum_param[jL+1]] = wo[iwo] / np.sqrt(dim[iL+2]*dim[iL+3]) / np.sqrt(dim[jL]*dim[jL+1])
            covg[cum_param[jL]:cum_param[jL+1],cum_param[iL+2]:cum_param[iL+3]] = covg[cum_param[iL+2]:cum_param[iL+3],cum_param[jL]:cum_param[jL+1]].T
            iwo = iwo + 1

    return covg






def mean2par(L,dim,meang):
    
    num_param = 0
    cum_param = np.zeros((L+1,))
    for iL in np.arange(0,L):
        num_param = num_param + dim[iL] * dim[iL+1]
        cum_param[iL+1] = num_param

    num_param = num_param.astype(int)
    cum_param = cum_param.astype(int)

    wmest = np.zeros((L,))

    for iL in np.arange(0,L):

        for id1 in np.arange(0,dim[iL]): # columns of weight matrix, rows of covariance
            for id2 in np.arange(0,dim[iL+1]): # rows  of weight matrix, rows of covariance
                wmest[iL] = wmest[iL] + meang[ cum_param[iL]+id2+dim[iL+1]*id1 ]

        wmest[iL] = wmest[iL] / dim[iL] / dim[iL+1]
             
    return wmest







def cov2par(L,dim,covg):
    
    num_param = 0
    cum_param = np.zeros((L+1,))
    for iL in np.arange(0,L):
        num_param = num_param + dim[iL] * dim[iL+1]
        cum_param[iL+1] = num_param

    num_param = num_param.astype(int)
    cum_param = cum_param.astype(int)

    wdest = np.zeros((4,L))
    wtest = np.zeros((2,L-1))
    woest = np.zeros(((L-2)*(L-1)//2,))
    
    
    # block-diagonal
    for iL in np.arange(0,L):

        for id1 in np.arange(0,dim[iL]): # columns of weight matrix, rows of covariance
            for id2 in np.arange(0,dim[iL+1]): # rows  of weight matrix, rows of covariance
                wdest[0,iL] = wdest[0,iL] + covg[ cum_param[iL]+id2+dim[iL+1]*id1 , cum_param[iL]+id2+dim[iL+1]*id1 ]
                for id3 in np.arange(0,dim[iL+1]): # rows  of weight matrix, columns of covariance
                    wdest[1,iL] = wdest[1,iL] + covg[ cum_param[iL]+id2+dim[iL+1]*id1 , cum_param[iL]+id3+dim[iL+1]*id1 ]
                for id3 in np.arange(0,dim[iL]): # columns of weight matrix, columns of covariance
                    wdest[2,iL] = wdest[2,iL] + covg[ cum_param[iL]+id2+dim[iL+1]*id1 , cum_param[iL]+id2+dim[iL+1]*id3 ]
                    for id4 in np.arange(0,dim[iL+1]): # rows  of weight matrix, columns of covariance
                        wdest[3,iL] = wdest[3,iL] + covg[ cum_param[iL]+id2+dim[iL+1]*id1 , cum_param[iL]+id4+dim[iL+1]*id3 ]

        #wdest[1,iL] = dim[iL] * dim[iL+1] ( w1 + w2/dim[iL+1] + w3/dim[iL] + w4/dim[iL]/dim[iL+1] )
        #wdest[2,iL] = dim[iL] * dim[iL+1]**2 * ( w2/dim[iL+1] + w4/dim[iL]/dim[iL+1] ) + dim[iL] * dim[iL+1] * ( w1 + w3/dim[iL] )
        #wdest[3,iL] = dim[iL]**2 * dim[iL+1] * ( w3/dim[iL] + w4/dim[iL]/dim[iL+1] ) + dim[iL] * dim[iL+1] * ( w1 + w2/dim[iL+1] )
        #wdest[4,iL] = dim[iL] * dim[iL+1] * w1 + dim[iL] * dim[iL+1]**2 * w2/dim[iL+1] + dim[iL]**2 * dim[iL+1] * w3/dim[iL] + dim[iL]**2 * dim[iL+1]**2 * w4/dim[iL]/dim[iL+1]

        lintr = np.array( [ [ dim[iL]*dim[iL+1], dim[iL], dim[iL+1], 1 ] , 
                           [ dim[iL]*dim[iL+1], dim[iL]*dim[iL+1], dim[iL+1], dim[iL+1] ] , 
                           [ dim[iL]*dim[iL+1], dim[iL], dim[iL]*dim[iL+1], dim[iL] ] , 
                           [ dim[iL]*dim[iL+1] , dim[iL]*dim[iL+1] , dim[iL]*dim[iL+1] , dim[iL]*dim[iL+1] ] ] )     
        ilintr = np.linalg.pinv(lintr)

        dum = ilintr @ wdest[:,iL]

        wdest[0,iL] = dum[0]
        wdest[1,iL] = dum[1]
        wdest[2,iL] = dum[2]
        wdest[3,iL] = dum[3]


    # tri-diagonal
    for iL in np.arange(0,L-1):
        for id1 in np.arange(0,dim[iL+1]): # columns of weight matrix, rows of covariance, and rows of weight matrix, columns of covariance
            for id2 in np.arange(0,dim[iL+2]): # rows  of weight matrix, rows of covariance
                for id3 in np.arange(0,dim[iL]): # columns of weight matrix, columns of covariance
                    wtest[0,iL] = wtest[0,iL] + covg[ cum_param[iL+1]+id2+dim[iL+2]*id1 , cum_param[iL]+id1+dim[iL+1]*id3 ]
                    wtest[0,iL] = wtest[0,iL] + covg[ cum_param[iL]+id1+dim[iL+1]*id3 , cum_param[iL+1]+id2+dim[iL+2]*id1 ] # transpose block
                    for id4 in np.arange(0,dim[iL+1]): # rows  of weight matrix, columns of covariance
                        wtest[1,iL] = wtest[1,iL] + covg[ cum_param[iL+1]+id2+dim[iL+2]*id1 , cum_param[iL]+id4+dim[iL+1]*id3 ]
                        wtest[1,iL] = wtest[1,iL] + covg[ cum_param[iL]+id4+dim[iL+1]*id3 , cum_param[iL+1]+id2+dim[iL+2]*id1 ] # transpose block

        #wtest[5,iL] = 2 * dim[iL] * dim[iL+1] * dim[iL+2] * ( w5/sqrt(dim[iL]*dim[iL+2]) + w6/dim[iL+1]/sqrt(dim[iL]*dim[iL+2]) )
        #wtest[6,iL] = 2 * dim[iL] * dim[iL+1] * dim[iL+2] * w5/sqrt(dim[iL]*dim[iL+2]) + 2 * dim[iL] * dim[iL+1]**2 * dim[iL+2] * w6/dim[iL+1]/sqrt(dim[iL]*dim[iL+2])

        lintr = np.array( [ [ 2 * dim[iL+1] * np.sqrt(dim[iL]*dim[iL+2]) , 2 * np.sqrt(dim[iL]*dim[iL+2]) ] , 
                           [ 2 * dim[iL+1] * np.sqrt(dim[iL]*dim[iL+2]) , 2 * dim[iL+1] * np.sqrt(dim[iL]*dim[iL+2]) ] ] )     
        ilintr = np.linalg.pinv(lintr)

        dum = ilintr @ wtest[:,iL]

        wtest[0,iL] = dum[0]
        wtest[1,iL] = dum[1]


    # other blocks
    iwo = 0
    for iL in np.arange(0,L-2):
        for jL in np.arange(0,iL+1):
            woest[iwo] = ( np.mean(covg[cum_param[iL+2]:cum_param[iL+3],cum_param[jL]:cum_param[jL+1]]) + np.mean(covg[cum_param[jL]:cum_param[jL+1],cum_param[iL+2]:cum_param[iL+3]]) ) / 2 * np.sqrt(dim[iL+2]*dim[iL+3]*dim[jL]*dim[jL+1])
            iwo = iwo + 1
             
    return wdest, wtest, woest









def grad2par(L,dim,grad):
    
    num_param = 0
    cum_param = np.zeros((L+1,))
    for iL in np.arange(0,L):
        num_param = num_param + dim[iL] * dim[iL+1]
        cum_param[iL+1] = num_param

    num_param = num_param.astype(int)
    cum_param = cum_param.astype(int)
    
    R = np.shape(grad)[1]

    wmest = np.zeros((L,))
    wdest = np.zeros((4,L))
    wtest = np.zeros((2,L-1))
    woest = np.zeros(((L-2)*(L-1)//2,))

    # mean and block-diagonal
    for iL in np.arange(0,L):
        for ir in np.arange(0,R):
            
            for id1 in np.arange(0,dim[iL]): 
                for id2 in np.arange(0,dim[iL+1]): 
                    wmest[iL] = wmest[iL] + grad[ cum_param[iL]+id2+dim[iL+1]*id1 , ir ]
                    wdest[0,iL] = wdest[0,iL] + grad[ cum_param[iL]+id2+dim[iL+1]*id1, ir ]**2
            
            for id1 in np.arange(0,dim[iL]): 
                dum = 0
                for id2 in np.arange(0,dim[iL+1]): 
                    dum = dum + grad[ cum_param[iL]+id2+dim[iL+1]*id1, ir ]
                wdest[1,iL] = wdest[1,iL] + dum**2
            
            for id2 in np.arange(0,dim[iL+1]): 
                dum = 0
                for id1 in np.arange(0,dim[iL]): 
                    dum = dum + grad[ cum_param[iL]+id2+dim[iL+1]*id1, ir ]
                wdest[2,iL] = wdest[2,iL] + dum**2
                
            wdest[3,iL] = wdest[3,iL] + np.sum(grad[ cum_param[iL]:cum_param[iL+1] , ir ])**2
                    
        wmest[iL] = wmest[iL] / dim[iL] / dim[iL+1] / R
        wdest[:,iL] = wdest[:,iL] / R

        #wdest[1,iL] = dim[iL] * dim[iL+1] ( w1 + w2/dim[iL+1] + w3/dim[iL] + w4/dim[iL]/dim[iL+1] )
        #wdest[2,iL] = dim[iL] * dim[iL+1]**2 * ( w2/dim[iL+1] + w4/dim[iL]/dim[iL+1] ) + dim[iL] * dim[iL+1] * ( w1 + w3/dim[iL] )
        #wdest[3,iL] = dim[iL]**2 * dim[iL+1] * ( w3/dim[iL] + w4/dim[iL]/dim[iL+1] ) + dim[iL] * dim[iL+1] * ( w1 + w2/dim[iL+1] )
        #wdest[4,iL] = dim[iL] * dim[iL+1] * w1 + dim[iL] * dim[iL+1]**2 * w2/dim[iL+1] + dim[iL]**2 * dim[iL+1] * w3/dim[iL] + dim[iL]**2 * dim[iL+1]**2 * w4/dim[iL]/dim[iL+1]

        lintr = np.array( [ [ dim[iL]*dim[iL+1], dim[iL], dim[iL+1], 1 ] , 
                           [ dim[iL]*dim[iL+1], dim[iL]*dim[iL+1], dim[iL+1], dim[iL+1] ] , 
                           [ dim[iL]*dim[iL+1], dim[iL], dim[iL]*dim[iL+1], dim[iL] ] , 
                           [ dim[iL]*dim[iL+1] , dim[iL]*dim[iL+1] , dim[iL]*dim[iL+1] , dim[iL]*dim[iL+1] ] ] )     
        ilintr = np.linalg.pinv(lintr)

        dum = ilintr @ wdest[:,iL]

        wdest[0,iL] = dum[0]
        wdest[1,iL] = dum[1]
        wdest[2,iL] = dum[2]
        wdest[3,iL] = dum[3]

        
    # block-tridiagonal
    for iL in np.arange(0,L-1):
        for ir in np.arange(0,R):
            
            for id1 in np.arange(0,dim[iL+1]):
                dum0 = 0
                for id2 in np.arange(0,dim[iL+2]):
                    dum0 = dum0 + grad[ cum_param[iL+1]+id2+dim[iL+2]*id1 , ir ]
                dum1 = 0
                for id3 in np.arange(0,dim[iL]):
                    dum1 = dum1 + grad[ cum_param[iL]+id1+dim[iL+1]*id3 , ir ]
                wtest[0,iL] = wtest[0,iL] + dum0 * dum1
                
            wtest[1,iL] = wtest[1,iL] + np.sum(grad[ cum_param[iL+1]:cum_param[iL+2] , ir]) * np.sum(grad[ cum_param[iL]:cum_param[iL+1] , ir ])
            
        wtest[:,iL] = wtest[:,iL] / R

        #west[5,iL] = dim[iL] * dim[iL+1] * dim[iL+2] * ( w5/sqrt(dim[iL]*dim[iL+2]) + w6/dim[iL+1]/sqrt(dim[iL]*dim[iL+2]) )
        #west[6,iL] = dim[iL] * dim[iL+1] * dim[iL+2] * w5/sqrt(dim[iL]*dim[iL+2]) + dim[iL] * dim[iL+1]**2 * dim[iL+2] * w6/dim[iL+1]/sqrt(dim[iL]*dim[iL+2])

        lintr = np.array( [ [ dim[iL+1] * np.sqrt(dim[iL]*dim[iL+2]) , np.sqrt(dim[iL]*dim[iL+2]) ] , 
                           [ dim[iL+1] * np.sqrt(dim[iL]*dim[iL+2]) , dim[iL+1] * np.sqrt(dim[iL]*dim[iL+2]) ] ] )     
        ilintr = np.linalg.pinv(lintr)

        dum = ilintr @ wtest[:,iL]

        wtest[0,iL] = dum[0]
        wtest[1,iL] = dum[1]


        
    # other blocks
    iwo = 0
    for iL in np.arange(0,L-2):
        for jL in np.arange(0,iL+1):
            for ir in np.arange(0,R):
                woest[iwo] = woest[iwo] + np.mean(grad[ cum_param[iL+2]:cum_param[iL+3] , ir ]) * np.mean(grad[ cum_param[jL]:cum_param[jL+1] , ir ])
            woest[iwo] = woest[iwo] / R * np.sqrt(dim[iL+2]*dim[iL+3]*dim[jL]*dim[jL+1])
            iwo = iwo + 1
                 
    return wmest, wdest, wtest, woest








def grad2par_fast(L,dim,grad):
    
    num_param = 0
    cum_param = np.zeros((L+1,))
    for iL in np.arange(0,L):
        num_param = num_param + dim[iL] * dim[iL+1]
        cum_param[iL+1] = num_param

    num_param = num_param.astype(int)
    cum_param = cum_param.astype(int)
    
    R = np.shape(grad)[1]

    wmest = np.zeros((L,))
    wdest = np.zeros((4,L))
    wtest = np.zeros((2,L-1))
    woest = np.zeros(((L-2)*(L-1)//2,))

    # mean and block-diagonal
    for iL in np.arange(0,L):
        for ir in np.arange(0,R):
            
            grad_res = np.reshape( grad[ cum_param[iL]:cum_param[iL+1] , ir ] , (dim[iL+1],dim[iL]) , order = 'F' )
            wmest[iL] = wmest[iL] + np.sum( grad_res )
            wdest[0,iL] = wdest[0,iL] + np.sum( grad_res**2 )
            wdest[1,iL] = wdest[1,iL] + np.sum( np.sum( grad_res , axis=0 ) **2 )
            wdest[2,iL] = wdest[2,iL] + np.sum( np.sum( grad_res , axis=1 ) **2 )
            wdest[3,iL] = wdest[3,iL] + np.sum( grad_res )**2
                    
        wmest[iL] = wmest[iL] / dim[iL] / dim[iL+1] / R
        wdest[:,iL] = wdest[:,iL] / R

        lintr = np.array( [ [ dim[iL]*dim[iL+1], dim[iL], dim[iL+1], 1 ] , 
                           [ dim[iL]*dim[iL+1], dim[iL]*dim[iL+1], dim[iL+1], dim[iL+1] ] , 
                           [ dim[iL]*dim[iL+1], dim[iL], dim[iL]*dim[iL+1], dim[iL] ] , 
                           [ dim[iL]*dim[iL+1] , dim[iL]*dim[iL+1] , dim[iL]*dim[iL+1] , dim[iL]*dim[iL+1] ] ] )     
        ilintr = np.linalg.pinv(lintr)

        dum = ilintr @ wdest[:,iL]

        wdest[0,iL] = dum[0]
        wdest[1,iL] = dum[1]
        wdest[2,iL] = dum[2]
        wdest[3,iL] = dum[3]

        
    # block-tridiagonal
    for iL in np.arange(0,L-1):
        for ir in np.arange(0,R):
            
            grad_res0 = np.reshape( grad[ cum_param[iL]:cum_param[iL+1] , ir ] , (dim[iL+1],dim[iL]) , order = 'F' )
            grad_res1 = np.reshape( grad[ cum_param[iL+1]:cum_param[iL+2] , ir ] , (dim[iL+2],dim[iL+1]) , order = 'F' )

            wtest[0,iL] = wtest[0,iL] + np.sum( grad_res0 , axis=1 ) @ np.sum( grad_res1 , axis=0 )
            wtest[1,iL] = wtest[1,iL] + np.sum( grad_res0 ) * np.sum( grad_res1 )
            
        wtest[:,iL] = wtest[:,iL] / R

        lintr = np.array( [ [ dim[iL+1] * np.sqrt(dim[iL]*dim[iL+2]) , np.sqrt(dim[iL]*dim[iL+2]) ] , 
                           [ dim[iL+1] * np.sqrt(dim[iL]*dim[iL+2]) , dim[iL+1] * np.sqrt(dim[iL]*dim[iL+2]) ] ] )     
        ilintr = np.linalg.pinv(lintr)

        dum = ilintr @ wtest[:,iL]

        wtest[0,iL] = dum[0]
        wtest[1,iL] = dum[1]


        
    # other blocks
    iwo = 0
    for iL in np.arange(0,L-2):
        for jL in np.arange(0,iL+1):
            for ir in np.arange(0,R):
                woest[iwo] = woest[iwo] + np.mean(grad[ cum_param[iL+2]:cum_param[iL+3] , ir ]) * np.mean(grad[ cum_param[jL]:cum_param[jL+1] , ir ])
            woest[iwo] = woest[iwo] / R * np.sqrt(dim[iL+2]*dim[iL+3]*dim[jL]*dim[jL+1])
            iwo = iwo + 1
                 
    return wmest, wdest, wtest, woest









def matmat(L,wd1,wt1,wo1,wd2,wt2,wo2):
    
    dim = np.zeros((L+1,))
    for iL in range(L+1):
        dim[iL] = 2
    dim = dim.astype(int)
    
    mat1 = par2cov(L,dim,wd1,wt1,wo1)
    mat2 = par2cov(L,dim,wd2,wt2,wo2)
    
    mat0 = mat1 @ mat2
    
    wd0, wt0, wo0 = cov2par(L,dim,mat0)
    
    return wd0, wt0, wo0











def inverse(L,wd,wt,wo):
    
    dim = np.zeros((L+1,))
    for iL in range(L+1):
        dim[iL] = 2
    dim= dim.astype(int)

    mat = par2cov(L,dim,wd,wt,wo)
    imat = np.linalg.inv(mat)
    wdinv, wtinv, woinv = cov2par(L,dim,imat)

    return wdinv, wtinv, woinv








def squareroot(L,wd_2,wt_2,wo_2):
    
    dim = np.zeros((L+1,))
    for iL in range(L+1):
        dim[iL] = 2
    dim= dim.astype(int)

    mat = par2cov(L,dim,wd_2,wt_2,wo_2)
    evl, evc = np.linalg.eigh(mat)
    sqrtmat = (evc * np.sqrt(evl)) @ evc.T
    wdsqrt, wtsqrt, wosqrt = cov2par(L,dim,sqrtmat)
    
    return wdsqrt, wtsqrt, wosqrt









def eig_unique(L,wd,wt,wo):
    
    dim = np.zeros((L+1,))
    for iL in range(L+1):
        dim[iL] = 2
    dim= dim.astype(int)

    mat = par2cov(L,dim,wd,wt,wo)
    evl, evc = np.linalg.eigh(mat)
    
    return evl















def matvec(L,dim,wd0,wt0,wo0,vec0):
    
    num_param = 0
    cum_param = np.zeros((L+1,))
    for iL in np.arange(0,L):
        num_param = num_param + dim[iL] * dim[iL+1]
        cum_param[iL+1] = num_param

    num_param = num_param.astype(int)
    cum_param = cum_param.astype(int)

    vec1 = np.zeros((num_param,))
    
    Vec0 = []
    
    for iL in np.arange(L):
        mat = vec0[cum_param[iL]:cum_param[iL+1]].reshape( (dim[iL+1],dim[iL]) , order='F' )
        Vec0.append(mat)

    for iL in np.arange(L):
        vec1[cum_param[iL]:cum_param[iL+1]] = vec1[cum_param[iL]:cum_param[iL+1]] + wd0[0,iL] * vec0[cum_param[iL]:cum_param[iL+1]]
        vec1[cum_param[iL]:cum_param[iL+1]] = vec1[cum_param[iL]:cum_param[iL+1]] + wd0[1,iL]/dim[iL+1] * np.reshape( np.ones((dim[iL+1],1)) @ np.ones((1,dim[iL+1])) @ Vec0[iL] , (dim[iL+1]*dim[iL],) , order='F' )
        vec1[cum_param[iL]:cum_param[iL+1]] = vec1[cum_param[iL]:cum_param[iL+1]] + wd0[2,iL]/dim[iL] * np.reshape( Vec0[iL] @ np.ones((dim[iL],1)) @ np.ones((1,dim[iL])) , (dim[iL+1]*dim[iL],) , order='F' )
        vec1[cum_param[iL]:cum_param[iL+1]] = vec1[cum_param[iL]:cum_param[iL+1]] + wd0[3,iL]/dim[iL]/dim[iL+1] * np.ones((dim[iL+1]*dim[iL],1)) @ np.ones((1,dim[iL+1]*dim[iL])) @ vec0[cum_param[iL]:cum_param[iL+1]]

    for iL in np.arange(L):
        if iL<L-1:
            vec1[cum_param[iL]:cum_param[iL+1]] = vec1[cum_param[iL]:cum_param[iL+1]] + wt0[0,iL]/np.sqrt(dim[iL]*dim[iL+2]) * np.reshape( Vec0[iL+1].T @ np.ones((dim[iL+2],1)) @ np.ones((1,dim[iL])) , (dim[iL+1]*dim[iL],) , order='F' )
            vec1[cum_param[iL]:cum_param[iL+1]] = vec1[cum_param[iL]:cum_param[iL+1]] + wt0[1,iL]/dim[iL+1]/np.sqrt(dim[iL]*dim[iL+2]) * np.ones((dim[iL+1]*dim[iL],1)) @ np.ones((1,dim[iL+2]*dim[iL+1])) @ vec0[cum_param[iL+1]:cum_param[iL+2]]
        if iL>0:
            vec1[cum_param[iL]:cum_param[iL+1]] = vec1[cum_param[iL]:cum_param[iL+1]] + wt0[0,iL-1]/np.sqrt(dim[iL-1]*dim[iL+1]) * np.reshape( np.ones((dim[iL+1],1)) @ np.ones((1,dim[iL-1])) @ Vec0[iL-1].T , (dim[iL+1]*dim[iL],) , order='F' )
            vec1[cum_param[iL]:cum_param[iL+1]] = vec1[cum_param[iL]:cum_param[iL+1]] + wt0[1,iL-1]/dim[iL]/np.sqrt(dim[iL-1]*dim[iL+1]) * np.ones((dim[iL+1]*dim[iL],1)) @ np.ones((1,dim[iL]*dim[iL-1])) @ vec0[cum_param[iL-1]:cum_param[iL]]
        
    for iL in np.arange(L):
        for jL in np.arange(iL-1):
            iwo = round(jL + (iL-2)*(iL-1)/2)
            vec1[cum_param[iL]:cum_param[iL+1]] = vec1[cum_param[iL]:cum_param[iL+1]] + wo0[iwo]/np.sqrt(dim[iL]*dim[iL+1])/np.sqrt(dim[jL]*dim[jL+1]) * np.ones((dim[iL+1]*dim[iL],1)) @ np.ones((1,dim[jL+1]*dim[jL])) @ vec0[cum_param[jL]:cum_param[jL+1]]
        for jL in np.arange(iL+2,L):
            iwo = round(iL + (jL-2)*(jL-1)/2)
            vec1[cum_param[iL]:cum_param[iL+1]] = vec1[cum_param[iL]:cum_param[iL+1]] + wo0[iwo]/np.sqrt(dim[iL]*dim[iL+1])/np.sqrt(dim[jL]*dim[jL+1]) * np.ones((dim[iL+1]*dim[iL],1)) @ np.ones((1,dim[jL+1]*dim[jL])) @ vec0[cum_param[jL]:cum_param[jL+1]]
        
    return vec1    

    
 
 
 
 
 
def matvec_fast(L,dim,wd0,wt0,wo0,vec0):
    
    num_param = 0
    cum_param = np.zeros((L+1,))
    for iL in np.arange(0,L):
        num_param = num_param + dim[iL] * dim[iL+1]
        cum_param[iL+1] = num_param

    num_param = num_param.astype(int)
    cum_param = cum_param.astype(int)

    vec1 = np.zeros((num_param,))
    
    Vec0 = []
    
    for iL in np.arange(L):
        mat = vec0[cum_param[iL]:cum_param[iL+1]].reshape( (dim[iL+1],dim[iL]) , order='F' )
        Vec0.append(mat)

    for iL in np.arange(L):
        vec1[cum_param[iL]:cum_param[iL+1]] = vec1[cum_param[iL]:cum_param[iL+1]] + wd0[0,iL] * vec0[cum_param[iL]:cum_param[iL+1]]
        vec1[cum_param[iL]:cum_param[iL+1]] = vec1[cum_param[iL]:cum_param[iL+1]] + wd0[1,iL]/dim[iL+1] * np.reshape( np.repeat( np.sum( Vec0[iL] , axis=0 , keepdims=True ) , dim[iL+1] , axis=0 ) , (dim[iL+1]*dim[iL],) , order='F' )
        vec1[cum_param[iL]:cum_param[iL+1]] = vec1[cum_param[iL]:cum_param[iL+1]] + wd0[2,iL]/dim[iL] * np.reshape( np.repeat( np.sum( Vec0[iL] , axis=1 , keepdims=True ) , dim[iL] , axis=1 ) , (dim[iL+1]*dim[iL],) , order='F' )   
        vec1[cum_param[iL]:cum_param[iL+1]] = vec1[cum_param[iL]:cum_param[iL+1]] + wd0[3,iL]/dim[iL]/dim[iL+1] * np.sum(vec0[cum_param[iL]:cum_param[iL+1]]) * np.ones((dim[iL+1]*dim[iL],))

    for iL in np.arange(L):
        if iL<L-1:
            vec1[cum_param[iL]:cum_param[iL+1]] = vec1[cum_param[iL]:cum_param[iL+1]] + wt0[0,iL]/np.sqrt(dim[iL]*dim[iL+2]) * np.reshape( np.repeat( np.sum( Vec0[iL+1] , axis=0 , keepdims=True ).T , dim[iL] , axis=1 ) , (dim[iL+1]*dim[iL],) , order='F' )
            vec1[cum_param[iL]:cum_param[iL+1]] = vec1[cum_param[iL]:cum_param[iL+1]] + wt0[1,iL]/dim[iL+1]/np.sqrt(dim[iL]*dim[iL+2]) * np.ones((dim[iL+1]*dim[iL],)) * np.sum(vec0[cum_param[iL+1]:cum_param[iL+2]])
        if iL>0:
            vec1[cum_param[iL]:cum_param[iL+1]] = vec1[cum_param[iL]:cum_param[iL+1]] + wt0[0,iL-1]/np.sqrt(dim[iL-1]*dim[iL+1]) * np.reshape( np.repeat( np.sum( Vec0[iL-1] , axis=1 , keepdims=True ).T , dim[iL+1] , axis=0 ) , (dim[iL+1]*dim[iL],) , order='F' )
            vec1[cum_param[iL]:cum_param[iL+1]] = vec1[cum_param[iL]:cum_param[iL+1]] + wt0[1,iL-1]/dim[iL]/np.sqrt(dim[iL-1]*dim[iL+1]) * np.ones((dim[iL+1]*dim[iL],)) * np.sum(vec0[cum_param[iL-1]:cum_param[iL]])
        
    for iL in np.arange(L):
        for jL in np.arange(iL-1):
            iwo = round(jL + (iL-2)*(iL-1)/2)
            vec1[cum_param[iL]:cum_param[iL+1]] = vec1[cum_param[iL]:cum_param[iL+1]] + wo0[iwo]/np.sqrt(dim[iL]*dim[iL+1])/np.sqrt(dim[jL]*dim[jL+1]) * np.ones((dim[iL+1]*dim[iL],)) * np.sum(vec0[cum_param[jL]:cum_param[jL+1]])
        for jL in np.arange(iL+2,L):
            iwo = round(iL + (jL-2)*(jL-1)/2)
            vec1[cum_param[iL]:cum_param[iL+1]] = vec1[cum_param[iL]:cum_param[iL+1]] + wo0[iwo]/np.sqrt(dim[iL]*dim[iL+1])/np.sqrt(dim[jL]*dim[jL+1]) * np.ones((dim[iL+1]*dim[iL],)) * np.sum(vec0[cum_param[jL]:cum_param[jL+1]])
        
    return vec1    

    
    
    
   




