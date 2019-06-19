
# coding: utf-8

# # FISHER SCORE

# In[ ]:


def fisher_score(x, y):
    mean = np.mean(x, axis=0)
    classes = np.unique(y)
    sS = 0
    sD = 0
    Nk = []
    
    meanD = []
    varD = []
    
    for k in classes:
        elements = []
        s = 0
        for i in range(0, len(y)):
            if(y[i] == k):
                s += 1
                elements.append(x[i])
        meanD.append(np.mean(elements, axis=0))
        varD.append(np.var(elements, axis=0))
        Nk.append(s)
    
    for k in range(0, len(classes)):
        sS += (Nk[k] * ((meanD[k] - mean)**2))
        sD += (Nk[k] * varD[k])
    return sS/sD


# # PCA

# In[1]:


from scipy.linalg import svd 

def compute(x):
    mean = np.mean(x, axis = 0)
    cov = np.cov(x)
    
    S, U, V = svd(cov)
    M = S @ U @ np.tranpose(V)
    P = np.transpose(U)
    return {'S': S, 'U': U, 'V': V, 'M': M, 'P': P}

def transform(x, pca_result, dim=2):

