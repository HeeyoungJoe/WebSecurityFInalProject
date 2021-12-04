#%%
#tryout if these make better vectors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#tryout ensembling
from sklearn.ensemble import RandomForestClassifier

#tryout machine learning algorithm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
#display result
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import time
import pandas as pd
import numpy as np

#%% 
#update
from MyParser import MyParser

#%%
def try_ensemblers(x,y):
    #random forest
    #check how many featuers rf can take
    #hen does make_2D
    #them does print_2D
    pass

        
#%%
#input: simple data trainx and trainy
#output: SVC model that passed tsne, pca each 
# def try_SVC(trainx,trainy): #returns SVC models
#     #svm in different ways

#     #2D then SVC
#     tsne,pca=make_2D(trainx) #결과
#     #tsne
#     svc_tsne=SVC(kernel="rbf",degree=3,gamma="scale")
#     #pca
#     svc_pca=SVC(kernel="rbf",degree=3,gamma="scale")
    
#     #need work
#     #get the results of tsne pca
#     #choose between tsne-svc with best results
#     svc_tsne.fit(tsne,trainy)
#     svc_pca.fit(pca,trainy)
#     return svc_tsne,svc_pca #모델 줌

def try_simple_SVC(x,y,ratio):
    tsne,pca=make_2D(x)
    r,c=tsne.shape
    plane=r*ratio #train ratio
    #tsne
    svc_tsne=SVC(kernel="rbf",degree=3,gamma="scale")
    #pca
    svc_pca=SVC(kernel="rbf",degree=3,gamma="scale")
    
    svc_tsne.fit(tsne[:plane],y[:plane])
    svc_pca.fit(pca[:plane],y[:plane])
    
    tsnep=svc_tsne.predict(tsne[plane:])
    pcap=svc_pca.predict(pca[plane:])
    
    return tsnep,pcap
        
        
#%%

#tsne and pca
#input: batched data
#output: tsne, pca model
def make_2D(x):
    #x: row, feature size
    
    #t-sne
    print("\n\nTSNE preparing...")
    tsne=TSNE(n_components=2,learning_rate='auto',init='random')
    #pca
    print("\n\nPCA preparing...")
    pca=PCA(n_components=2)
    
    result_t=tsne.fit_transform(x)
    result_p=pca.fit_transform(x)
    
    return result_t,result_p #결과 줌

def print_2D(title,x,y):
    '''
    code from:
    https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
    '''
    fg=plt.figure(figsize=(10,10))
    ax=fg.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1',fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title(title, fontsize = 20)

    targets=['M','B']
    colors=['r','b']
    
    if y.shape!=(len(y),1):
        y=y.reshape((len(y),1))
    
    print("\n\nshape of x:",x.shape,"\nshape of y:",y.shape)

    data=pd.DataFrame(np.concatenate((x,y),axis=1),columns=['pc1','pc2','target'])

    for target, color in zip(targets,colors):
        indices=data['target']==target
        ax.scatter(data.loc[indices,'pc1'],data.loc[indices,'pc2'],c=color,s=50)
    ax.legend(targets)
    ax.grid()

if __name__=='__main__':
    a=MyParser('./pdf2csv/testcsv')
    a.parse()

    #Training SVC for real
    testx,testy,trainx,trainy=a.prepare_data(0.3,500)
    tsneSVC,pcaSVC=try_SVC(trainx,trainy) #gives model
 

    #make results into one big np array
    #in order to feed it into accuracy_score
    tsne,pca=make_2D(testx)
    print("\n|||Size of testx",testx.shape)
    predtsne=tsneSVC.predict(tsne)
    predpca=pcaSVC.predict(pca)
    print("\n|||Accuracy tsne:",accuracy_score(testy,predtsne))
    print("\n|||Accuracy pca:",accuracy_score(testy,predpca))
    #accuracy 되게 낮은데 그럴 수밖에 없음! testx가 같이 tsne pca 화 된게 아님


