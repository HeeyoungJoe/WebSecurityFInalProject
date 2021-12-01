#%%
#tryout if these make better vectors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#tryout ensembling
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
#tryout clustering
from sklearn.cluster import KMeans
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
    #KFold-->might have to do it separately because it might come from keras
    #hen does make_2D
    #them does print_2D
    pass
def try_nn(x,y):
    #try both cnn and lstm networks
    #then does make_2D
    #then does print_2D
    pass
#%%
def try_simpleSVC(trainx,trainy,testx,testy): #returns SVC models
    #svm in different ways

    tr,tc=trainx.shape
    #2D then SVC
    tsne,pca=make_2D(np.concatenate((trainx,testx),axis=0))
    #tsne
    svc_tsne=SVC(kernel="rbf",degree=3,gamma="scale")
    svc_tsne.fit(tsne[:tr],trainy)
    #pca
    svc_pca=SVC(kernel="rbf",degree=3,gamma="scale")
    svc_pca.fit(pca[:tr],trainy)
    #result
    pred_tsne=svc_tsne.predict(tsne[tr:])
    pred_pca=svc_pca.predict(pca[tr:])
    
    #print("\n dimentions of prediction:",pred_tsne.shape)
    #acc1=accuracy_score(testy,pred_tsne)
    #acc2=accuracy_score(testy,pred_pca)
    
    #print("\n|||Accuracy w TSNE:\n",acc1)
    #print("\n|||Accuracy w PCA:\n",acc2)
    
    return svc_tsne,svc_pca 
        
#%%
#input: batched data of trainx and trainy
#output: SVC model that passed tsne, pca each 
def try_SVC(trainx,trainy): #returns SVC models
    #svm in different ways

    #2D then SVC
    tsne,pca=make_2D(trainx) #trained models
    #tsne
    svc_tsne=SVC(kernel="rbf",degree=3,gamma="scale")
    #pca
    svc_pca=SVC(kernel="rbf",degree=3,gamma="scale")
    
    #need work
    #get the results of tsne pca
    #choose between tsne-sfc with best results
    for input,target in zip(trainx,trainy):
        svc_tsne.fit(input,target)
        svc_pca.fit(input,target)
    return svc_tsne,svc_pca 

#%%

#tsne and pca
#input: batched data
#output: tsne, pca model
def make_2D(x):
    #x: batch count, batch size, feature size
    
    #t-sne
    print("\n\nTSNE preparing...")
    tsne=TSNE(n_components=2,learning_rate='auto',init='random')
    #pca
    print("\n\nPCA preparing...")
    pca=PCA(n_components=2)
    
    for batch in x:
        tsne.fit(x)
        pca.fit(x)
    
    return tsne,pca #모델 줌

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
    parse_start=time.time()
    a.parse()
    parse_end=time.time()
    a.rebatch(100)

    print("\n\nTime spent:",parse_start-parse_end)

    x=a.batch_data[108]
    y=a.batch_target[108]

    t,p=make_2D(x)
    print_2D('T-SNE',t,y)
    print_2D('PCA',p,y)
    
    index=1
    
    for dat,tar in zip(a.batch_data,a.batch_target):
        if index==3: break
        t,p=make_2D(dat)
        print_2D(t,tar)
        index=index+1
    
    

    #Trying small version of try_SVC-->try_simple
    #take x, y as train data
    #and the next batch as test data
    
    test_x=a.batch_data[109]
    test_y=a.batch_target[109]
    try_simpleSVC(x,y,test_x,test_y)

    #Training SVC for real
    print(a.batch_data.shape)

    testx,testy,trainx,trainy=a.set_splitter(0.3)
    tsneSVC,pcaSVC=try_SVC(trainx,trainy)
    
    predtsne=np.empty()
    predpca=np.empty()
    for input in testx:
        predtsne.append(tsneSVC.predict(input))
        predpca.append(pcaSVC.predict(input))

    #make results into one big np array
    #in order to feed it into accuracy_score
    tb,tr,tc=testy.shape
    testy=np.reshape((tb*tr,1),testy)
    predtsne=np.reshape((tb*tr,1),predtsne)
    predpca=np.reshape((tb*tr,1),predpca)
    print("\n|||Accuracy tsne:",accuracy_score(testy,predtsne))
    print("\n|||Accuracy pca:",accuracy_score(testy,predpca))


