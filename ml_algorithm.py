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
from MyParser import MyParser
import matplotlib.pyplot as plt

import time
import pandas as pd
import numpy as np

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

def try_simple(x,y): #returns SVC models
    #svm in different ways
    models={}
    
    #2D then SVC
    tsne,pca=make_2D(x)
    #tsne
    svc_tsne=SVC(kernel="rb",degree=3,gamma="scale")
    svc_tsne.fit(tsne,y)
    #pca
    svc_pca=SVC(kernel="rb",degree=3,gamma="scale")
    svc_pca.fit(pca,y)
    #result
    models["tsne_SVC"]=svc_tsne
    models["pca_SVC"]=svc_pca
    
    #plain SVC 
    size_x,size_y=x.shape
    svc_no=SVC(kernel="rbf",degree=size_x+1,gamma="scale")
    svc_no.fit(x,y)
    models["plain"]=svc_no
    
    return models
        

'''
해야하고 할 수 있는 것
(1) SVC를 어떻게 쓸 것인가
일단 2D로 옮겼을 때 직관적으로 구분이 더 가는건 pca이다
pca를 한 뒤에 svc를 돌린다
- 돌린 뒤 나뉜 그룹에 대해
-   각 그룹의 벡터의 태그 비율을 센다

(2) random forest 돌리기
- 돌린 뒤에 두 그룹이 나올 것이다
- 각 그룹의 벡터의 태그 비율을 센다

(3) k-means 돌리기 
- 그냥도 돌려보고
- pca한 뒤의 것도 돌려본다

고민되는 것
(1) NN- keras로 돌려야 할 것 같음. 코드 구조 관찰 요망/ sklearn이랑 한번 더 비교
(2) CNN이 잘 어울릴 것 같은데... 어떤 레이어를 몇 번 겹칠지 고민
선행 연구 결과는 없을까?


그리고 해야 하는 것
- libsvm 속 전처리 결과에 따라 달라지는지 비교할 수 있어야 한다. 
 
'''
 
#%%
def make_2D(x):
    #t-sne
    #t-sne simply returns
    print("\n\n=============T-SNE=============\n")
    tsne=TSNE(n_components=2,learning_rate='auto',init='random').fit_transform(x)
    print(tsne)
    #pca
    print("\n\n=============PCA=============\n")
    pca=PCA(n_components=2).fit_transform(x)
    print(pca)
    return tsne,pca #모델이 아니라 결과를 줌

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
#%%
if __name__=='__main__':
    a=MyParser('./pdf2csv/testcsv')
    parse_start=time.time()
    a.parse()
    parse_end=time.time()
    a.rebatch(100)

    print("\n\nTime spent:",parse_start-parse_end)

    x=a.batch_data[108]
    y=a.batch_target[108]
#%%
    t,p=make_2D(x)
    print_2D('T-SNE',t,y)
    print_2D('PCA',p,y)
    # index=1
   
    for dat,tar in zip(a.batch_data,a.batch_target):
        if index==3: break
        t,p=make_2D(dat)
        print_2D(t,tar)
        index=index+1
    
    
#%%
    
    #take x, y as train data
    #and the next batch as test data
    SVC_models=try_nn(x,y)
    test_x=a.batch_data[109]
    test_y=a.batch_target[109]
    
    #try test data
    pred_plainSVC=SVC_models['plain'].predict(test_x)
    pred_tsneSVC=SVC_models['tsne_SVC'].predict(test_x)
    pred_pcaSVC=SVC_models['pca_SVC'].predict(test_x)

    #calculate accuracy
    acc_plainSVC=accuracy_score(test_y,pred_plainSVC)
    acc_tsneSVC=accuracy_score(test_y,pred_tsneSVC)
    acc_pcaSVC=accuracy_score(test_y,pred_pcaSVC)
    print("\nAccuracy of SVC\n")  
    print("|||plain:",acc_plainSVC,"\n")
    print("|||tsne-SVC:",acc_tsneSVC,"\n")
    print("|||pca-SVC:",acc_pcaSVC,"\n")

# %%
