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
    return tsne,pca

def print_2D(x,y):
    '''
    code from:
    https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
    '''
    fg=plt.figure(figsize=(10,10))
    ax=fg.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1',fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)

    targets=['M','B']
    colors=['r','b']
    #need work
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
    a.rebatch(20)

    print("\n\nTime spent:",parse_start-parse_end)

    index=1
    for dat,tar in zip(a.batch_data,a.batch_target):
        if index==3: break
        t,p=make_2D(dat)
        print_2D(t,tar)
        index=index+1

    #tryout ml

    a.rebatch(10000)

