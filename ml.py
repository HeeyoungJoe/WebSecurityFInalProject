#%%
import pandas as pd
import numpy as np
from os import listdir
import time

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
#%%
def parse(testpath,trainpath,limit): 
    #test set과 train set을 모두 파싱한다
    #둘의 열 수가 동일한지 확인한다. 
    data=pd.read_csv(trainpath)
    test=pd.read_csv(testpath)
    
    traindata=data.to_numpy()
    np.random.shuffle(traindata)
    traindata=traindata[:limit]
    trainX=traindata[:,1:]#dataframe
    trainY=traindata[:,0]#dataframe 
    
    testdata=test.to_numpy()
    #지금 testdata가 따로 없어서 들어가는 곳
    np.random.shuffle(testdata)
    testdata=testdata[:10]
    #여기까지
    testX=testdata[:,1:]#dataframe
    testY=testdata[:,0]#dataframe 

    
    tr_r,tr_c=trainX.shape
    t_r,t_c=testX.shape
    if tr_c!=t_c:
        print("The number of features don't match!")
        return -1
    

    #cocnatenate with test data
    totalX=np.concatenate((trainX,testX),axis=0)
    totalY=np.concatenate((trainY,testY),axis=0)
    print("\n|||size!\t",totalX.shape,"\t",totalY.shape)
    return tr_r,totalX,totalY


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
    
def try_simple_SVC(x,y):
    svc=SVC(kernel="rbf",degree=3,gamma="scale")
    svc.fit(x,y)  
    print("\n|||SVC well fit")
    return svc

def try_simple_rf(x,y):
    rf=RandomForestClassifier(n_estimators=5,random_state=0)
    rf.fit(x,y)
    return rf
#%%
def runRF(trainpath,testpath):
    limit=100
    count,X,Y=parse(testpath,trainpath,limit)
    print("\n|||Count:",count)
    
    #without 2D
    pltrainX=X[:count]
    pltrainY=Y[:count]
    pltestX=X[count:]
    pltestY=Y[count:]
    rf=try_simple_rf(pltrainX,pltrainY)
    plpred=rf.predict(pltestX)
    print("\n|||Accuracy on RF:\t",accuracy_score(pltestY,plpred))

def runSVC(trainpath,testpath):
    limit=100
    count,X,Y=parse(testpath,trainpath,limit)
    print("\n|||Count:",count)
    tsne_x,pca_x=make_2D(X)
    print("\n|||make 2d result",tsne_x.shape,pca_x.shape)
    #cut train and test
    train_tsne_x=tsne_x[:count]
    train_pca_x=pca_x[:count]
    test_tsne_x=tsne_x[count:]
    test_pca_x=pca_x[count:]
    trainY=Y[:count]
    testY=Y[count:]
    
    tsneSVC=try_simple_SVC(train_tsne_x,trainY)
    pcaSVC=try_simple_SVC(train_pca_x,trainY)
    
    pred_tsne=tsneSVC.predict(test_tsne_x)
    pred_pca=pcaSVC.predict(test_pca_x)
    
    print("\n|||prediction done")
    print("\n|||pred_tsne result",pred_tsne.shape)
    print("\n|||pred_pca result",pred_pca.shape)
    acc1=accuracy_score(testY,pred_tsne)
    acc2=accuracy_score(testY,pred_pca)
    
    print("\n|||accucacy for \n|||tsne \t%f\n|||pca \t%f\n"%(acc1,acc2))
#%%
if __name__=="__main__":
    testpath="./pdf2csv/output.csv"
    trainpath="./pdf2csv/output.csv"
    print("~~~~~~~~~~~~~~~~~~~~~~~~~SVM~~~~~~~~~~~~~~~~~~~~~~~~")
    runSVC(trainpath,testpath)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~RF~~~~~~~~~~~~~~~~~~~~~~~~")
    runRF(trainpath,testpath)