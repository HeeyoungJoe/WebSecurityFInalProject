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

def parse_train(trainpath,ratio,limit): 
    #test set과 train set을 모두 파싱한다
    #둘의 열 수가 동일한지 확인한다. 
    data=pd.read_csv(trainpath).to_numpy()
    
    np.random.shuffle(data)
    data=data[:limit]
    X=data[:,1:]#dataframe
    Y=data[:,0]#dataframe 
    
    r,c=X.shape
    count=int((1-ratio)*r)

    return count,totalX,totalY

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
def slice_data(x,y,count):
    trainX=x[:count]
    trainY=y[:count]
    testX=x[count:]
    testY=y[count:]
    return trainX,trainY,testX,testY
def runRF(trainpath,testpath):
    limit=100
    if testpath==None:
        count,X,Y=parse_train(trainpath,0.3,limit)
    else:
        count,X,Y=parse(trainpath,testpath,limit)
    #without 2D
    trainX,trainY,testX,testY=slice_data(X,Y,count)
    rf=try_simple_rf(trainX,trainY)
    pred=rf.predict(testX)
    if testpath==None:
        print("\n|||Accuracy on RF:\t",accuracy_score(testY,pred))
    
    return pred

def runSVC(trainpath,testpath):
    limit=100
    if testpath==None:
        count,X,Y=parse_train(trainpath,0.3,limit)
    else:
        count,X,Y=parse(testpath,trainpath,limit)
    
    print("\n|||Count:",count)
    tsne_x,pca_x=make_2D(X)
    print("\n|||make 2d result",tsne_x.shape,pca_x.shape)
    #cut train and test
    train_tsne_x,train_tsne_y,test_tsne_x,test_tsne_y=slice_data(tsne_x,Y,count)
    train_pca_x,train_pca_y,test_pca_x,test_pca_y=slice_data(pca_x,Y,count)

    
    tsneSVC=try_simple_SVC(train_tsne_x,train_tsne_y)
    pcaSVC=try_simple_SVC(train_pca_x,train_pca_y)
    
    pred_tsne=tsneSVC.predict(test_tsne_x)
    pred_pca=pcaSVC.predict(test_pca_x)
    
    print("\n|||prediction done")
    print("\n|||pred_tsne result",pred_tsne.shape)
    print("\n|||pred_pca result",pred_pca.shape)
    
    if testpath==None:
        acc1=accuracy_score(test_tsne_y,pred_tsne)
        acc2=accuracy_score(test_pca_y,pred_pca)
        print("\n|||accucacy for \n|||tsne \t%f\n|||pca \t%f\n"%(acc1,acc2))
    
    return pred_tsne,pred_pca
#%%
if __name__=="__main__":
    #debug
    testpath=None
    trainpath="./pdf2csv/output.csv"#훈련
    print("~~~~~~~~~~~~~~~~~~~~~~~~~SVM~~~~~~~~~~~~~~~~~~~~~~~~")
    runSVC(trainpath,testpath) #accuacy print
    print("~~~~~~~~~~~~~~~~~~~~~~~~~RF~~~~~~~~~~~~~~~~~~~~~~~~")
    runRF(trainpath,testpath) #accucacy print
    
    #release
    testpath="#"
    print("~~~~~~~~~~~~~~~~~~~~~~~~~SVM~~~~~~~~~~~~~~~~~~~~~~~~")
    pred_tsne,pred_pca=runSVC(trainpath,testpath) #accuacy print
    print("~~~~~~~~~~~~~~~~~~~~~~~~~RF~~~~~~~~~~~~~~~~~~~~~~~~")
    predrf=runRF(trainpath,testpath) #accucacy print
    
    
    #runreleaseSVC() #추정치 array 주는 거