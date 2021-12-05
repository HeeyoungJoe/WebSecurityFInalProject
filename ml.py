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
import pickle
#%%
def column_slice(data):
    X=data[:,1:-1]
    Y=data[:,0] 
    name=data[:,-1]
    return X,Y,name
def parse(testpath,trainpath,limit): 
    #test set과 train set을 모두 파싱한다
    #둘의 열 수가 동일한지 확인한다. 
    data=pd.read_csv(trainpath)
    test=pd.read_csv(testpath)
    
    ###########TRAIN############
    #numpy로 바꾸고
    traindata=data.to_numpy()
    np.random.shuffle(traindata)
    
    #자르기
    traindata=traindata[:limit]
    trainX,trainY,trainname=column_slice(traindata)
    
    ###########TEST##############
    #numpy로 바꾸고
    testdata=test.to_numpy()
    testX,testY,testname=column_slice(testdata)
    
    tr_r,tr_c=trainX.shape
    t_r,t_c=testX.shape
    if tr_c!=t_c:
        print("The number of features don't match!")
        return -1
    

    #cocnatenate with test data
    totalX=np.concatenate((trainX,testX),axis=0)
    totalY=np.concatenate((trainY,testY),axis=0)
    totalName=np.concatenate((trainname,testname),axis=0)
    return tr_r,totalX,totalY,totalName

def parse_train(trainpath,ratio,limit): 
    #test set과 train set을 모두 파싱한다
    #둘의 열 수가 동일한지 확인한다. 
    data=pd.read_csv(trainpath)
    data=data.to_numpy()
    
    if data.ndim<=1:
        return data[0],data[1:-1],data[-1]
    
    np.random.shuffle(data)
    if data.ndim==2:
        r,c=data.shape
        if r>limit:
            data=data[:limit]
    
    X,Y,name=column_slice(data)
    
    count=int((1-ratio)*r)
    print("\n******************************")
    print("\nCheck Parsed Result:")
    print("\nX:",X.shape,"\tY:",Y.shape)
    print("\n******************************")
    return count,X,Y,name

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
def row_slice(x,y,count):
    trainX=x[:count]
    trainY=y[:count]
    testX=x[count:]
    testY=y[count:]
    return trainX,trainY,testX,testY
def runRF(trainpath,testpath):
    limit=1000
    if testpath==None:
        count,X,Y,name=parse_train(trainpath,0.3,limit)
    else:
        count,X,Y,name=parse(trainpath,testpath,limit)
    #without 2D
    trainX,trainY,testX,testY=row_slice(X,Y,count)
    rf=try_simple_rf(trainX,trainY)
    
    if testpath==None:
        print("\n|||Accuracy on RF:\t",accuracy_score(testY,pred))
    
    return model,pred,name

def final(trainpath):
    data=pd.read_csv(trainpath).to_numpy()
    trainX=data[:,1:-1]
    trainY=data[:,0]
    name=data[:,-1]
    rf=try_simple_rf(trainX,trainY)
    
    return rf,name
    
def runSVC(trainpath,testpath):
    limit=1000
    if testpath==None:
        count,X,Y,name=parse_train(trainpath,0.3,limit)
    else:
        count,X,Y,name=parse(testpath,trainpath,limit)
    
    print("\n|||Count:",count)
    tsne_x,pca_x=make_2D(X)
    print("\n|||make 2d result",tsne_x.shape,pca_x.shape)
    #cut train and test
    train_tsne_x,train_tsne_y,test_tsne_x,test_tsne_y=row_slice(tsne_x,Y,count)
    train_pca_x,train_pca_y,test_pca_x,test_pca_y=row_slice(pca_x,Y,count)

    
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
    
    return pred_tsne,pred_pca,name

def print_result(pred,name):
    for pr,na in zip(pred,name):
        print("\n|||Doc[",na,"]\tis classified as\t[",pr,"]")
#%%
if __name__=="__main__":
    #debug
    testpath=None#fix it as None
    trainpath="./chang_kim/output_test.csv"#훈련
    print("~~~~~~~~~~~~~~~~~~~~~~~~~SVM~~~~~~~~~~~~~~~~~~~~~~~~")
    pred_tsne,pred_pca,name=runSVC(trainpath,testpath) #accuacy print
    print("~~~~~~~~~~~~~~~~~~~~~~~~~RF~~~~~~~~~~~~~~~~~~~~~~~~")
    predrf,name=runRF(trainpath,testpath) #accucacy print
    
    
    print_result(pred_tsne,name)
    print_result(pred_pca,name)
    print_result(predrf,name)
    
    #웹용_
  
        
    trainpath="./pdf2csv/output.csv"
    rf_model,name=final(trainpath)#모델

    filename="finalized_model.sav"
    pickle.dump(rf_model,open(filename,'wb'))