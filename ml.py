 
from numpy.typing import _32Bit
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

limit=1000000
percent=0.3
 
def column_slice(data):
    X=data[:,1:-1]
    Y=data[:,0] 
    name=data[:,-1]
    return X,Y,name

def row_slice(x,y,percent):
    count=int(y.size*percent)
    trainX=x[:count]
    trainY=y[:count]
    testX=x[count:]
    testY=y[count:]
    return trainX,trainY,testX,testY

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
    r,c=traindata.shape
    if limit<r:
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
    
    return X,Y,name
 
#tsne and pca
#input: batched data
#output: tsne, pca model
def make_2D(x):
    #x: row, feature size
    
    r,c=x.shape
    if r==0:
        x=np.reshape(x,(1,x.size))
    
    #t-sne
    tsne=TSNE(n_components=2,learning_rate='auto', init='random')
    #pca
    pca=PCA(n_components=2)
    
    result_t=tsne.fit_transform(x)
    result_p=pca.fit_transform(x)
    
    print("\n|||The reduced results show shape of:",result_t.shape,result_p.shape)
    print(result_p)
    print(result_t)
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

    data=pd.DataFrame(np.concatenate((x,y),axis=1),columns=['pc1','pc2','target'])

    for target, color in zip(targets,colors):
        indices=data['target']==target
        ax.scatter(data.loc[indices,'pc1'],data.loc[indices,'pc2'],c=color,s=50)
    ax.legend(targets)
    ax.grid()
    
def try_simple_SVC(x,y):
    svc=SVC(kernel="rbf",degree=3,gamma="scale")
    svc.fit(x,y)  
    return svc

def try_simple_rf(x,y):
    rf=RandomForestClassifier(n_estimators=5,random_state=0)
    rf.fit(x,y)
    return rf
 

def runRF(trainpath,testpath):

    if testpath==None:
        X,Y,name=parse_train(trainpath,0.3,limit)
    else:
        tr_r,X,Y,name=parse(trainpath,testpath,limit)
    #without 2D
    trainX,trainY,testX,testY=row_slice(X,Y,percent)
    rf=try_simple_rf(trainX,trainY)
    pred=rf.predict(testX)
    
    if testpath==None:
        print("\n|||Accuracy for \n|||RF:\t",accuracy_score(testY,pred))
    
    return rf,pred,name


  
def runSVC(trainpath,testpath):

    if testpath==None:
        X,Y,name=parse_train(trainpath,0.3,limit)
    else:
        tr_r,X,Y,name=parse(testpath,trainpath,limit)
    #reduce x to 2-dim
    tsne_x,pca_x=make_2D(X)
    
    #cut train and test
    train_tsne_x,train_tsne_y,test_tsne_x,test_tsne_y=row_slice(tsne_x,Y,percent)
    train_pca_x,train_pca_y,test_pca_x,test_pca_y=row_slice(pca_x,Y,percent)

    
    tsneSVC=try_simple_SVC(train_tsne_x,train_tsne_y)
    pcaSVC=try_simple_SVC(train_pca_x,train_pca_y)
    
    pred_tsne=tsneSVC.predict(test_tsne_x)
    pred_pca=pcaSVC.predict(test_pca_x)
    
    if testpath==None:
        acc1=accuracy_score(test_tsne_y,pred_tsne)
        acc2=accuracy_score(test_pca_y,pred_pca)
        print("\n|||Accucacy for \n|||tsne \t%f\n|||pca \t%f\n"%(acc1,acc2))
    
    return pred_tsne,pred_pca,name

def final(trainpath):
    data=pd.read_csv(trainpath).to_numpy()
    trainX=data[:,1:-1]
    trainY=data[:,0]
    name=data[:,-1]
 
    rf=try_simple_rf(trainX,trainY)
    
    return rf,name

def print_result(pred,name):
    for pr,na in zip(pred,name):
        print("\n|||Doc[",na,"]\tis classified as\t[",pr,"]")

 
if __name__=="__main__":
    #모델들의 정확도를 비교하기 위한 코드
    #test dataset이 별도의 경로에 저장되어있지 않아
    #rain path만 지정해준다.
    #run~라는 함수들에서 testpath가 None으로 되어있다면 train dataset을
    #분할해서 accuracy를 저장해준다. 
    testpath=None#fix it as None
    trainpath="./chang_kim/output3.csv"#훈련 데이터
    print("~~~~~~~~~~~~~~~~~~~~~~~~~SVM~~~~~~~~~~~~~~~~~~~~~~~~")
    pred_tsne,pred_pca,name=runSVC(trainpath,testpath) #accuacy 
    print("~~~~~~~~~~~~~~~~~~~~~~~~~RF~~~~~~~~~~~~~~~~~~~~~~~~")
    model,predrf,name=runRF(trainpath,testpath) #accucacy print 

 
    
    #웹용 모델 저장
    #output del은 웹에서 실습을 하기 위해 6개의 row를 제외한 것
    trainpath="./chang_kim/output_del.csv"
    rf_model,name=final(trainpath)#모델
    
    filename="finalized_model.sav"
    pickle.dump(rf_model,open(filename,'wb'))

    