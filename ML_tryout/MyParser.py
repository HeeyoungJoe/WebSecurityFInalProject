#
import pandas as pd
import numpy as np
from os import listdir
import time
#
class MyDocument:
    '''
    csv file into structure below

    - data: numpy array of csv content
    - X: 첫번째 열 제외
    - Y: 첫번째 열(문자)
    - size: size of numpy array
    - filename: filename 
    '''
    def __init__(self,data,filename):
        self.data=data
        #2D일 것을 가정
        
        self.X=self.data.iloc[:,1:]#dataframe
        #need work 여기는 M, B임 
        self.Y=self.data.iloc[:,0]#dataframe 
        #X의 행x열 tuple
        self.size=(len(self.X),len(self.X.columns))
        self.filename=filename



class MyParser:
    #make sure to print
    #--> parse begin
    #--> parse end
    #--> batch result
    '''
    path into structure below
    
    - file_history: list of MyDocument instances
    - batch_size=number of files
    - batch_count=1 -->rebatch 후에는 batch count 를 의미함
    - batch_data: one big numpy instance -->rebatch 후에는 하나의 batch row size를 의미함
    - batch_target: target numpy instance
    '''
    def __init__(self,path):
        self.path=path
        self.file_history=[]
        self.batch_data=None
        self.batch_target=None

    
    '''
    methods
    
    - parse(self): parse all data in path
    - parse_single(self,filename): parse single data (used in parse)
    - pad_row(self, n): pad columns of all files
    - rebatch(self,n): return a numpy object with all numpy matrices combined
    (1) 만약 새 batch 사이즈로 현재 있는 row 개수가 나누어 떨어지지 않으면 pad_row를 실행한다.
    (2) resize data&target
    (3) update batch size, batch count, remainder count
    '''
    def parse(self): 
        #딱 한 번만 맨 처음에 실행됨을 가정한다. -->그래야지 batch_count=1인 상황만 고려할 수 있다.
        #needwork
        #문서끼리 섞이면 안된다.
        for filename in listdir(self.path):
            self.parse_single(filename)

        #alarm
        print("\n|||Parsed Result:\n|||Document count:\t%d\n|||Batch_size:\t%d\n"%(len(self.file_history),self.batch_data.size))
        print("call (instance).batch_data for train input")
        print("list of numpy array of size: [.batch_count,.batch_size,.max_ft])\n")
        print("and (instance).batch_target for train output")
        print("list of numpy numpy array of size: [.batch_count,.batch_size,1])\n")
    
    def parse_single(self,filename):
        data=pd.read_csv(self.path+'/'+filename) #csv 파일 읽기
        parsed=MyDocument(data,filename) #MyDocument 오브젝트로 변환
        self.file_history.append(parsed) #변환 완성된 fd를 file_history에 저장

        #X
        x=parsed.X.to_numpy()
        if self.batch_data==None:
            self.batch_data=x
        else:
            self.batch_data=np.concatenate((self.batch_data,x),axis=0)
        print("\nbatch data in!...size%d\n"%x.size)
   
        #Y
        y=parsed.Y.to_numpy()
        if self.batch_target==None:
            self.batch_target=y
        else:
            self.batch_target=np.concatenate((self.batch_target,y),axis=0)
        print("\nbatch target in!...size %d\n"%y.size)
        
        #여기의 x, y는 각각 2D, 1D 행렬
        return x

'''
    def prepare_data(self,percentage,limit):
        #randomize
        
        #1. for all document splitted to M or B
        #need work

        dr,dc=self.batch_data.shape
        tr=self.batch_target.size
        self.batch_target=np.reshape(self.batch_target,(tr,1))
        
        dataset=np.concatenate((self.batch_data,self.batch_target),axis=1)
        np.random.shuffle(dataset)
        index=int(limit*percentage)
        testx=dataset[:index,:dc]
        testy=dataset[:index,dc:]
        trainx=dataset[index:limit,:dc]
        trainy=dataset[index:limit,dc:]
        return testx,testy,trainx,trainy
'''
def prepare_dataset(x,y,percentage,limit):
    #randomize
    
    #1. for all document splitted to M or B
    #need work

    dr,dc=x.shape
    tr=y.size
    y=np.reshape(y,(tr,1))
    
    dataset=np.concatenate((x,y),axis=1)
    np.random.shuffle(dataset)
    index=int(limit*percentage)
    testx=dataset[:index,:dc]
    testy=dataset[:index,dc:]
    trainx=dataset[index:limit,:dc]
    trainy=dataset[index:limit,dc:]
    return testx,testy,trainx,trainy   
#  
if __name__=='__main__':
    a=MyParser('./pdf2csv/testcsv')
    parse_start=time.time()
    a.parse()
    parse_end=time.time()

    print("\n\nTime spent:",parse_start-parse_end)

    print("\n\nSneak peek into data:\n")
    print(a.batch_data[:1,:5])
    print(a.batch_target[:5])
#
    testx,testy,trainx,trainy=prepare_dataset(a.batch_data,a.batch_target,0.3,100)



# %%
