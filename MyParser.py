import pandas as pd
import numpy as np
from os import listdir
import time
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
        self.batch_count=1
        self.batch_data=None
        self.batch_target=None
        self.max_ft=0
        self.batch_size=0
    
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
    def parse(self): #딱 한 번만 맨 처음에 실행됨을 가정한다. -->그래야지 batch_count=1인 상황만 고려할 수 있다.
        for filename in listdir(self.path):
            self.parse_single(filename)
        #행*열 사이즈다. 
        #parse는 시작시 한번만 이뤄지므로 batch_count는 이 함수가 실행될 때 항상 1이다.
        #따라서 batch_size는 모든 파일 파싱이 이뤄진 후 딱 한번만 실행되면 된다. 
        self.batch_size=self.batch_data.size 
        
        #feature의 개수는 행*열 사이즈인 batch_size에서 행 사이즈와 같은 batch_target의 사이즈를 나누면 된다.
        #딱 처음에 한 번만 구해주면 된다. 
        self.max_ft=int(self.batch_size/self.batch_target.size)
        '''
        #batch data 와 batch target size 조정
        self.batch_data.resize((self.batch_count,int(self.batch_size/self.max_ft),self.max_ft))
        self.batch_target.resize((self.batch_count,self.batch_size,1))
        '''
        #alarm
        print("\n|||Parsed Result:\n|||Document count:\t%d\n|||Batch_size:\t%d\n|||Batch_count:\t%d\n"%(len(self.file_history),self.batch_data.size,self.batch_count))
        print("call (instance).batch_data for train input(numpy array of size: [.batch_count,.batch_size,.max_ft])\n and (instance).batch_target for train output (numpy array of size: [.batch_count,.batch_size,1])")
    def parse_single(self,filename):
        data=pd.read_csv(self.path+'/'+filename) #csv 파일 읽기
        parsed=MyDocument(data,filename) #MyDocument 오브젝트로 변환
        self.file_history.append(parsed) #변환 완성된 fd를 file_history에 저장

        #numpy dataset 만들기
        # 모든 파일의 열 개수가 같다고 가정하고 있다. 
        #X
        if self.batch_data==None:
            self.batch_data=parsed.X.to_numpy()
            print("\nFirst batch data in!...size%d\n"%self.batch_data.size)
            #한 csv 파일에는 여러 pdf 파일에 대한 벡터가 있다. 그래서 이미 2D ndarray일 것이다.        
        else:
            self.batch_data=np.concatenate((self.batch_data,parsed.X.to_numpy))
              
        #Y
        #string numpy array 가능하다.
        if self.batch_target==None:
            self.batch_target=parsed.Y.to_numpy()
            print("\nFirst batch target in!...size %d\n"%self.batch_target.size)
        else:
            self.batch_target=np.concatenate((self.batch_target,parsed.Y.to_numpy()))


    def pad_row(self,old_size,new_size): 
        #pad rows when it is re-batched
        #근데 padding 했을때 malicous benign이 애매해져서 그냥 자르는 것으로 할 생각. 
        '''
        if self.batch_count==1:
            concatX=np.full((n,self.max_ft),-1)
            concatY=np.full((n,1),'B')
            self.batch_size+=n
            self.batch_data=np.concatenate(self.batch_data,concatX)
        '''
        cut=old_size%new_size
        if self.batch_count==1: #one big batch
            self.batch_data=self.batch_data[:self.batch_data.size-cut]
            self.batch_target=self.batch_target[:self.batch_target.size-cut]
        elif self.batch_count>1:#several batch
            leave=int(old_size/new_size)
            self.batch_data=self.batch_data[:leave,:,:]
            self.batch_target=self.batch_target[:leave,:,:]

    
    def rebatch(self,new_batch_size):
        
        if self.batch_size%new_batch_size!=0:
            self.pad_row(self.batch_size,new_batch_size)
        
        #그냥 숫자 변수 챙기기
        tmp=self.batch_size
        self.batch_size=new_batch_size
        self.batch_count=int(tmp/new_batch_size)
        #데이터 변수 챙기기
        self.batch_data=np.resize(self.batch_data,(self.batch_count,self.batch_size,self.max_ft)) #need work 
        self.batch_target=np.resize(self.batch_target,(self.batch_count,self.batch_size))    

        print("\n|||Batch size updated: [%d,%d,%d]"%(self.batch_count,self.batch_size,self.max_ft))   
    
    def printSize(self):
        print("\n\n[Let's look about its size]\n")
        print("|||Batch_data:",self.batch_data.shape,"\n")
        print("|||Batch_target:",self.batch_count,"\n")
 
'''     
if __name__=='__main__':
    a=MyParser('./pdf2csv/testcsv')
    parse_start=time.time()
    a.parse()
    parse_end=time.time()

    print("\n\nTime spent:",parse_start-parse_end)

    print("\n\nSneak peek into data:\n")
    print(a.batch_data[:1,:5])
    print(a.batch_target[:5])

    a.rebatch(10000)

'''