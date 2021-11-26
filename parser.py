import pandas as pd
import numpy as np
from os import listdir

class MyDocument:
    #make sure to print
    #--> alarm parsing done

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
        self.X=self.data.loc[:,1:]#dataframe
        #need work 여기는 M, B임 
        self.Y=self.data.loc[:,0]#dataframe 
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
    
    - group of MyDocument instances
    - batch_size=number of files
    - batch_count=1
    - remainder_count=0
    - batch_data: one big numpy instance
    '''
    def __init__(self,path):
        self.path=path
        self.file_history=[]
        self.batch_count=1
        self.remainder_count=0
        batch_data=None
        batch_target=None
        max_ft=0
        batch_size=0
    
    '''
    methods
    
    - parse(self): parse all data in path
    - parse_single(self,filename): parse single data (used in parse)
    - pad(self, n): pad columns of all files
    - rebatch(self,new_batch_size): return a numpy object with all numpy matrices combined
    (1) concatenate
    (2) split in size n (default 1)
    (3) comment batch size, batch count, remainder count
    '''
    def parse(self): #딱 한 번만 맨 처음에 실행됨을 가정한다. -->그래야지 batch_count=1인 상황만 고려할 수 있다.
        for filename in listdir(self.path):
            self.parse_single(filename)
    
    def parse_single(self,filename):
        data=pd.read_csv(self.path+'/'+filename) #csv 파일 읽기
        parsed=MyDocument(data,filename) #MyDocument 오브젝트로 변환
        self.file_hisory.append(parsed) #변환 완성된 fd를 file_history에 저장

        #numpy dataset 만들기
        #default으로 하나의 numpy object 
        # --> 차원: ((파일 개수)*(행 개수))x(열 개수)
        # 모든 파일의 열 개수가 같다고 가정하고 있다. 
        #pad시 변경 -->필터링 필요
        #X
        if self.batch_data==None:
            self.batch_data=parsed.X.to_numpy
            #pad시 변경--> max 구하는 식 
            self.max_ft=parsed.size[1] #모든 파일의 열 개수가 같다고 가정하고 있다. 
        else:
            self.batch_data.append(parsed.X.to_numpy,axis=0)
        #Y
        #need work 
        #SVC allows target to be dataframe,RF는 y를 안쓴다. 다른 알고리즘은 모른다.
        if self.batch_target==None:
            self.batch_target=parsed.Y
        else:
            #need work
            #제대로 1:batch_size 인지 확인해야한다. 
            self.batch_target.append(parsed.Y)

        self.batch_size+=parsed.size[0]
    def pad_row(self,n): #pad rows when it is re-batched
        if self.batch_count==1:
            concatX=np.full((n,self.max_ft),-1)
            concatY=np.full((n,1),'B')
            self.batch_size+=n
            self.batch_data=np.concatenate(self.batch_data,concatX)
            

    def pad_column(self,n):
        pass
    
    def rebatch(self,new_batch_size):
        tmp=self.batch_size
       
        if tmp%new_batch_size!=0:
            self.pad_row(tmp%new_batch_size)
        
        #그냥 숫자 변수 챙기기
        self.batch_size=new_batch_size
        self.batch_count=int(tmp/new_batch_size)
        #데이터 변수 챙기기
        self.batch_data=np.resize(self.batch_data,(self.batch_count,new_batch_size,self.max_ft))
        #need work
        #y는 pandas dataframe이라 이렇게 resize하지 않는다. 
        self.batch_target=np.resize(self.batch_target,(self.batch_count,new_batch_size,1))       
    
        
