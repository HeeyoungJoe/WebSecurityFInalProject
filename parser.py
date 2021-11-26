import pandas as pd
import numpy as np

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
        self.Y=self.data.loc[:,0]#dataframe
        #행x열 tuple
        self.size=(len(self.X),len(self.X.loc[0]))
        self.filename=filename

    '''
    method
    - to_vector(self): dataframe X, Y to numpy array 
    returns npX, npY
    '''

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
        batch_size=0
    
    '''
    methods
    
    - parse(self): parse all data in path
    - parse_single(self,filename): parse single data (used in parse)
    - pad(self, n): pad columns of all files
    - batch(self,n): return a numpy object with all numpy matrices combined
    (1) concatenate
    (2) split in size n (default 1)
    (3) comment batch size, batch count, remainder count
    '''
    def parse(self):
        for filename in listdir(self.path):
            self.batch_size+=1
            self.parse_single(filename)
    
    def parse_single(self,filename):
        data=pd.read_csv(self.path+'/'+filename)
        parsed=MyDocument(data,filename)
        self.file_hisory.append(parsed)

        #numpy dataset 만들기
        if self.batch_data==None:
            self.batch_data=parsed.X.to_numpy
        else:
            self.batch_data.append(parsed.X.to_numpy,axis=0)
        if self.batch_target==None:
            self.batch_target=parsed.Y.to_numpy
        else:
            self.batch_target.append(parsed.Y.to_numpy,axis=0)
