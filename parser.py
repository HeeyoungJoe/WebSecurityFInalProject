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
        self.X=self.data.loc[:,1:]
        self.Y=self.data.loc[:,0]
        #행x열 tuple
        self.size=(len(self.X),len(self.X.loc[0]))
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
        self.path=pathself
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