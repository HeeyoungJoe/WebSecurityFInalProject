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
        self.batch_data=[]
        self.batch_target=[]

    
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
        print("\n|||Parsed Result:\n|||Document count:\t%d\n|||Batch_size:\t%d\n"%(len(self.file_history),len(self.batch_data.size)))
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
        self.batch_data.append(x)
        print("\nbatch data in!...size%d\n"%x.size)
   
        #Y
        y=parsed.Y.to_numpy()
        self.batch_target.append(y)
        print("\nbatch target in!...size %d\n"%y.size)
        
        #여기의 x, y는 각각 2D, 1D 행렬
        return x

    #need work
    
    '''
    randomly pick train data 
    things to make sure
    1. randomize
    2. split to test and train
    per document
    
    input: path
    output: set of (traindata, testdata)
    
    function
    
    -randomize set per document
    -set splitter -->but returning sets
    
    do I need rebatch? -->아니 필요 없어. 그냥 자를거야. 
    
    
    '''

    def prepare_data(self,percentage):
        #randomize
        
        #1. for all document splitted to M or B
        #need work

        #get data
        dataset=[]
        for document in self.batch_data: #document is 2D
            #get M part and B part
            #random pick from both 
            #prepare test and train set
            np.random.shuffle(document)
            index=int(document.size[0]*percentage)
            testx=self.batch_data[:index]
            testy=self.batch_target[:index]
            trainx=self.batch_data[index:]
            trainy=self.batch_target[index:]
            dataset.append((testx,testy,trainx,trainy))
        return dataset
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