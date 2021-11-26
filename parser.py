import pandas as pd
import numpy as np

class MyDocument:
    #make sure to print
    #--> alarm parsing done

    '''
    csv file into structure below

    - data: numpy array of csv content
    - size: size of numpy array
    - filename: filename 
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

    '''
    methods
    
    - pad(self, n): pad columns of all files
    - batch(self,n): return a numpy object with all numpy matrices combined
    (1) concatenate
    (2) split in size n (default 1)
    (3) comment batch size, batch count, remainder count
'''