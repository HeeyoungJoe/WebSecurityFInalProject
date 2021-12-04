import sys
sys.path.insert(1, './')
from ml import final,print_result
import pandas as pd
import pickle

trainpath="./pdf2csv/output.csv"
testpath=None
rf_final,name=final(trainpath)#모델

filename="finalized_model.sav"
pickle.dump(rf_final,open(filename,'wb'))
#load model later
#model=pickle.load(open(filename,'rb'))
