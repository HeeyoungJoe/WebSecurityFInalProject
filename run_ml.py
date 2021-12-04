import sys
sys.path.insert(1, './')
from ml import final,print_result
import pandas as pd
import pickle


############load model################
#linux
folderpath="/home/dori/hidost/build"

#testfile
filename=sys.argv[1]
testpath=folderpath+"/"+filename
#modelfile
modelpath=folderpath+"/finalized_model.sav"
#first, call model
model=pickle.load(open(modelpath,'rb'))
#second, call data
data=pd.read_csv(testpath).to_numpy()
x,y,name=column_slice(data) #ml.py
#last, run model
pred=model.predict(x)
print_result(pred,name)
