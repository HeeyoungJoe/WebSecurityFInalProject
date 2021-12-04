import sys
sys.path.insert(1, './')
from ml import final,print_result
import pandas as pd

trainpath="./pdf2csv/output.csv"
testpath=None
rf_final,name=final(trainpath)#모델
testdata=pd.read_csv(testpath).to_numpy()
pred=rf_final.predict(column_slice(testdata))
print_result(pred,name)