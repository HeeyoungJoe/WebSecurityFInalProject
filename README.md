<h1>testcsv 폴더</h1>
역할: column수에 따른 ml 성능 확인<br>
<h2>output.csv</h2>
역할: ML 모델을 훈련시킬때 사용될 파일<br>
(1000개 이상의 파일들이 가지고 있는 features)<br>
<h2>output2.csv</h2>
역할: output.csv와 다른 수의 column을 가지고 있는 csv파일<br>
(2000개 이상의 파일들이 가지고 있는 features)<br>
<h2>output3.csv</h2>
역할: output.csv와 다른 수의 column을 가지고 있는 csv파일<br>
(500개 이상의 파일들이 가지고 있는 features)<br>

<h1>build폴더</h1>
<h2>bpdfs.txt & mpdfs.txt</h2>
역할: training file을 만들때 사용된 전체 파일의 경로 텍스트 파일<br>
<h2>pdfs.txt & test.txt</h2>
역할: 사용자가 업로드할 파일들과 hidost 실행을 위한 test파일의 경로 텍스트 파일<br>
<h2>cached-bpdfs.txt & cached-mpdfs.txt</h2>
역할: training file을 만들때 사용된 전체 파일을 캐싱한 텍스트 파일<br>
<h2>cached-pdfs.txt & cached-test.txt</h2>
역할: 사용자가 업로드할 파일들과 hidost 실행을 위한 test파일을 캐싱한 경로 텍스트 파일<br>
<h2>test.txt</h2>
역할: PDF parsing에서 hidost를 실행시킬 때 benign 파일로써 사용될 파일<br>
<h2>output.csv</h2>
역할: ML 모델을 훈련시킬때 사용될 파일 (test를 위해 6개의 파일들을 빼놓은 csv파일)<br>
<h2>libsvm_to_csv.py</h2>
역할: training csv 파일을 만들 때 사용된 python 코드<br>
함수:<br>


<h2>libsvm_to_csv2.py</h2>
역할: 사용자가 업로드한 파일을 csv파일로 parsing하기 위해 사용된 python 코드<br>
<h2>pdf_to_libsvm.py</h2>
역할: 사용자가 파일을 업로드한 후 hidost 실행과 csv 파일 생성을 위한 쉘 스크립트<br>
<h2>ml.py</h2>
역할: Machine learning classifier algorithm을 실행하고 비교하기 위해 만든 파일<br>

함수:<br>

<h4>Pre-processing>
<h5>parse(traindata_path,testdata_path,limit)</h5><br>
input: <br>
traindata_path (str): path to traindata csv file
testdata_path (str): path to testdata csv file
limit (int): cut data if exceeds limit

returns:<br>
tr_r (int): index separating train and test data
totalX (numpy array, 2-dim):input data
totalY(numpy array, 1-dim): target data
totalName(numpy array 1-dim): file index

<h5>parse_train(data_path,ratio,limit)</h5>
input:<br>
data_path (str): path to data csv file
ratio (float): ratio of train data
limit (int): cut data if exceeds limit
returns:<br>
X (numpy array, 2-dim):input data
Y (numpy array, 1-dim): target data
name(numpy array 1-dim): file index

<h5>column_slice(data)</h5>
input:<br>
data (numpy array, 2-dim): csv data read into a numpy array without X, Y, index separation

returns:<br>
X (numpy array, 2-dim):input data
Y (numpy array, 1-dim): target data
name(numpy array 1-dim): file index

<h5>row_slice(X,y,percent)
input:<br>
X (numpy array, 2-dim):input data
Y (numpy array, 1-dim): target data
percent (float): train data ratio

returns:<br>
trainX (numpy array, 2-dim): train input data
trainY (numpy array, 1-dim): target data
testX (numpy array, 2-dim): testinput data
testY (numpy array, 1-dim): target data

<h4>Machine-learning</h4>
<h5>make_2D(x)
input:<br>
x (numpy array, 2-dim): input data

output:<br>
result_t, result_p (numpy array, 2-dim): array of size (number of rows in the original data )* 2 that is either reduced with t-SNE or PCA

<h5>try_simple_SVC(X,Y)
input:<br>
X (numpy array, 2-dim):input data
Y (numpy array, 1-dim): target data

output:<br>
svc (sklearn SVC model): model fitted with X and Y

<h5>try_simple_rf(x)
input:<br>
X (numpy array, 2-dim):input data

output:<br>
rf (sklearn RF model): model fitted with X 

<h5>runSVC(x,y)
input:<br>
X (numpy array, 2-dim):input data
Y (numpy array, 1-dim): target data

output:<br>
pred_tsne (numpy array, 1-dim): test prediction result with tsne input fed SVC
pred_pca (numpy array, 1-dim): test prediction result with pca input fed SVC
name(numpy array, 1-dim): file name array

<h5>runRF(x,y)
input: <br>
X (numpy array, 2-dim):input data
Y (numpy array, 1-dim): target data

output:<br>
rf (sklearn RF model): fitted model
pred (numpy array, 1-dim): test prediction result with rf
name(numpy array, 1-dim): file name array

<h5>final(trainpath) 
input:<br>
trainpath (str): train data file path. No splitting of data for testing

return:<br>
rf (sklearn RF model): fitted model


<h4>Presentation</h4>
<h5>print_result(pred, name)
input:<br>
pred (numpy array, 1-dim): prediction result array
name (numpy array, 1-dim): file name array

return:<br>
void

<h5>print_2D(title,X,y)
input:<br>
title (str): title of the graph
X (numpy array, 2-dim):input data
Y (numpy array, 1-dim): target data

return:<br>
void
Use of code

<h2>run_ml.py</h2>

역할: 웹에서 최종으로 선택된 pre-processed classifier algorithm 모델을 이용해 바로 prediction을 얻을 수 있는 ml 모델 코드<br>

<h2>ls.sh</h2>
역할: 웹에서 업로드되어 /dataset/input 폴더에 저장된 bulk 파일을 unzip 하여 해당 directory에 압축을 풀어주고, bulk파일을 삭제하는 코드<br>

<h2>pdf_to libsvm.sh</h2>
역할: /dataset/input 내부에 저장된 pdf들의 feature를 추출하고 libsvm_to_csv2.py를 실행하여 output_test.csv 파일로 파싱하고, 해당 csv 파일을 run_ml.py의 훈련 모델을 통해 예측값을 얻을 수 있는 코드
