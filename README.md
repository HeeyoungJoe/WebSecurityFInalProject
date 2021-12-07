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

<h1>build 폴더</h1>
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
      - traindata_path (str): path to traindata csv file<br>
      - testdata_path (str): path to testdata csv file<br>
      - limit (int): cut data if exceeds limit<br>

returns:<br>
      - tr_r (int): index separating train and test data<br>
      - totalX (numpy array, 2-dim):input data<br>
      - totalY(numpy array, 1-dim): target data<br>
      - totalName(numpy array 1-dim): file index<br>

<h5>parse_train(data_path,ratio,limit)</h5>
input:<br>
      - data_path (str): path to data csv file<br>
      - ratio (float): ratio of train data<br>
      - limit (int): cut data if exceeds limit<br>

returns:<br>
      - X (numpy array, 2-dim):input data<br>
      - Y (numpy array, 1-dim): target data<br>
      - name(numpy array 1-dim): file index<br>

<h5>column_slice(data)</h5>
input:<br>
      - data (numpy array, 2-dim): csv data read into a numpy array without X, Y, index separation<br>

returns:<br>
      - X (numpy array, 2-dim):input data<br>
      - Y (numpy array, 1-dim): target data<br>
      - name(numpy array 1-dim): file index<br>

<h5>row_slice(X,y,percent)</h5>
input:<br>
      - X (numpy array, 2-dim):input data<br>
      - Y (numpy array, 1-dim): target data<br>
      - percent (float): train data ratio<br>

returns:<br>
      - trainX (numpy array, 2-dim): train input data<br>
      - trainY (numpy array, 1-dim): target data<br>
      - testX (numpy array, 2-dim): testinput data<br>
      - testY (numpy array, 1-dim): target data<br>

<h4>Machine-learning</h4>
<h5>make_2D(x)</h5>
input:<br>
      - x (numpy array, 2-dim): input data<br>

output:<br>
      - result_t, result_p (numpy array, 2-dim): array of size (number of rows in the original data )* 2 that is either reduced with t-SNE or PCA<br>

<h5>try_simple_SVC(X,Y)</h5>
input:<br>
      - X (numpy array, 2-dim):input data<br>
      - Y (numpy array, 1-dim): target data<br>

output:<br>
      - svc (sklearn SVC model): model fitted with X and Y<br>

<h5>try_simple_rf(x)</h5>
input:<br>
      - X (numpy array, 2-dim):input data<br>

output:<br>
      - rf (sklearn RF model): model fitted with X <br>

<h5>runSVC(x,y)</h5>
input:<br>
      - X (numpy array, 2-dim):input data<br>
      - Y (numpy array, 1-dim): target data<br>

output:<br>
      - pred_tsne (numpy array, 1-dim): test prediction result with tsne input fed SVC<br>
      - pred_pca (numpy array, 1-dim): test prediction result with pca input fed SVC<br>
      - name(numpy array, 1-dim): file name array<br>

<h5>runRF(x,y)</h5>
input: <br>
      - X (numpy array, 2-dim):input data<br>
      - Y (numpy array, 1-dim): target data<br>

output:<br>
      - rf (sklearn RF model): fitted model<br>
      - pred (numpy array, 1-dim): test prediction result with rf<br>
      - name(numpy array, 1-dim): file name array<br>

<h5>final(trainpath) </h5>
input:<br>
      - trainpath (str): train data file path. No splitting of data for testing<br>

return:<br>
      - rf (sklearn RF model): fitted model<br>

<h4>Presentation</h4>
<h5>print_result(pred, name)</h5>
input:<br>
      - pred (numpy array, 1-dim): prediction result array<br>
      - name (numpy array, 1-dim): file name array<br>

return:<br>
      - void<br>

<h5>print_2D(title,X,y)</h5>
input:<br>
      - title (str): title of the graph<br>
      - X (numpy array, 2-dim):input data<br>
      - Y (numpy array, 1-dim): target data<br>

return:<br>
      - void<br>



<h2>run_ml.py</h2>
역할: 웹에서 최종으로 선택된 pre-processed classifier algorithm 모델을 이용해 바로 prediction을 얻을 수 있는 ml 모델 코드<br>
<h2>ls.sh</h2>
역할: 웹에서 업로드되어 /dataset/input 폴더에 저장된 bulk 파일을 unzip 하여 해당 directory에 압축을 풀어주고, bulk파일을 삭제하는 코드<br>
<h2>pdf_to libsvm.sh</h2>
역할: /dataset/input 내부에 저장된 pdf들의 feature를 추출하고 libsvm_to_csv2.py를 실행하여 output_test.csv 파일로 파싱하고, 해당 csv 파일을 run_ml.py의 훈련 모델을 통해 예측값을 얻을 수 있는 코드
