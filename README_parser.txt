[class-MyDocument]
objectives- single file

<init object>
*actions
- read csv files
- make them become numpy arrays
*class variables
- size: track their size
- filename: track filename
- X: 첫번째 열 제외
- Y: 첫번째 열

[class-MyParser]
objectives- several file

<init object>
*class action
- parse all files in path
*class variables
- batch_size: default 0/batch 하나당 행 개수
- batch_count: default 1/ batch 개수
- max_ft:default 0/batch 하나당 열 개수
- batch_data: [batch_count,batch_size,max_ft] 사이즈의 numpy ndarray
- batch_target: [batch_count,batch_size,1] 사이즈의 numpy ndarray

<methods>
- parse(self): parse all data in path
- parse_single(self,filename): parse single data (used in parse)
- pad_row(self, n): pad columns of all files
- rebatch(self,n): return a numpy object with all numpy matrices combined
(1) 만약 새 batch 사이즈로 현재 있는 row 개수가 나누어 떨어지지 않으면 pad_row를 실행한다.
(2) resize data&target
(3) update batch size, batch count, remainder count

print
--> parse result
--> batch result