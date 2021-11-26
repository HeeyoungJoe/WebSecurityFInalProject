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

print
--> alarm parsing done

[class-MyParser]
objectives- several file

<init object>
*class action
- parse all files in path
- make batch with batch() class method
*class variables
- batch_size: default None
- batch_count: default 1
- remainder_count: default None
- batch_data: if file number less than threshold k, combine all data 
into one big numpy instance (2D데이터들이라면 3D로 모음)

<methods>
- parse(self): parse all data in path
- parse_single(self,filename): parse single data (used in parse)
- pad(self, n): pad columns of all files
- rebatch(self,n): return a numpy object with all numpy matrices combined
(1) concatenate
(2) split in size n (default 1)
(3) comment batch size, batch count, remainder count

print
--> parse begin
--> parse end
--> batch result