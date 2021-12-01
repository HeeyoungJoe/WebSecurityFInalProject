import csv
f = open('./data.libsvm', 'r')

lines = f.readlines()
for line in lines:
	
	minus_list = [-1]*275
	
	l = line.split(' #')[0]
	l = l.split(' ')
	file_name = line.split(' #')[1].strip('\n')
	
	for i in range(len(l)):
		if i != 0:
			key = int(l[i].split(':')[0])
			val = float(l[i].split(':')[1])
			minus_list[key-1]=val
			
	# 교수님 코드에서는 if l[0]값에 따라 mb = 'M' or mb = 'B' 넣고 모든 값에 대해 print 했지만 우리는 사용자가 넣은 파일에 대해서만 print
	# mb를 처음부터 넣지 않을 수도 있음
	if int(l[0]) == 1:
		mb = 'M'
	
	 	result_string=mb+', '+str(minus_list).strip('[]')+', '+file_name
	 	print(result_string)
