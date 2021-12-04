import csv
f = open('./data.libsvm', 'r')

lines = f.readlines()
index = 0
for line in lines:

	minus_list = [-1]*275

	l = line.split(' #')[0]
	l = l.split(' ')
	file_name = line.split(' #')[1].strip('\n')

	for i in range(len(l)):
		if i != 0:
			key = int(l[i].split(':')[0])
			val = float(l[i].split(':')[1])
			minus_list[key-1] = val
	if int(l[0]) == 1:
		mb = 'M'

		result_string = mb+', '+str(minus_list).strip('[]')+', '+str(index)
		print(result_string)
		index += 1
