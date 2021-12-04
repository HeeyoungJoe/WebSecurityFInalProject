import csv
f = open('./data.libsvm', 'r')
fw = open('./filename.csv', 'a', newline='')
wr = csv.writer(fw)

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
         minus_list[key-1] = val
   if int(l[0]) == 1:
      mb = 'M'
      
      result_string = mb+', '+str(minus_list).strip('[]')+', '+index
    wr.writerow([index, file_name])
      # result_string = mb+', '+str(minus_list).strip('[]')+', '+file_name
      print(result_string)   
    
fw.close()
