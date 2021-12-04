#!/bin/bash

# 사용자가 입력한 data들을 ~/dataset/input에 넣어줘야함.
# 현재 코드는 사용자가 input한 pdf file들이 모두 malicious 파일이라고 가정하고 실행
# ML알고리즘을 돌릴때 csv의 M부분은 drop하고 실행되어서 상관없음.

cd ~/dataset/input
ls -d $PWD/* > ~/hidost/build/pdfs.txt    # cached-ben에 들어갈 txt
# feat-extract에서 benign file들의 cached.txt가 비어있으면 안된다고 해서 빈pdf파일을 하나 test directory에 넣을 예정
cd ~/dataset/test     
ls -d $PWD/* > ~/hidost/build/test.txt

cd ~/hidost/build/
mkdir cache-pdfs cache-test
./src/cacher -i pdfs.txt --compact --values -c cache-pdfs/ -t10 -m256
./src/cacher -i test.txt --compact --values -c cache-test/ -t10 -m256
find $PWD/cache-pdfs -name '*.pdf' -not -empty > cached-pdfs.txt
find $PWD/cache-test -name '*.pdf' -not -empty > cached-test.txt

./src/feat-extract -m cached-pdfs.txt -b cached-test.txt -f features.nppf --values -o data.libsvm

# libsvm_to_csv2.py에서 -m cached-pdfs.txt에 있는 파일들만 print > output_test
python libsvm_to_csv2.py > output_test.csv
