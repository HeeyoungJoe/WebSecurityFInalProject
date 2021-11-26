#!/bin/bash

# 사용자가 입력한 data들을 어느 directory에 넣을지를 정해야함.
# 현재 코드는 사용자가 input한 pdf file들이 모두 benign 파일이라고 가정하고 실행
cd ~/dataset/input
ls -d $PWD/* > ~/hidost/build/pdfs.txt    # cached-ben에 들어갈 txt
cd ~/dataset/test
ls -d $PWD/* > ~/hidost/build/test.txt

cd ~/hidost/build/
mkdir cache-pdfs cache-test
./src/cacher -i pdfs.txt --compact --values -c cache-pdf/ -t10 -m256
./src/cacher -i test.txt --compact --values -c cache-test/ -t -m256
find $PWD/cache-pdf -name '*.pdf' -not -empty > cached-pdfs.txt
find $PWD/cache-test -name '*.pdf' -not -empty > cached-test.txt

./src/feat-extract -b cached-pdfs.txt -m cached-test.txt -f features.nppf --values -o data.libsvm

python libsvm_to_csv.py > output.csv
