#!/bin/bash

# 사용자가 입력한 data들을 어느 directory에 넣을지를 정해야함.
# 현재 코드는 사용자가 input한 pdf file들이 모두 benign 파일이라고 가정하고 실행
cd ~/dataset/
ls -d $PWD/* > ~/hidost/build/pdfs.txt    # cached-ben에 들어갈 txt
touch empty-pdfs.txt                      # cached-mal에 들어갈 txt

cd ~/hidost/build/
mkdir cache-pdfs cache-empty
./src/cacher -i pdfs.txt --compact --values -c cache-pdf/ -t10 -m256
./src/cacher -i empty-pdfs.txt --compact --values -c cache-empty/ -t -m256
find $PWD/cache-pdf -name '*.pdf' -not -empty > cached-pdfs.txt
find $PWD/cache-empty -name '*.pdf' -not -empty > cached-empty.txt

./src/feat-extract -b cached-pdfs.txt -m cached-empty.txt -f features.nppf --values -o data.libsvm

python libsvm_to_csv.py > output.csv
