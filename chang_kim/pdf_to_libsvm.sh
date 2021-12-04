#!/bin/bash

cd ~/dataset/input
ls -d $PWD/* > ~/hidost/build/pdfs.txt
cd ~/dataset/test
ls -d $PWD/* > ~/hidost/build/test.txt

cd ~/hidost/build/
mkdir cache-pdfs cache-test

./src/cacher -i pdfs.txt --compact --values -c cache-pdfs/ -t10 -m256
./src/cacher -i test.txt --compact --values -c cache-test/ -t10 -m256
find $PWD/cache-pdfs -name '*.pdf' -not -empty > cached-pdfs.txt
find $PWD/cache-test -name '*.pdf' -not -empty > cached-test.txt

./src/feat-extract -b cached-test.txt -m cached-pdfs.txt -f features.nppf --values -o data.libsvm

python libsvm_to_csv2.py > output_test.csv
