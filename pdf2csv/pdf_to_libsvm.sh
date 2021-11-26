#!/bin/bash

cd ~/dataset/
ls -d $PWD/* > ~/hidost/build/pdfs.txt

cd ~/hidost/build/
mkdir cache-pdf
./src/cacher -i pdfs.txt --compact --values -c cache-pdf/ -t10 -m256
find $PWD/cache-pdf -name '*.pdf' -not -empty > cached-pdfs.txt

./src/feat-extract -b cached-pdfs.txt -f features.nppf --values -o data.libsvm

python libsvm_to_csv.py > output.csv
