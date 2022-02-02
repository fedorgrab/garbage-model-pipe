#!/bin/bash
tar -xf data.tar.gz
echo "Data unzipped"
pip -q install -r requirements.txt
echo "Python dependencies installed"
rm ./data/*/.*.jpeg
python train.py

