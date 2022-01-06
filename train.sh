tar -xzvf data.tar.gz
pip install -r requirements.txt
rm ./data/*/.*.jpeg
python3 train.py
