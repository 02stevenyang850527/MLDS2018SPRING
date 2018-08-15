#!/bin/bash
wget -O save.zip "https://www.dropbox.com/s/78vr6zwws0cvymx/save.zip?dl=1"
wget -O save2.zip "https://www.dropbox.com/s/dewuplri3bzw5rl/save2.zip?dl=1"
unzip save.zip
unzip save2.zip
python3 hw2_2_model1.py -f $1 -m test -r 151
python3 hw2_2_model2.py -f $1 -m test -r 40
python3 ensemble.py $2
