# create a conda environment
conda env create -f environment.yml

conda activate mt_baseline

pip install pip -U

# remove the following line if you are outside of China.
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# install tensorflow
pip install tensorflow-probability==0.7.0
pip install tensorflow-gpu==1.14.0
pip install tensorflow_hub==0.4.0
pip install tensor2tensor==1.13
pip install tensorflow-datasets==1.3.2

# make sure the version is 1.12.0 and True for is_gpu_available
python -c 'import tensorflow as tf; print(tf.__version__); print(tf.test.is_gpu_available());'

# install japanese and chinese segmentation tool
#sudo apt-get install mecab libmecab-dev mecab-ipadic
#sudo apt-get install mecab-ipadic-utf8
#sudo apt-get install python-mecab

# install chinese and japanese processing tools
pip install certifi==2019.3.9
pip install glob3==0.0.1
pip install hanziconv==0.3.2
pip install jieba==0.39
pip install opencc-python-reimplemented==0.1.4
pip install mecab-python3

# install subword-nmt
pip install subword-nmt

# install filelock
pip install filelock

pip install pandas
pip install nltk==3.5
