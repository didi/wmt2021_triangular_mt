# download dev data and the existing_parallel training data, then make softlink

# download data
mkdir -p data/orig

curl https://iwslt.oss-cn-beijing.aliyuncs.com/dev_dataset.tgz -o data/orig/dev_dataset.tgz
curl https://iwslt.oss-cn-beijing.aliyuncs.com/existing_parallel.tgz -o data/orig/existing_parallel.tgz

tar zxvf data/orig/dev_dataset.tgz -C data/orig/
tar zxvf data/orig/existing_parallel.tgz -C data/orig/

mkdir -p data/raw/dev.ja_zh.v01 data/raw/train.ja_zh.existing_parallel
cp data/orig/dev_dataset/segments.ja data/raw/dev.ja_zh.v01/ja
cp data/orig/dev_dataset/segments.zh data/raw/dev.ja_zh.v01/zh

cp data/orig/existing_parallel/segments.ja data/raw/train.ja_zh.existing_parallel/ja
cp data/orig/existing_parallel/segments.zh data/raw/train.ja_zh.existing_parallel/zh
