# bash jobs_oneclick.sh run.ja_zh.v01.json 2 5
# bash jobs_oneclick.sh run.ja_zh.v01.json 1,2,3,6
source /nfs/cold_project/users/xingshi/miniconda3/etc/profile.d/conda.sh
conda activate next_sent
cd /nfs/project/users/xingshi/mt/
CONFIG=$1
bash oneclick.sh config/$CONFIG $2 $3
