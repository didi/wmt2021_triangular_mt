# one click to start the whole MT pipeline.

# run step 1 to step 4
# bash pipeline.sh config/run.jp_zh.toy.v01 1-4
# OR
# run step 1 to step 4
# bash pipeline.sh config/run.jp_zh.toy.v01 1 4

# run step 1 and step 4
# bash pipeline.sh config/run.jp_zh.toy.v01 1,4

# step 1: prepare data
# step 2: generate tf records
# step 3: train
# setp 4: decode_test_all


CONFIG=$1

if [ $# -eq 2 ]
then
    STEPS=$(python scripts/parse_args.py $2)
elif [ $# -eq 3 ]
then
    STEPS=$(seq $2 $3)
fi

for STEP in $STEPS;
do

if [ $STEP -eq 1 ]
then

    # prepare data
    python scripts/mt.py $CONFIG init
    python scripts/mt.py $CONFIG prepare_data
fi

## MT train part
if [ $STEP -eq 2 ]
then

    # generate tf records
    python scripts/mt.py $CONFIG datagen

fi

if [ $STEP -eq 3 ]
then
    
    # train
    python scripts/mt.py $CONFIG train

fi


if [ $STEP -eq 4 ]
then

    # decode_test_all
    python scripts/mt.py $CONFIG decode_test_all

fi

done

echo "done"
