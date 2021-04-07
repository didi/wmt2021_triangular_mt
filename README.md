# Baseline code for WMT 2021 Triangular MT

Updated on 04/07/2021. 

The baseline code for the shared task [`Triangular MT: Using English to improve Russian-to-Chinese machine translation`](http://www.statmt.org/wmt21/triangular-mt-task.html). 

## NOTE

All scripts should run from root folder:

```
bash **.sh
bash scripts/**.sh
python scripts/**.py
```

## Requirements

A linux machine GPU and installed CUDA >= 10.0

## Setup

1. Install `miniconda` on your machine.
2. Run `setup_env.sh` with interactive mode:
    ```bash
       bash -i setup_env.sh
    ```
    Note: If you are using a server outside of China, you'd better delete two tsinghua mirrors in `environment.yml` line `3-4` and `setup_env.sh` line `9` for a better speed.
    
## Registration

To participate please register to the shared task on Codalab .
[`Link to Codalab website`](https://competitions.codalab.org/competitions/30446). 

### Detailed Configuration

We will use the toolkit [`tensor2tensor`](https://github.com/tensorflow/tensor2tensor) to train a Transformer based NMT system. 
`config/run.ru_zh.big.single_gpu.json` lists all the configurations. 

```json
{
    "version": "ru_zh.big.single_gpu",
    "processed": "ru_zh",
    "hparams": "transformer_big_single_gpu",
    "model": "transformer",
    "problem": "machine_translation",
    "n_gpu": 1,
    "eval_early_stopping_steps": 14500,
    "eval_steps": 10000,
    "local_eval_frequency": 1000,
    "keep_checkpoint_max": 10,
    "beam_size": [
        4
    ],
    "alpha": [
        1.0
    ]
}
```

The hyperparameter set is `transformer_big_single_gpu`. 
We will use only `1` GPU. 
The model will evaluate the dev loss and save the checkpoint every `1000` steps. 
If the dev loss doesn't decrease for `14500` steps, the training will stop. 
When decoding the test set, we will use beam size `4` and use alpha value of 1.0. 
The larger the alpha value, the longer the generated translation will be.

`processed` indicates the version of the processed files. Here is `config/processed.ru_zh.json`:

```json
{
    "version": "ru_zh",
    "train": "train.ru_zh",
    "dev": "dev.ru_zh",
    "tests": [
        "dev.ru_zh"
    ],
    "bpe": true,
    "vocab_size": 30000
}
``` 
It indicates that the training folder is `data/raw/train.ru_zh`, dev folder is `data/raw/dev.ru_zh` and test folder is `data/raw/dev.ru_zh`, i.e. we use the dev as test. 
The preprocessing pipeline will use byte-pair-encoding (BPE) and the number of merge operations are `30000`. 

## Train and Decode


To train a Russian to Chinese NMT system: 

```
conda activate mt_baseline
bash pipeline.sh config/run.ru_zh.big.single_gpu.json 1 4
```

`1` is the start step and `4` is the end step.

- step 1: prepare data
- step 2: generate tf records
- step 3: train
- setp 4: decode_test : decode test with all combinations of (beam, alpha)

After step 4, all the decoded results will be in folder `data/run/ru_zh.big.single_gpu_tmp/decode`:
* `decode.b4_a1.0.test0.txt`: the decoded BPE subwords using beam size 4 and alpha value 1.0.
* `decode.b4_a1.0.test0.tok`: the decoded tokens when we merge the BPE subwords into whole words.
* `decode.b4_a1.0.test0.char`： the decoded utf8 characters of `decode.b4_a1.0.test0.tok` after removing space.
* `bleu.b4_a1.0.test0.tok`: the token level BLEU score.
* `bleu.b4_a1.0.test0.char`: the character level BLEU score. 

The reference files are in folder `data/run/ru_zh.big.single_gpu_tmp/decode`.

#### Note 

Please use your own dev set if you need to tune. We have released the russian source of the dev set on Codalab. You can submit your system outputs on Codalab to get the Bleu score on the released dev set. 

## Independent Evaluation Script

Folder `eval` contains the evaluation scripts to calculate the character-level BLEU score:

```
cd eval
python bleu.py hyp.txt ref.txt
```
Where `hyp.txt` and `ref.txt` can be either normal Chinese (i.e. without space between characters) or character-split Chinese.

See 'example.sh' for detailed examples. 
