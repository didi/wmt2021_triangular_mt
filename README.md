# Baseline model for IWSLT Open Domain Translation

Update in 1/20/2020


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
    
## Data Download

Run `download_data.sh` to download dev and exsiting_parallel data:
```bash
bash download_data.sh
```

This script will download and decompress the data to `data/orig` and then copy the parallel files as `data/raw/dev.ja_zh.v01/(ja|zh)` and `data/raw/train.ja_zh.existing_parallel/(ja|zh)`

To download other data, please register at http://iwslt.org/doku.php?id=open_domain_translation to get the download link.



### Detailed Configuration

We will use the toolkit [`tensor2tensor`](https://github.com/tensorflow/tensor2tensor) to train a Transformer based NMT system. 
`config/run.ja_zh.big.single_gpu.existing_parallel.json` lists all the configurations. 

```json
{
    "version": "ja_zh.big.single_gpu.existing_parallel",
    "processed": "ja_zh.existing_parallel",
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
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
        1.2,
        1.4,
        1.6
    ]
}
```

The hyperparameter set is `transformer_big_single_gpu`. 
We will use only `1` GPU. 
The model will evaluate the dev loss and save the checkpoint every `1000` steps. 
If the dev loss doesn't decrease for `14500` steps, the training will stop. 
When decoding the test set, we will use beam size `4` and try different alpha values from 0.0 to 1.6. 
The larger the alpha value, the longer the generated translation will be.

`processed` indicates the version of the processed files. Here is `config/processed.ja_zh.existing_parallel.json`:

```json
{
    "version": "ja_zh.existing_parallel",
    "train": "train.ja_zh.existing_parallel",
    "dev": "dev.ja_zh.v01",
    "tests": [
        "dev.ja_zh.v01"
    ],
    "bpe": true,
    "vocab_size": 30000
}
``` 
It indicates that the training folder is `data/raw/train.ja_zh.existing_parallel`, dev folder is `data/raw/dev.ja_zh.v01` and test folder is `data/raw/dev.ja_zh.v01`, i.e. we use the dev as test. 
The preprocessing pipeline will use byte-pair-encoding (BPE) and the number of merge operations are `30000`. 

## Train and Decode


To train a Japanese to Chinese NMT system: 

```
conda activate mt_baseline
bash pipeline.sh config/run.ja_zh.big.single_gpu.existing_parallel.json 1 4
```

`1` is the start step and `4` is the end step.

- step 1: prepare data
- step 2: generate tf records
- step 3: train
- setp 4: decode_test : decode test with all combinations of (beam, alpha)

After step 4, all the decoded results will be in folder `data/run/ja_zh.big.single_gpu.existing_parallel_tmp/decode`:
* `decode.b4_a0.0.test0.txt`: the decoded BPE subwords using beam size 4 and alpha value 0.0.
* `decode.b4_a0.0.test0.tok`: the decoded tokens when we merge the BPE subwords into whole words.
* `decode.b4_a0.0.test0.char`： the decoded utf8 characters of `decode.b4_a0.0.test0.tok` after removing space.
* `bleu.b4_a0.0.test0.tok`: the token level BLEU score.
* `bleu.b4_a0.0.test0.char`: the character level BLEU score. 

The reference files are in folder `data/run/ja_zh.big.single_gpu.existing_parallel_tmp/decode`.

