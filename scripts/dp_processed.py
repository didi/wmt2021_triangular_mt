"""Generate the processed data

python scripts/dp_processed.py config/processed.ja_zh.toy.v01.json

"""

import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, root_dir)

import json
from src.util import run_cmd
from src.tok import apply_bpe
from filelock import Timeout, FileLock
from pathlib import Path


def remove_freq(fnin ,fnout):
  with open(fnin) as fin, open(fnout,'w') as fout:
    for line in fin:
      ll = line.strip().split()
      fout.write(ll[0] + "\n")

      
if __name__ == "__main__":
  fn_config_json = sys.argv[1]
  with open(fn_config_json) as f:
    args = json.load(f)

  # make folder
  version = args['version']
  processed_dir = os.path.join(root_dir, "data/processed/{}".format(version))
  run_cmd("mkdir -p {}".format(processed_dir))


  fn_lock = os.path.join(processed_dir, 'lock')
  lock = FileLock(fn_lock)
  with lock:
    
    # extract languages
    langs = version.split('.')[0].split('_')
    lang1 = langs[0]
    lang2 = langs[1]

    # tokenize the raw data if didn't
    
    dirs = []
    dirs.append(args['train'])
    dirs.append(args['dev'])
    dirs += args['tests']
    dirs = [os.path.join(root_dir, "data/raw/{}".format(d)) for d in dirs] 
    prefixes = ['train','dev'] + ['test{}'.format(i) for i in range(len(args['tests']))]


    for d in dirs:
      raw_dir = d
      tokenizer_path = os.path.join(root_dir, "scripts/dp_tokenize.py")
      run_cmd('python {} {}'.format(tokenizer_path, raw_dir))

    # run bpe if necessary
    run_bpe = args['bpe']
    vocab_size = args['vocab_size']
    if run_bpe:

      # Do BPE, prepare 6+ files:

      # - vocab.zh
      # - vocab.ja
      #
      # - train.zh.bpe
      # - train.ja.bpe
      #
      # - dev.zh.bpe
      # - dev.ja.bpe
      #
      # - test0.zh.bpe
      # - test0.ja.bpe
      # - test1.zh.bpe
      # - test1.ja.bpe
      # test1 and test2 are corresponding to the args['tests']

      raw_train_dir = dirs[0]

      print("[BPE] BPE with #merges: {}".format(vocab_size))

      fn_bpe_codes = os.path.join(processed_dir, 'bpe.codes')
      input_suffix = ".tok"
      if not os.path.exists(fn_bpe_codes):

        cmd = """subword-nmt learn-joint-bpe-and-vocab --input {raw_train_dir}/{lang1}{input_suffix} {raw_train_dir}/{lang2}{input_suffix} \
                            -s {vocab_size} \
                            -o {processed_dir}/bpe.codes \
                            --write-vocabulary {processed_dir}/vocab.{lang1}.freq {processed_dir}/vocab.{lang2}.freq
        """.format(raw_train_dir = raw_train_dir,
                   lang1 = lang1, lang2 = lang2,
                   vocab_size = vocab_size,
                   processed_dir = processed_dir,
                   input_suffix = input_suffix
        )

        run_cmd(cmd)

        for _lang in [lang1, lang2]:
          vocab_freq = "{processed_dir}/vocab.{lang}.freq".format(processed_dir = processed_dir, lang = _lang)
          vocab =  "{processed_dir}/vocab.{lang}".format(processed_dir = processed_dir, lang = _lang)
          remove_freq(vocab_freq, vocab)

      for raw_dir, prefix in zip(dirs, prefixes):
        for lang in langs:
          fn_bpe = os.path.join(processed_dir, "{}.{}.bpe".format(prefix, lang))
          fn_bpe_codes = "{}/bpe.codes".format(processed_dir)
          fn_tok = "{}/{}{}".format(raw_dir, lang, input_suffix)

          if not os.path.exists(fn_bpe):
            apply_bpe(fn_bpe_codes, fn_tok, fn_bpe)

    else:
      # TODO
      pass

    # make soft link to char and tok file
    for raw_dir, prefix in zip(dirs, prefixes):
      for lang in langs:
        fn_raw_char = "{raw_dir}/{lang}.char".format(raw_dir=raw_dir, lang=lang)
        fn_char = os.path.join(processed_dir, "{}.{}.char".format(prefix, lang))
        if os.path.exists(fn_raw_char) and not os.path.exists(fn_char):
          cmd = "ln -snf {} {}".format(fn_raw_char, fn_char)
          run_cmd(cmd)

        fn_raw_tok = "{raw_dir}/{lang}.tok".format(raw_dir=raw_dir, lang=lang)
        fn_tok = os.path.join(processed_dir, "{}.{}.tok".format(prefix, lang))
        if os.path.exists(fn_raw_tok) and not os.path.exists(fn_tok):
          cmd = "ln -snf {} {}".format(fn_raw_tok, fn_tok)
          run_cmd(cmd)

        fn_raw_dict = "{raw_dir}/{lang}.dict".format(raw_dir=raw_dir, lang=lang)
        fn_dict = os.path.join(processed_dir, "{}.{}.dict".format(prefix, lang))
        if os.path.exists(fn_raw_dict) and not os.path.exists(fn_dict):
          cmd = "ln -snf {} {}".format(fn_raw_dict, fn_dict)
          run_cmd(cmd)
  
    
  
  
