"""Make softlinks to the processed data

python dp_run.py run.ja_zh.toy.v01.json

"""
import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, root_dir)

import json
import subprocess as sp
from src.util import run_cmd

  
if __name__ == "__main__":
  # load config
  fn_config_json = sys.argv[1]
  with open(fn_config_json) as f:
    args = json.load(f)

  env = {}
  env['root_dir'] = root_dir
  # generate_processed_pipeline
  processed_version = args['processed']
  env["processed_version"] = processed_version
  processed_json = os.path.join(root_dir, "config/processed.{processed_version}.json".format(**env))
  env["processed_json"] = processed_json
  run_cmd('python {root_dir}/scripts/dp_processed.py {processed_json}'.format(**env))

  # make folder
  version = args['version']
  env['version'] = version
  run_tmp_dir = os.path.join(root_dir, "data/run/{version}_tmp".format(**env))
  env['run_tmp_dir'] = run_tmp_dir
  run_cmd("mkdir -p {run_tmp_dir}".format(**env))
  run_dir = os.path.join(root_dir, "data/run/{version}".format(**env))
  env['run_dir'] = run_dir
  run_cmd("mkdir -p {run_dir}".format(**env))

  processed_version = args['processed']
  env['processed_version'] = processed_version
  processed_dir = os.path.join(root_dir, "data/processed/{processed_version}".format(**env))
  env['processed_dir'] = processed_dir
  
  # extract languages
  langs = version.split('.')[0].split('_')
  lang_src = langs[0]
  lang_tgt = langs[1]
  env['lang_src'] = lang_src
  env['lang_tgt'] = lang_tgt
  
  # make softlinks

  ## vocab
  run_cmd("ln -s {processed_dir}/vocab.{lang_src} {run_tmp_dir}/inputs.vocab".format(**env))
  run_cmd("ln -s {processed_dir}/vocab.{lang_tgt} {run_tmp_dir}/targets.vocab".format(**env))

  # train
  run_cmd("ln -s {processed_dir}/train.{lang_src}.bpe {run_tmp_dir}/inputs.train.txt".format(**env))
  run_cmd("ln -s {processed_dir}/train.{lang_tgt}.bpe {run_tmp_dir}/targets.train.txt".format(**env))

  # dev
  run_cmd("ln -s {processed_dir}/dev.{lang_src}.bpe {run_tmp_dir}/inputs.eval.txt".format(**env))
  run_cmd("ln -s {processed_dir}/dev.{lang_tgt}.bpe {run_tmp_dir}/targets.eval.txt".format(**env))

  # tok and char
  fn_tok = "{processed_dir}/dev.{lang_tgt}.tok".format(**env)
  fn_run_tok = "{run_tmp_dir}/targets.eval.tok".format(**env)

  if os.path.exists(fn_tok) and not os.path.exists(fn_run_tok):
    run_cmd("ln -s {} {}".format(fn_tok, fn_run_tok))

  fn_char = "{processed_dir}/dev.{lang_tgt}.char".format(**env)
  fn_run_char = "{run_tmp_dir}/targets.eval.char".format(**env)

  if os.path.exists(fn_char) and not os.path.exists(fn_run_char):
    run_cmd("ln -s {} {}".format(fn_char, fn_run_char))

  
  # test
  test_id = 0
  while True:
    env['test_id'] = test_id
    test_path = "{processed_dir}/test{test_id}.{lang_src}.bpe".format(**env)
    if os.path.exists(test_path):
      run_cmd("ln -s {processed_dir}/test{test_id}.{lang_src}.bpe {run_tmp_dir}/inputs.test{test_id}.txt".format(**env))
      run_cmd("ln -s {processed_dir}/test{test_id}.{lang_tgt}.bpe {run_tmp_dir}/targets.test{test_id}.txt".format(**env))

      # tok, char and dict
      fn_tok = "{processed_dir}/test{test_id}.{lang_tgt}.tok".format(**env)
      fn_run_tok = "{run_tmp_dir}/targets.test{test_id}.tok".format(**env)
      if os.path.exists(fn_tok) and not os.path.exists(fn_run_tok):
        run_cmd("ln -s {} {}".format(fn_tok, fn_run_tok))
        
      fn_char = "{processed_dir}/test{test_id}.{lang_tgt}.char".format(**env)
      fn_run_char = "{run_tmp_dir}/targets.test{test_id}.char".format(**env)
      if os.path.exists(fn_char) and not os.path.exists(fn_run_char):
        run_cmd("ln -s {} {}".format(fn_char, fn_run_char))

      fn_dict = "{processed_dir}/test{test_id}.{lang_tgt}.dict".format(**env)
      fn_run_dict = "{run_tmp_dir}/targets.test{test_id}.dict".format(**env)
      if os.path.exists(fn_dict) and not os.path.exists(fn_run_dict):
        run_cmd("ln -s {} {}".format(fn_dict, fn_run_dict))
        
      test_id += 1
    else:
      break


  
  
