# -*- coding: utf-8 -*-
"""Generate config/run.*.json files in a batch
"""
import json

def generate_run(direction, lang_pair, data_version, hparam_name, hparam, ngpu, model = "transformer", replicate_id = None):
  run_version = "{direction}.{hparam_name}.{data_version}".format(direction = direction, hparam_name = hparam_name, data_version = data_version)
  if replicate_id != None:
    run_version = run_version + ".r{}".format(replicate_id)
  json_name = "run.{}.json".format(run_version)
  d = {
    "version":run_version,
    "processed":"{}.{}".format(lang_pair, data_version),
    "hparams":hparam, 
    "model":model,
    "problem":"machine_translation",
    "n_gpu":ngpu,
    "eval_early_stopping_steps":14500,
    "eval_steps":10000,
    "local_eval_frequency":1000,
    "keep_checkpoint_max":10,
    "beam_size": [4],
    "alpha": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
  }

  return json_name, d

def generate_processed(lang_pair, data_version, vocab_size = 30000):
  processed_version = "{}.{}".format(lang_pair, data_version)
  d = {
    "version":processed_version,
    "train":"train.{}".format(processed_version),
    "dev":"dev.ja_zh.v01",
    "tests":["dev.ja_zh.v01"],
    "bpe": True,
    "vocab_size": vocab_size
  }
  json_name = "processed.{}.json".format(processed_version)
  return json_name, d

def v16():
  directions = ['ja_zh','zh_ja']
  hparams = {
    #"big.1gpu":("transformer_big_single_gpu",1),
    "big.single_gpu":("transformer_big_single_gpu",1)
  }
  
  data_versions = {"existing_parallel"}
  lang_pair = "ja_zh"
  
  for direction in directions:
    for hparam_name in hparams:
      hparam, ngpu = hparams[hparam_name]
      for data_version in data_versions:
        
        json_name, d = generate_processed(lang_pair, data_version)
        with open("config/{}".format(json_name), 'w') as f:
          print(json_name)
          json.dump(d, f, ensure_ascii=False, indent = 4)

        json_name, d = generate_run(direction, lang_pair, data_version, hparam_name, hparam, ngpu)
        with open("config/{}".format(json_name), 'w') as f:
          print(json_name)
          json.dump(d, f, ensure_ascii=False, indent = 4)

if __name__ == "__main__":
  v16()
