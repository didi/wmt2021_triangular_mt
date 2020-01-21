# -*- coding: utf-8 -*-
"""The python version to run tensor2tensor

python mt.py config/run.ja_zh.toy.v01.json action

action could be: 

- init : make all necessary folders
- prepare_data : clean, simplification of Chinese, Tokenization, BPE, and Make softlinks
- datagen: make data into TensorRecord format
- train: Training the model
- decode_test_all: do multiple decoding according to the beam_size and alpha combination 

"""

import os
import sys
import argparse

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, root_dir)

import json
import subprocess as sp
from src.util import run_cmd, calc_bleu, load_bleu_ratio
from src.tok import unbpe, tok2char
from pathlib import Path
from datetime import datetime


def get_tests_num(env):
  with open(env['processed_json']) as f:
    j = json.load(f)
    env['n_test']
    return len(j['tests'])
  
def get_env(args, env):
  # get env variables
  
  processed_version = args['processed']
  processed_json = os.path.join(root_dir, "config/processed.{}.json".format(processed_version))
  
  version = args['version']

  langs = version.split('.')[0].split('_')
  env['lang_src'] = langs[0]
  env['lang_tgt'] = langs[1]
  
  run_tmp_dir = os.path.join(root_dir, "data/run/{}_tmp".format(version))
  run_dir = os.path.join(root_dir, "data/run/{}".format(version))
  scripts_dir = os.path.join(root_dir, 'scripts')
  user_dir = os.path.join(root_dir, 'src/model')
  train_dir = os.path.join(root_dir, "train/{}".format(args['version']))
  result_dir = os.path.join(run_tmp_dir, 'decode')
  
  env['processed_json'] = processed_json
  env['user_dir'] = user_dir
  env['root_dir'] = root_dir
  env['tmp_dir'] = run_tmp_dir
  env['data_dir'] = run_dir
  env['scripts_dir'] = scripts_dir
  env['train_dir'] = train_dir
  env['result_dir'] = result_dir

  with open(env['processed_json']) as f:
    j = json.load(f)
    env['n_test'] = len(j['tests'])

  # for train_steps
  if not "train_steps" in args:
    args['train_steps'] = 250000
    
  return env

def init(args, env):
  cmd = "mkdir -p {tmp_dir} {data_dir} {train_dir} {result_dir}".format(**env)
  run_cmd(cmd)

def prepare_data(args, env):
  run_cmd('python {root_dir}/scripts/dp_run.py {run_json}'.format(**env))

def datagen(args, env):
  cmd = """
  t2t-datagen \
  --t2t_usr_dir={user_dir} \
  --data_dir={data_dir} \
  --tmp_dir={tmp_dir} \
  --problem={problem}
  """.format(**env, **args)
  run_cmd(cmd)

def train(args, env):
  cmd = """
  t2t-trainer \
  --t2t_usr_dir={user_dir} \
  --data_dir={data_dir} \
  --problem={problem} \
  --model={model} \
  --hparams_set={hparams} \
  --worker_gpu={n_gpu} \
  --schedule=train_and_evaluate \
  --export_saved_model=True \
  --eval_steps={eval_steps} \
  --keep_checkpoint_max={keep_checkpoint_max} \
  --local_eval_frequency={local_eval_frequency} \
  --eval_early_stopping_steps={eval_early_stopping_steps} \
  --train_steps={train_steps} \
  --output_dir={train_dir}
  """.format(**env, **args)
  run_cmd(cmd)

def decode_and_bleu(args, env, beam_size, alpha, part="eval", checkpoint = None):
  """
  Args:
    part: string: "eval" or "test0" or other things
  """

  checkpoint_path = ""
  if checkpoint:
    result_dir_bk = env['result_dir']
    env['result_dir'] = result_dir_bk + "/" + checkpoint
    cmd = "mkdir -p {}".format(env['result_dir'])
    run_cmd(cmd)
    checkpoint_path = "{}/bk/{}".format(env["train_dir"],checkpoint)
    
  decode_file = "{tmp_dir}/inputs.{part}.txt".format(tmp_dir = env['tmp_dir'], part = part )
  decode_reference_tok = "{tmp_dir}/targets.{part}.tok".format(tmp_dir = env['tmp_dir'], part = part )
  decode_reference_char = "{tmp_dir}/targets.{part}.char".format(tmp_dir = env['tmp_dir'], part = part )
  
  dtag = "b{beam_size}_a{alpha}".format(beam_size = beam_size, alpha = alpha)
  decode_output = "{result_dir}/decode.{dtag}.{part}.txt".format(result_dir = env['result_dir'], dtag = dtag, part = part)
  decode_output_tok = "{result_dir}/decode.{dtag}.{part}.tok".format(result_dir = env['result_dir'], dtag = dtag, part = part)
  decode_output_char = "{result_dir}/decode.{dtag}.{part}.char".format(result_dir = env['result_dir'], dtag = dtag, part = part)

  bleu_output = "{result_dir}/bleu.{dtag}.{part}.txt".format(result_dir = env['result_dir'], dtag = dtag, part = part)
  bleu_output_tok = "{result_dir}/bleu.{dtag}.{part}.tok".format(result_dir = env['result_dir'], dtag = dtag, part = part)
  bleu_output_char = "{result_dir}/bleu.{dtag}.{part}.char".format(result_dir = env['result_dir'], dtag = dtag, part = part)

  time_output = "{result_dir}/time.{dtag}.{part}.txt".format(result_dir = env['result_dir'], dtag = dtag, part = part)


  # decode
  batch_size = 32
  if beam_size > 10:
    batch_size = 1

  cmd = """
  t2t-decoder \
  --t2t_usr_dir={user_dir} \
  --data_dir={data_dir} \
  --problem={problem} \
  --model={model} \
  --hparams_set={hparams} \
  --output_dir={train_dir} \
  --checkpoint_path={checkpoint_path} \
  --decode_hparams="beam_size={beam_size_value},alpha={alpha_value},batch_size={batch_size}" \
  --decode_from_file={decode_file} \
  --decode_to_file={decode_output} 
  """.format(**env, **args, beam_size_value = beam_size, alpha_value = alpha, decode_file = decode_file, decode_output = decode_output, batch_size = batch_size, checkpoint_path = checkpoint_path)

  start = datetime.now()
  if not Path(decode_output).exists():
    run_cmd(cmd)
  else:
    print("Already existing: {}".format(bleu_output))

  end = datetime.now()

  spend = end - start

  with open(time_output,'w') as f:
    f.write("{}\n".format(spend.total_seconds())) 
    
  unbpe(decode_output, decode_output_tok)

  # calculate word bleu
  calc_bleu(root_dir, decode_output_tok, decode_reference_tok, bleu_output_tok)
  run_cmd("cat {}".format(bleu_output_tok))

  if env['lang_tgt'] != "en":
    tok2char(decode_output_tok, decode_output_char)
    # calculate char bleu
    calc_bleu(root_dir, decode_output_char, decode_reference_char, bleu_output_char)
    run_cmd("cat {}".format(bleu_output_char))
    
  if checkpoint:
    # set the result_dir back
    env['result_dir'] = result_dir_bk

    
def decode_dev(args, env, checkpoint = None):
  for beam_size in args['beam_size']:
    for alpha in args['alpha']:
      decode_and_bleu(args, env, beam_size, alpha, part = "eval", checkpoint = checkpoint)

def decode_test_all(args, env, checkpoint = None):
  n_test = env['n_test']
  for beam_size in args['beam_size']:
    for alpha in args['alpha']:
      for i in range(n_test):
        decode_and_bleu(args, env, beam_size, alpha, part = "test{}".format(i), checkpoint = checkpoint)
      
def decode_test(args, env, beam_size = None, alpha = None, checkpoint = None):
  if beam_size != None:
    n_test = env['n_test']
    for i in range(n_test):
      decode_and_bleu(args, env, beam_size, alpha, part = "test{}".format(i), checkpoint = checkpoint)
  else:
    # search for the best blue settings and decode
    best_bleu = 0.0
    best_beam = None
    best_alpha = None
    for beam_size in args['beam_size']:
      for alpha in args['alpha']:
        dtag = "b{beam_size}_a{alpha}".format(beam_size = beam_size, alpha = alpha)        
        bleu_output_char = "{result_dir}/bleu.{dtag}.eval.char".format(result_dir = env['result_dir'], dtag = dtag)
        bleu_score, ratio = load_bleu_ratio(bleu_output_char)
        print(bleu_output_char)
        print('BLEU={} ratio={}'.format(bleu_score, ratio))
        if bleu_score >= best_bleu:
          best_bleu = bleu_score
          best_beam = beam_size
          best_alpha = alpha

    print('[Best!] BLEU={} beam={} alpha={}'.format(best_bleu, best_beam, best_alpha))
    
    n_test = env['n_test']
    for i in range(n_test):
      decode_and_bleu(args, env, best_beam, best_alpha, part = "test{}".format(i), checkpoint = checkpoint)

def average(args, env):
  cmd="""
  t2t-avg-all \
  --model_dir={train_dir} \
  --output_dir={train_dir}/avg \
  --n={keep_checkpoint_max}
  """.format(**env, **args)
  run_cmd(cmd)

def export(args, env):
  checkpoint_path = ""
  if "checkpoint_path_export" in args and args["checkpoint_path_export"] != "" :
    checkpoint_path = "{}/{}".format(env['train_dir'], args['checkpoint_path_export'])

  export_dir=  "{train_dir}/export/b{beam_size_export}_a{alpha_export}".format(**env, **args)
    
  cmd="""
  t2t-exporter \
    --t2t_usr_dir={user_dir} \
    --data_dir={data_dir} \
    --problem={problem} \
    --model={model} \
    --hparams_set={hparams} \
    --output_dir={train_dir} \
    --checkpoint_path={checkpoint_path} \
    --decode_hparams="beam_size={beam_size_export},alpha={alpha_export}" \
    --export_dir={export_dir}
  """.format(**env,**args, checkpoint_path = checkpoint_path, export_dir = export_dir)
  run_cmd(cmd)
  

def get_args():
  parser = argparse.ArgumentParser(description='MT pipeline.')
  parser.add_argument('run_json', default=None, help='the run json configuration file. e.g. run.ja_zh.toy.v01.json')
  parser.add_argument('action', default=None, help='supported actions: init, prepare_data, datagen, train, decode_dev, decode_test, decode_test_all, average, export')
  parser.add_argument('--beam_size', default = None, type = int, help = 'beam_size for decoding, e.g. 4')
  parser.add_argument('--alpha', default = None, type = float, help = 'alpha value for decoding, e.g. 0.6')
  parser.add_argument('--checkpoint', default = None, type = str, help = 'checkpoint path for decoding, e.g. model.ckpt-40000, the corresponding model file should be copied to train_dir/bk/ ')
  args = parser.parse_args()
  return args
  
if __name__ == "__main__":

  cmd_args = get_args()
  
  fn_config_json = cmd_args.run_json
  
  with open(fn_config_json) as f:
    args = json.load(f)

  env = {}
  env['run_json'] = os.path.abspath(fn_config_json)
  env = get_env(args, env)
  
  action = cmd_args.action
  
  if action == "decode_test":
    beam_size = cmd_args.beam_size
    alpha = cmd_args.alpha
    checkpoint = cmd_args.checkpoint
    decode_test(args, env, beam_size, alpha)
  elif action == "decode_test_all" or action == "decode_dev":
    checkpoint = cmd_args.checkpoint
    vars()[action](args, env, checkpoint = checkpoint)
  else:
    vars()[action](args, env)
