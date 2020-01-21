"""Util functions 
"""

from pathlib import Path
import subprocess as sp
import re


def run_cmd(s, dry_run = False):
  print(s)
  if not dry_run:
    sp.call(s, shell=True)

def cmd_output(s):
  print(s)
  output = sp.check_output(s, shell=True)
  return output
  
### BLEU calculation ###

def calc_bleu(root_dir, fn_target, fn_reference, fn_bleu):
  multi_bleu = "{root_dir}/scripts/multi-bleu-detok.perl".format(root_dir = root_dir)
  cmd = "perl {} {} < {} > {}".format(multi_bleu, fn_reference, fn_target, fn_bleu)
  run_cmd(cmd)

def load_bleu_ratio(fn_bleu):
  with open(fn_bleu) as f:
    line = f.readline()
    ll = line.strip().split()
    score = float(ll[2][:-1])
    ratio = float(ll[5][6:-1])
    return score, ratio
  
def load_force_score(target_bpe, fn_score):
  scores = []
  with open(fn_score) as f:
    for line in f:      
      score = float(line.strip())
      scores.append(score)
      
  ns = []
  with open(target_bpe) as f:
    for line in f:
      n_bpe = len(line.strip().split()) + 1 # +1 for EOS
      ns.append(n_bpe)

  total_s = 0
  total_n = 0
  for score, n in zip(scores, ns):
    total_s += score * n
    total_n += n
      
  return total_n, total_s

      
