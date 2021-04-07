#!/usr/bin/python
#
# How to run: 
# python bleu.py hyp.txt ref.txt
#
# both hyp.txt and ref.txt can be either normal Chinese or char-split Chinese
# 

import subprocess as sp
import sys
import os

def char_level_tok(fn_in, fn_out):
    """
    char-level tokenization.
    """    
    with open(fn_in, 'r') as fin, open(fn_out, 'w') as fout:
        for line in fin:
            line = "".join(line.strip().split())
            char_list = [x for x in line]
            out_str = " ".join(char_list)
            fout.write(out_str + "\n")



def calc_bleu(fn_target, fn_reference):
    multi_bleu = "multi-bleu-detok.perl"
    cmd = "perl {} {} < {}".format(multi_bleu, fn_reference, fn_target)
    run_cmd(cmd)
  
def run_cmd(s, dry_run = False):
    if not dry_run:
        sp.call(s, shell=True)

def rm(fn):
    cmd = "rm {}".format(fn)
    run_cmd(cmd)
        
def main():
    fn_hyp = sys.argv[1]
    fn_ref = sys.argv[2]
    fn_hyp_char = fn_hyp + ".tmp.char"
    fn_ref_char = fn_ref + ".tmp.char"
    
    char_level_tok(fn_hyp, fn_hyp + ".tmp.char")
    char_level_tok(fn_ref, fn_ref + ".tmp.char")

    calc_bleu(fn_hyp_char, fn_ref_char)
    
    rm(fn_hyp_char)
    rm(fn_ref_char)
    

if __name__ == "__main__":
  main()
