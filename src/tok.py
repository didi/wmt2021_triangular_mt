"""Tokenization and bpe files
"""

import argparse
import os
import sys

import jieba
import MeCab
from hanziconv import HanziConv
from src.util import run_cmd

#### BPE ####

def apply_bpe(fn_bpe_codes, fn_tok, fn_bpe):
    cmd = "subword-nmt apply-bpe -c {} < {} > {}".format(fn_bpe_codes, fn_tok, fn_bpe)
    run_cmd(cmd)

def unbpe(fn_bpe, fnout):
    fnin = fn_bpe
    with open(fnin) as fin, open(fnout,'w') as fout:
        for line in fin:
            line = line.strip()
            line = line.replace("@@ ","")
            fout.write(line + "\n")

def tok2char(fn_tok, fnout, replace_unk = True):
  fnin = fn_tok
  with open(fnin) as fin, open(fnout,'w') as fout:
    for line in fin:
      if replace_unk:
        line = line.replace("UNK", "*")
      ll = line.strip().split()
      line = "".join(ll)
      line = " ".join(line)
      fout.write(line + "\n")

#### Tokenization ####

def mecab_tok(mecab_tagger, input_str):
    str_tok = mecab_tagger.parse(input_str)
    toks_splited = str_tok.split('\n')
    # drop EOS tag at the end
    list_toks = [i.split('\t')[0] for i in toks_splited][:-2]
    out_str = " ".join(list_toks)
    return out_str


def jieba_tok(line):
    seg_list = jieba.cut(line)
    out_str = " ".join(seg_list)
    return out_str


def char_level_tok(fn_in, fn_out, lang):
    """
    char-level tokenization.
    """    
    if os.path.exists(fn_out):
        print("Already exists: {}".format(fn_out))
        return
    else:
        print("Generating: {}".format(fn_out))

    with open(fn_in, 'r') as fin, open(fn_out, 'w') as fout:
        for line in fin:
            line = line.strip()
            if lang == "zh":
                line = HanziConv.toSimplified(line)
            char_list = [x for x in line]
            out_str = " ".join(char_list)
            fout.write(out_str + "\n")

            
def tok_file(fn_in, fn_out, lang):
    """
    word-level tokenization(jieba for Chinese, MeCab for Japanese).
    """
    if os.path.exists(fn_out):
        print("Already exists: {}".format(fn_out))
        return 
    else:
        print("Generating: {}".format(fn_out))
        
    with open(fn_in, 'r') as fin, open(fn_out, 'w') as fout:
        if lang == "zh":
            for line in fin:
                # clean line
                clean_line = line.strip()
                # do Chinese simplification
                # cc = OpenCC('t2s')
                simplified_line = HanziConv.toSimplified(line)
                # tokenization
                line_segmented = jieba_tok(simplified_line)
                s_line = line_segmented.strip()
                fout.write(s_line + "\n")
        elif lang == "ja":
            ja_tok = MeCab.Tagger("-Ochasen")
            for line in fin:
                # clean line
                clean_line = line.strip()
                # tokenization
                line_segmented = mecab_tok(ja_tok, clean_line)
                s_line = line_segmented.strip()
                fout.write(s_line + "\n")


    
