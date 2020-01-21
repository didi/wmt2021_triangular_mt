"""Tokenize the raw data folder

python dp_tokenize.py data/raw/dev.ja_zh.toy.v01

OR 

# only split into charactoer

python dp_tokenize.py data/raw/dev.ja_zh.toy.v01 char

OR 

# only do tokenization 

python dp_tokenize.py data/raw/dev.ja_zh.toy.v01 token


"""

import argparse
import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, root_dir)

from src.tok import char_level_tok, tok_file
from filelock import Timeout, FileLock


if __name__ == '__main__':
    #main()
    """
    ## This python script does the following pipeline steps:
    ## Step1: clean, strip leading and ending whitespace.
    ## Step2: do simplification for Chinese data.
    ## Step3: do tokenization
    ## Step4: Split into Characters if it's Chinese/Japanese && is dev/test
    """
    raw_dir = sys.argv[1]
    do_tokenize = True
    do_split_into_char = True

    if len(sys.argv) > 2:
        if sys.argv[2] == "char":
            do_tokenize = False
        elif sys.argv[2] == "token":
            do_split_into_char = False


    fn_lock = os.path.join(raw_dir, 'lock')
    lock = FileLock(fn_lock)
    
    with lock:
    
        dir_name = os.path.basename(raw_dir)

        is_train = False
        if dir_name.startswith('train'):
            is_train = True

        print("[Tokenization] for folder: {}".format(raw_dir))

        for lang in ['zh','ja']:
            fn_in = os.path.join(raw_dir, lang)
            if os.path.exists(fn_in):
                # token file
                fn_out = os.path.join(raw_dir, "{}.tok".format(lang))
                if do_tokenize: 
                    tok_file(fn_in, fn_out, lang)

                # char file
                if (not is_train) and (lang != "en") and do_split_into_char:
                    fn_out = os.path.join(raw_dir, "{}.char".format(lang))
                    char_level_tok(fn_in, fn_out, lang)
                    


                    
                
    
    
    
    
    
    
    
