# Calculate BLEU score
# python bleu.py hyp.txt ref.txt
# both hyp.txt and ref.txt can be normal Chinese or char-split Chinese

# The following commands should provide the same BLEU score
python bleu.py data/hyp.zh data/ref.zh
python bleu.py data/hyp.zh data/ref.zh.char
python bleu.py data/hyp.zh.char data/ref.zh
python bleu.py data/hyp.zh.char data/ref.zh.char
