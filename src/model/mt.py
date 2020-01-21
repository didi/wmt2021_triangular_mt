from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensor2tensor.data_generators import problem, generator_utils
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.models.transformer import transformer_big_single_gpu, transformer_base_single_gpu, transformer_tiny, transformer_base
from tensor2tensor.utils import registry

import tensorflow as tf

@registry.register_problem
class MachineTranslation(translate.TranslateProblem):
  """Translation by provding 6 files: 
  - inputs.train.txt
  - targets.train.txt
  - intputs.eval.txt
  - targets.eval.txt
  - inputs.vocab
  - targets.vocab
  """

  @property
  def vocab_type(self):
    return text_problems.VocabType.TOKEN

  @property
  def oov_token(self):
    return "UNK"

  @property
  def source_vocab_name(self):
    return "inputs.vocab"

  @property
  def target_vocab_name(self):
    return "targets.vocab"

  def get_vocab(self, data_dir, is_target=False):
    vocab_filename = os.path.join(data_dir, self.target_vocab_name if is_target else self.source_vocab_name)
    if not tf.gfile.Exists(vocab_filename):
      raise ValueError("Vocab %s not found" % vocab_filename)
    return text_encoder.TokenTextEncoder(vocab_filename, replace_oov=self.oov_token)

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    
    # Vocab
    src_token_path = (os.path.join(data_dir, self.source_vocab_name), self.source_vocab_name)
    target_token_path = (os.path.join(data_dir, self.target_vocab_name), self.target_vocab_name)

    for token_path, vocab_name in [src_token_path, target_token_path]:
      if not tf.gfile.Exists(token_path):
        bpe_vocab = os.path.join(tmp_dir, vocab_name)
        with tf.gfile.Open(bpe_vocab) as f:
          vocab_list = f.read().split("\n")
        vocab_list.append(self.oov_token)
        text_encoder.TokenTextEncoder(
          None, vocab_list=vocab_list).store_to_file(token_path)

    tag = 'eval'
    if dataset_split == problem.DatasetSplit.TRAIN:
      tag = 'train'

    fn_inputs = os.path.join(tmp_dir, "inputs.{}.txt".format(tag))
    fn_targets = os.path.join(tmp_dir, "targets.{}.txt".format(tag))
      
    return text_problems.text2text_txt_iterator(fn_inputs, fn_targets)

  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
    generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
    encoder = self.get_vocab(data_dir)
    target_encoder = self.get_vocab(data_dir, is_target=True)
    return text_problems.text2text_generate_encoded(generator, encoder, target_encoder,
                                                    has_inputs=self.has_inputs)

  def feature_encoders(self, data_dir):
    source_token = self.get_vocab(data_dir)
    target_token = self.get_vocab(data_dir, is_target=True)
    return {
      "inputs": source_token,
      "targets": target_token,
    }

