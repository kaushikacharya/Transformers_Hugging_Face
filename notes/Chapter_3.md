# Fine-tuning a Pretrained Model

1. [Introduction](#introduction)
2. [Processing the data](#processing-the-data)
3. [Fine-tuning a model with the Trainer API](#fine-tuning-a-model-with-the-trainer-api)
4. [A full training](#a-full-training)
5. [Fine-tuning, Check!](#fine-tuning-check)
6. End-of-chapter Quiz

## Introduction

- Learnings in this chapter:
  - How to prepare a large dataset from the Hub.
  - How to use the high-level ```Trainer``` API to fine-tune a model.
  - How to use a custom training loop.
  - How to leverage the Accelerate library to easily run the custom training loop on any distributed setup.

## Processing the data

- We will continue with the example from the [previous chapter](Chapter_2.md).
- Dataset used in this section:
  - MRPC (Microsoft Research Paraphrase Corpus)
    - Pair of sentences with label indicating if the pair paraphrase or not (i.e. both sentences mean the same thing.)
    - One of the 10 datasets for [GLUE](https://gluebenchmark.com/) benchmark.

### Loading a dataset from the Hub

- [Hugging Face Datasets overview (Pytorch) youtube video](https://www.youtube.com/watch?v=_BZearw7f0w)

### Preprocessing a dataset

- GLUE benchmark has 8 out of 10 tasks related to sentence pairs.
- Hence BERT have dual objectives:
  - Language modeling objective
  - Objective related to sentence pairs.
- ```token_type_ids```
  - Presence of this field depends on the model. DistilBERT model's tokenizer does not output this field.
    - ?? Is it that DistilBERT uses [SEP] to identify the next sentence?
    - *They are only returned when the model will know what to do with them*
      - ?? Is it that DistilBERT does not train for objective related to pair of sentences?
- Datasets are [Apache Arrow](https://arrow.apache.org/) files stored on the disk.
- Multiprocessing usage
  - Reason for not using:
    - Tokemizers library already uses multiple threads to tokenize the samples faster.
  - Should be used if not using a fast tokenizer.

### Dynamic padding

- Instead of choosing length of the longest sentence in the entire dataset, choose longest sentence for each batch.
- Cons:
  - Dynamic shapes don't work well on all accelerators.
- Hence apply dynamic padding to CPUs and GPUs. For TPUs, one can use fixed size.

### Notebook (Processing the data)

- [notebook](../code/notebooks/chapter3/section2_pt.ipynb)

## Fine-tuning a model with the Trainer API

- Trainer automatically removes/renames column based on the model signature.

### Training

### Evaluation

- Load the metrics associated with dataset from [Evaluate](https://github.com/huggingface/evaluate/) library.

### Notebook (Fine-tuning a model with the Trainer API)

- [Notebook](../code/notebooks/chapter3/section3.ipynb)

## A full training

### Write your training loop in PyTorch

- This would help in debugging.
- Achieve the same results as we did in the last section without using the ```Trainer``` class.

### Prepare for training

- Postprocessing of tokenized_datasets (auotmatically done bby Trainer)
  - Remove the columns unexpected by model.
  - Rename the column ```label``` to ```labels```.
  - Set the format of the datasetsso they return PyTorch tensors instead of lists.
- Need for learning rate decay scheduler for Adam optimizer
  - [Stackoverflow thread](https://stackoverflow.com/questions/39517431/should-we-do-learning-rate-decay-for-adam-optimizer)
  - [Stackexchange thread](https://stats.stackexchange.com/questions/200063/adam-optimizer-with-exponential-decay)
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) by  Ilya Loshchilov and Frank Hutter.
  - AdamW's algorithm

### The training loop

### The evaluation loop

### Supercharge your training loop with Accelerate

- [Accelerate examples](https://github.com/huggingface/accelerate/tree/main/examples)

### Notebook (A full training)

- [notebook](../code/notebooks/chapter3/section4.ipynb)

## Fine-tuning, Check!

- Chapter recap:
  - [Datasets Hub](https://huggingface.co/datasets)
  - Learned how to load and preprocess datasets, including using dynamic padding and collators.
  - Implemented your own fine-tuning and evaluation of a model.
  - Implemented a lower-level training loop.
  - Used Accelerate to easily adapt training loop so that it works for multiple GPUs and TPUs.

## End-of-chapter Quiz
