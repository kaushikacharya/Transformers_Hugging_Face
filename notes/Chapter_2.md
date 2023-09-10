# Using Transformers

1. [Introduction](#introduction)
2. [Behind the pipeline](#behind-the-pipeline)
3. [Models](#models)
4. [Tokenizers](#tokenizers)
5. [Handling multiple sequences](#handling-multiple-sequences)
6. [Putting it all together](#putting-it-all-together)
7. [Basic usage completed!](#basic-usage-completed)
8. End-of-chapter quiz

## Introduction

- The main features of the library:
  - Ease of use
  - Flexibility
    - Core: All models are simple ```nn.Module``` or ```tf.keras.Model```.
    - Can be handled like any other models in their respective machine learning (ML) frameworks.
  - Simplicity
    - Each model has its own layers unlike other ML libraries which are built on modules that are shared across files.
- Covered in this chapter
  - Begins with an end-to-end example using a model and a tokenizer together to replicate the ```pipeline()``` function introduced in [Chapter 1](./Chapter_1.md).

## Behind the pipeline

### Preprocessing with a tokenizer

- Tokenizer
  - Raw text => Tokens => Special tokens => Input IDs
  - Special tokens:
    - [CLS]
    - [SEP]
  - ```AutoTokenizer```
    - Loads tokenizer for any checkpoint.
    - ```tokenizer = AutoTokenizer.from_pretrained(checkpoint)```
  - Responsibilities:
    - Splitting the input into tokens:
      - words, subwords, or symbols (like punctuation).
    - Mapping each token to an integer.
    - Adding additional inputs that may be useful to the model.
- Model
  - Input IDs => Logits

### Going through the model

- ```model = AutoModel.from_pretrained(checkpoint)```
- For the base Transformer model, it outputs
  - Hidden states, also known as features.
    - High dimensional vector representing the contextual understanding of that input by the Transformer model.
  - These outputs are usually inputs to another part of the model, known as head.

- **A high-dimensional vector?**
  - Usual three dimensions of the vector output:
    - Batch size
    - Sequence length
    - Hidden size

- **Model heads: Making sense out of numbers**
  - The model head take the high-dimensional vector of hidden states as input and project them onto a different dimension.
  - Usually composed of one or a few linear layers.
  - Different architectures available in Transformers, with each one designed around tackling a specific task includes:
    - *Model (retrieve the hidden states)
    - *ForSequenceClassification

### Postprocessing the output

### Notebook (Behind the pipeline)

- [notebook](../code/notebooks/chapter2/section2_pt.ipynb)

## Models

### Creating a Transformer

- **Different loading methods**
- **Saving methods**
  - ```model.save_pretrained()```
  - Saves two files:
    - config.json
      - Stores the attributes necessary to build the model architecture.
    - pytorch_model.json
      - Known as the *state dictionary*.
      - Contains model's weights.

### Using a Transformer model for inference

- **Using the tensors as inputs to the model**

### Models Notebook

- [notebook](../code/notebooks/chapter2/section3_pt.ipynb)

## Tokenizers

- Three different algorithms for tokenizer:
  - Word-based
  - Character-based
  - Subword-based

- Word-based tokenizer
  - Text splitting
    - Based on spaces or punctuation.
  - Cons:
    - Very similar words will have different representation.
      - e.g. dog, dogs
        - Word representations would ignore that one is plural of another.
    - Large vocabulary size.
      - If we limit vocabulary size, then all the OOV words will have the same representation.

- Character-based tokenizer
  - Two primary benefits
    - Much smaller vocabulary
    - Much fewer out-of-vocabulary tokens.
  - Cons
    - Unlike ideogram languages e.g. Chinese, characters doesn't mean a lot on its own.
    - Large number of tokens to be processed by the model.

- Subword tokenization
  - Philosophy:
    - Frequently used words should not be split into smalled subwords.
    - Rare words should be decomposed into meaningful subwords.
  - Useful in agglutinative langauges:
    - e.g. Turkish
    - One can form (almost) arbitrarily long complex words by stringing together subwords.
  - Techniques:
    - Byte-level BPE (usage e.g. GPT-2)
    - WordPiece (usage e.g. BERT)
    - SentencePiece or Unigram

- Loading and saving
  - Loading the BERT tokenizer
    - ```from transformers import BertTokenizer```
    - Alternatively
      - ```from transformers import AutoTokenizer```

### Encoding

- Special tokens
  - BERT
    - Uses ```[CLS]```, ```[SEP]```
  - RoBERT
    - Uses ```<s>```, ```</s>```

- Encoding
  - Two step process
    - Tokenization

      ```python
      sequence = <string>
      tokens = tokenizer.tokenize()
      ```

    - Conversion to input IDs

      ```python
      ids = tokenizer.convert_tokens_to_ids(tokens)
      ```

- Decoding
  - Going the other way around
    - From vocabulary indices to string.
  
    ```python
    decoded_string = tokenizer.decode(ids)
    ```

- ### Notebook (Tokenizers)

  - [notebook](../code/notebooks/chapter2/section4_pt.ipynb)

## Handling multiple sequences

- Batching
  - Act of sending multiple sentences through the model, all at once.

### Models expect a batch of inputs

- Tensors are of rectangular shape.
  - Padding mitigates the challenge of different input lengths.

### Padding the inputs

- Special word *padding token* is added to the sentence with fewer values.
- Padding token ID:
  - ```tokenizer.pad_token_id```

### Attention masks

- Attention mask tensors are filled with
  - 1s: Indicate the corresponding tokens should be attended to.
  - 0s: Corresponding tokens to be ignored by the attention layers of the model.

### Longer sequences

### Notebook (Handling multiple sequences)

- [Notebook](../code/notebooks/chapter2/section5_pt.ipynb)

## Putting it all together

### Special tokens

- Special token ID is added at the beginning and the end.
  - [CLS] at the beginning
  - [SEP] at the end
- Variations:
  - Some models may add different special tokens.
  - Some models may add only at the beginning or at the end.
  - Some models may not add any special token.

### Wrapping up: From tokenizer to model

### Notebook (Putting it all together)

- [notebook](../code/notebooks/chapter2/section6_pt.ipynb)

## Basic usage completed!

- Chapter recap:
  - Learned the basic building blocks of a Transformer model.
  - Learned what makes up a tokenization pipeline.
  - Use a Transformer model in practice.
  - Learned how to leverage a tokenizer to convert text to tensors that are understandable by the model.
  - Set up a tokenizer and a model together to get from text to predictions.
  - Learned the limitations of input IDs, and learned about attention masks.
  - Payed around with versatile and configurable tokenizer methods.
