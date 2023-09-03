# Transformer Models

1. [Introduction](#introduction)
2. [Natural Language Processing](#natural-language-processing)
3. [Transformers, what can they do?](#transformers-what-can-they-do)
4. [How do Transformers work?](#how-do-transformers-work)
5. [Encoder models](#encoder-models)
6. [Decoder models](#decoder-models)
7. [Sequence-to-sequence models](#sequence-to-sequence-models)
8. [Bias and limitations](#bias-and-limitations)
9. [Summary](#summary)
10. [End-of-chapter quiz](#end-of-chapter-quiz)

## Introduction

- Hugging Face ecosystem
  - [Transformers](https://github.com/huggingface/transformers)
  - [Datasets](https://github.com/huggingface/datasets)
  - [Tokenizers](https://github.com/huggingface/tokenizers)
  - [Accelerate](https://github.com/huggingface/accelerate)
  - [Hugging Face Hub](https://huggingface.co/models)

- Sections
  - Section #1: Introduction
  - Section #2: Diving in
  - Section #3: Advanced

- Section: Introduction
  - Chapters: 1 to 4
  - Introduction to the main concepts of the Transformers library.

- Section: Diving in
  - Chapters: 5 to 8
  - Basics of Datasets and Tokeziners
  - Then diving into classic NLP tasks.

- Section: Advanced
  - Chapters: 9 to 12
  - Transformer models for speech processing and computer vision.
  - Learn how to buid and share demos of your models.
  - Optimize the models for production environments.

- Recommended courses:
  - Pre:
    - [fast.ai’s Practical Deep Learning for Coders](https://course.fast.ai/)
  - Post:
    - [DeepLearning.AI’s Natural Language Processing Specialization](https://www.coursera.org/specializations/natural-language-processing)

- [Project Ideas](https://discuss.huggingface.co/c/course/course-event/25)

- Jupyter Notebooks
  - Repository: [huggingface/notebooks](https://github.com/huggingface/notebooks)
    - Code from the course.
  - Instructions to generate the notebooks locally provided in [course](https://github.com/huggingface/course#-jupyter-notebooks) repository.

- Chapter content
  - Usage of ```pipeline()``` function to solve NLP tasks.
  - Transformer architecture
  - Distinguish between encoder, decoder, encoder-decoder architectures and use cases.

## Natural Language Processing

- What is NLP?
  - NLP is a field of linguistics and machine learning focused on understanding everything related to human beings.
- Why is it challenging?

## Transformers, what can they do?

### Transformers are everywhere!

- [Model Hub](https://huggingface.co/models)
  - Contains thousands of pretrained models that anyone can download and use.

### Working with pipelines

- ```pipeline()``` regroups together all the steps to go from raw text to usable predictions.
- Includes
  - pre-processing
    - The input is preprocessed into a format the model can understand.
  - model
    - The processed inputs are passed to the model.
  - post-processing
    - The predictions of the model are post-processed, so that it makes sense to us.
- [Documentation](https://huggingface.co/docs/transformers/main_classes/pipelines)

### Zero-shot classification

- Joe Davison, Research Engineer at Hugging Face's [blog](https://joeddav.github.io/blog/2020/05/29/ZSL.html)
  - Latent embedding approach
    - Author experimented with Sentence-BERT.
    - Observation:
      - t-SNE visualization shows that data seems to cluster together by class reasonably well, but the labels are poorly aligned.
      - Reason: Sentence-BERT is designed to learn effective sentence-level, not single or multi-word representations.
    - To use word vectors as our label representations, we would need annotated data to learn an alignment between the S-BERT sequence representations and the word2vec label representations.
    - This issue is addressed by learning a least-squared linear projection matrix $Z$ with L2 regularization from $\Phi_{sent}(V)$ to $\Phi_{word}(V)$ where V are the words in the vocabulary.
    - t-SNE visualization exhibits this projection makes the label embeddings much better aligned with their corresponding data clusters while maintaining the superior performance of S-BERT compared to pooled word vectors.
    - Few-shot learning
      - TODO Bayesian linear regression with a Gaussian prior on the weights centered at the identity matrix and variance controlled by $\lambda$.

### Text generation

### Using any model from the Hub in a pipeline

- [Model Hub](https://huggingface.co/models)

### Mask filling

### Named entity recognition

### Question answering

### Summarization

### Translation

- Model ```Helsinki-NLP/opus-mt-fr-en``` recommends [Moses tokenizer](https://github.com/alvations/sacremoses)

### Notebook

- [notebook](../code/notebooks/chapter1/section3.ipynb)
  - Tasks covered:
    - ```sentiment-analysis```

## How do Transformers work?

### A bit of Transformer history

- The Transformer architecture was introduced in June 2017.
- Focus of original research: translation tasks.
- Followed by the introduction of several influential models, including:
  - DistilBERT:
    - A distilled version of BERT
    - 60 % faster, 40 % lighter in memory
    - Still retains 97 % of BERT's performance.
- Broadly, they can be grouped into three categories:
  - GPT-like (also called *auto-regressive* Trensformer models)
  - BERT-like (also called *auto-encoding* Transformer models)
  - BART/T5-like (also called sequence-to-sequence Transformer models)

### Transformers are language models

- All the transformer models mentioned above have been trained as *language models*.
- Training:
  - On large amounts of raw text.
  - Self-supervised fashion
- The model develops a **statistical understanding** of the language it has been trained on.
- Transfer learning:
  - The general pretrained model is fine-tuned in a supervised way.
  - Language modeling
    - Causal language modeling
    - Masked language modeling

### Transformers are big models

- Number of parameters (in Millions) for the models over the past few years.
- CO2 emissions of training a model compared against CO2 emissions for a variety of human activities.

### Transfer Learning

- Transfer Learning
  - The act of initializing a model with anothe model's weights.
  - Usually applied by dropping the head of the pretrained model while keeping its body.
- BERT training corpus:
  - Wikipedia
  - 11k books
- Fine-tuning:
  - First acquire a pretrained language model.
  - Perform additional training with a dataset specific to your task.

### General architecture

- [Youtube video on transformer architecture](https://www.youtube.com/watch?v=H39Z_720T5s)
- Encoder:
  - Self-attention
  - Bi-directional
- Decoder:
  - Masked self-attention
  - [Auto-regressive](https://en.wikipedia.org/wiki/Autoregressive_model)
    - Output variable depends linearly on
      - Its own previous values
      - Stochastic term
  - Uni-directional

### Introduction (Transformer)

- Two primary blocks of the model
  - Encoder (left)
    - Receives an input and builds a representation of its features.
    - Model is optimized to acquire understanding from the input.
  - Decoder (right)
    - Uses the encoder's representation (features) along with other inputs to generate a target sequence.
    - Model is optimized for generating outputs.
- Each of these parts can be used independently, depending on the task:
  - **Encoder-only models**
    - Good for tasks that require understanding of the input.
    - e.g. sentence classification, named entity recognition
  - **Decode-only models**
    - Good for generative tasks.
    - e.g. text generation
  - **Encoder-decoder models** or **sequence-to-sequence models**
    - Good for generative tasks that require an input.
    - e.g. translation, summarization

### Attention layers

- A word by itself has a meaning, but that meaning is deeply affected by the context, which can be any other word (or words) before or after the word being studied.
- This is explained with example of translation from English to French for the sentence:
  - *You like this course*.
    - For translation of "like", we need to attend to the adjacent word "you" because in French the verb "like" is conjugated differently depending on the subject.

### The original architecture

- Originally designed for translation.

### Architectures vs checkpoints

- Architecture
  - Skeleton of the model.
  - Definition of each layer and each operation.
- Checkpoint
  - Weights that will be loaded in a given architecture.
- Model
  - This course will specify *architecture* or *checkpoint* when it metters to reduce ambiguity.

## Encoder models

- The vector representation of the words holds the means of the words in the context of the sentence.

## Decoder models

- Auto-regressive models
  - At each stage, for a given word the attention layers can only access the words positioned before it in the sentence.
- The pretraining of decoder models usually revolves around predicting the next word in the sentence.
- The masked self-attention layer hides the values of context on the right.
- These models are best suited for tasks involving text generation.

## Sequence-to-sequence models

- Sequence-to-sequence tasks
  - Many-to-many
    - translation
    - summary
- Weights are not necessarily shared across the encoder and decoder.
- Encoder-decoder models use both parts of the Transformer architecture.
- At each level, the attention layers of the encoder can access all the words in the initial sentence, whereas the attention layers of the decoder can only access the words positioned before a given word in the input.
- Sequence-to-sequence models are best suited for tasks revolving around generating new sentences depending on a given input, such as summarization, translation, or generative question answering.

## Bias and limitations

- [Notebook](../code/notebooks/chapter1/section8.ipynb)
  - The fill mask tasks shows the gender bias in the trained model.

## Summary

- In this chapter, we saw how to approach different NLP tasks using the high-level ```pipeline()``` function from Transformers.
- We saw how to search for and use models in the Hub.
- We discussed how Transformer models work at a high level, and talked about the importance of transfer learning and fine-tuning.

## End-of-chapter quiz

- Possible source for the bias observed in a model
  - The metric the model was optimizing for is biased.
    - A less obvious source of bias is the was the model is trained.
