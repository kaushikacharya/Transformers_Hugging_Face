# Transformer Models

1. [Introduction](#introduction)
2. [Natural Language Processing](#natural-language-processing)
3. [Transformers, what can they do?](#transformers-what-can-they-do)

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
