# Transformers Course by Hugging Face

## Course Info

- [Course url](https://huggingface.co/learn/nlp-course)
- Instructor: Hugging Face

## Jupyter Notebooks

- Instructions to generate the [notebooks](https://github.com/huggingface/notebooks/tree/main/course/en) locally provided in the [course](https://github.com/huggingface/course#-jupyter-notebooks) repository.

## Chapters

|#|Chapter|
|-|-------|
|0|[Setup](./notes/Chapter_0.md)|
|1|[Introduction](./notes/Chapter_1.md)|
|2|[Using Transformers](./notes/Chapter_2.md)|
|3|[Fine-tuning a Pretrained Model](./notes/Chapter_3.md)|
|4|[Sharing Models and Tokenizers](./notes/Chapter_4.md)|

## Assignments

|Chapter #|Assignment|
|---------|----------|
|1|[Pipeline](./code/notebooks/chapter1/section3.ipynb)|
|1|[Bias](./code/notebooks/chapter1/section8.ipynb)|
|2|[Behind the pipeline (PyTorch)](./code/notebooks/chapter2/section2_pt.ipynb)|
|2|[Models (PyTorch)](./code/notebooks/chapter2/section3_pt.ipynb)|
|2|[Tokeniers (PyTorch)](./code/notebooks/chapter2/section4_pt.ipynb)|
|2|[Handling multiple sequences (PyTorch)](./code/notebooks/chapter2/section5_pt.ipynb)|
|2|[Putting it all together (PyTorch)](./code/notebooks/chapter2/section6_pt.ipynb)|
|3|[Processing the data (PyTorch)](./code/notebooks/chapter3/section2_pt.ipynb)|
|3|[Fine-tuning a model with the Trainer API](./code/notebooks/chapter3/section3.ipynb)|
|3|[Full training](./code/notebooks/chapter3/section4.ipynb)|
|4|[Using pretrained models](./code/notebooks/chapter4/section2_pt.ipynb)|
|4|[Sharing pretrained models](./code/notebooks/chapter4/section3_pt.ipynb)|

## Best Practices

- Installing Python packages in Jupyter Notebooks
  - [vscode jupyter](https://github.com/microsoft/vscode-jupyter/wiki/Installing-Python-packages-in-Jupyter-Notebooks)
    - recommends
      - ```%pip install```
      - ```python -m pip install```
    - discourages
      - ```!pip install```
        - as this could install packages in the wrong environment.
