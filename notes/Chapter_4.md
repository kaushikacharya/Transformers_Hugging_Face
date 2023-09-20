# Sharing Models and Tokenizers

1. [The Hugging Face Hub](#the-hugging-face-hub)
2. [Using pretrained models](#using-pretrained-models)
3. [Sharing pretrained models](#sharing-pretrained-models)
4. [Building a model card](#building-a-model-card)
5. [Part 1 completed!](#part-1-completed)

## The Hugging Face Hub

- [Hugging Face Hub](https://huggingface.co/)
  - Central platform for state-of-the-art models and datasets.
- Focus of current chapter: models
- Next chapter on datasets.
- Models in the Hub not limited to Transformers or even NLP.

## Using pretrained models

- [pipeline](./Chapter_1.md#working-with-pipelines) using [camembert-base](https://huggingface.co/camembert-base) example.
- [CamemBERT website](https://camembert-model.fr/)
- Watch out on using pipeline
  - Chosen checkpoint is suitable for the task it's going to used for.
  - Recommended: Use the task selector in the Hugging Face Hub interface in order to select the appropriate checkpoints.

### Notebook (Using pretrained models)

- [Notebook](../code/notebooks/chapter4/section2_pt.ipynb)

## Sharing pretrained models

- Three ways to create new model repositories:
  - Using the ```push_to_hub``` API
  - Using the ```huggingface_hub``` Python library
  - Using the web interface
- Managing a repo on the Model Hub
  - [![Managing a repo on the Model Hub](https://img.youtube.com/vi/9yY3RB_GSPM/hqdefault.jpg)](https://www.youtube.com/embed/9yY3RB_GSPM)

### Using the ```push_to_hub``` API

- [![Push to Hub API](https://img.youtube.com/vi/Zh0FfmVrKX0/hqdefault.jpg)](https://www.youtube.com/embed/Zh0FfmVrKX0)
- Login methods:
  - In notebook:

    ```Python
      from huggingface_hub import notebook_login

      notebook_login()
    ```
  
  - In a terminal:
  
    ```
    huggingface-cli login
    ```

- Upload from ```Trainer``` API:
  - Set ```push_to_hub=True``` in ```TrainingArguments```
  - ```Trainer``` uploads model to the Hub each time it is saved.
    - Defined by ```save_strategy``` argument in ```TrainingArguments```.
  - After training gets over, one needs to run a final ```trainer.push_to_hub()``` to upload the last version of the model.
    - This generates a model card with
      - all the relevant metadata,
      - reporting the hyperparameters used, and
      - evaluation results

- At a lower level, models, tokenizers and configuration objects can access the Model Hub directly via their ```push_to_hub()``` method.

- [Transformers documentation](https://huggingface.co/docs/transformers/model_sharing) for model sharing
- [huggingface_hub Python package](https://github.com/huggingface/huggingface_hub)

### Using the ```huggingface_hub``` Python library

### Using the web interface

- New repository can be created using the [web interface](https://huggingface.co/new).

### Uploading the model files

- The system to manage files on the Hugging Face Hub is based on
  - git for regular files
  - [git-lfs](https://git-lfs.github.com/) for larger files
- Three different ways of uploading files to the Hub:
  - The ```upload_file``` approach
  - The ```Repository``` class
    - Manages a local repository in a git-like manner.
  - The git-based approach

### Notebook (Sharing pretrained models)

- [Notebook](../code/notebooks/chapter4/section3_pt.ipynb)

## Building a model card

- Concept origin:
  - [Model Cards for Model Reporting by Margaret Mitchell et al](https://arxiv.org/abs/1810.03993)
- The model card usually starts with a very brief, high-level overview of what the model is for, followed by additional details in the following sections:
  - **Model description**
    - Provides basic details of the model that includes
      - Architecture
      - Version
      - Whether introduced in a paper
      - Whether original implementation available
      - Author
      - General information about the model
      - Any copyrights
      - General information about
        - Training procedures
        - Parameters
  - **Intended uses & limitations**
    - Intended use-cases, including
      - Languages
      - Fields
      - Domains
    - Document known out-of-scope areas
    - Areas likely to perform suboptimally
  - **How to use**
    - Include examples of how to use the model.
      - Showcase usage of
        - ```pipeline()```
        - models and tokenizers
  - **Training data**
    - Dataset(s) the model was trained on along with brief description of the dataset(s).
  - **Training procedure**
    - Describe all the relevant aspects of training that are useful from a reproducibility perspective.
    - Includes
      - Preprocessing and postprocessing that were done on the data.
      - Details such as
        - Number of epochs
        - Batch size
        - Learning rate
  - **Variable and metrics**
    - Describe the metrics used for evaluation, and on which dataset split.
    - Goal: Make it easy to compare your model's performance with other models.
  - **Evaluation results**
    - Finally, provide an indication of how well the model performs on the evaluation dataset.

### Example

- Examples of well-crafted model cards:
  - [bert-base-cased](https://huggingface.co/bert-base-cased)
  - [gpt2](https://huggingface.co/gpt2)
  - [distilbert](https://huggingface.co/distilbert-base-uncased)
- Examples from models built by different organizations as available [here](https://github.com/huggingface/model_card/blob/master/examples.md).

### Note

- Model cards are not mandatory for publishing models.

### Model card metadata

- Metadata identifies the categories a model belongs to.
- Example:
  - [camembert-base model card](https://huggingface.co/camembert-base/blob/main/README.md)
- [Full model card specification](https://github.com/huggingface/hub-docs/blame/main/modelcard.md)

## Part 1 completed!

- [Community event](https://huggingface.co/blog/course-launch-event) for the launch of Part 2 of this course.
