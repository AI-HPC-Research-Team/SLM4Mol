# LLM4Mol

LLM4Mol offers a comprehensive review of Transformer models and Large Language Models (LLMs) in molecular modeling and design. This project focuses on molecular recognition, generation, optimization, captioning, and property prediction. We have created the ChEBI-20-MM benchmark to conduct 1,263 experiments, aiming to identify key factors influencing these models' performance. Our end-to-end visual analysis also uncovers the chemical intuition embedded in these models.

![Overview of tasks in review](figures/figure1.png)

**Paradigm of the Review**:
- **a. Molecular Modeling and Design Tasks**: Showcasing six task types, their standard modeling methods, and data examples.
- **b. Processes of Tasks**: Dividing molecular data into internal and external categories. Internal information is key to molecular representation and can be converted via various tools, while external information is more accessible and comprehensible to humans. This section outlines the research scope, detailing inputs and outputs for each task.

**Note**: The ChEBI-20-MM benchmark and model sections detail respective directories. Some data and checkpoints might not be available due to size constraints and permissions.

## ChEBI-20-MM
We introduce ChEBI-20-MM, a multi-modal benchmark developed from the ChEBI-20 dataset, integrating data like InChI, IUPAC, and images for a diverse range of molecular tasks.

Contents:
- `train.csv` (26,406 records)
- `validation.csv` (3,300 records)
- `test.csv` (3,300 records)
- `image` folder: Molecular images from Pubchem (e.g., `cid.png`)

Download links:
- [ChEBI-20-MM](https://huggingface.co/datasets/liupf/ChEBI-20-MM)
- [MoleculeNet Datasets](https://moleculenet.org/datasets-1)

## Review of Models
### Developments of Models
A timeline illustrating key developments in transformer-based models for molecular modeling and design.

![Timeline of key developments](figures/figure2.png)

### Categories and Architectures of Models
An overview of model categories and architectures in molecular modeling and design:
- **a. Tasks and Models**: Relationship between six downstream tasks and model architectures.
- **b. Encoder-Decoder Model Architectures**: Three main frameworks: Text-Text, Graph-Text, and Image-Text, each suited for specific molecular tasks.

![Model architectures](figures/figure3.png)

## Evaluation Framework
**Benchmark Experiments Overview**:
Our study includes tests across eight primary model architectures, featuring common backbone models or composite models. We conducted a total of 1,263 experiments, showcasing the adaptability of various models to different molecular tasks.

![Overview of Evaluations](figures/figure4.png)

- `ckpts` - This folder contains checkpoints for finetuning
    - image_ckpts
        - [Swin Transformer-SwinOCSR](https://github.com/suanfaxiaohuo/SwinOCSR)
        - [Swin Transformer](https://github.com/suanfaxiaohuo/SwinOCSR)
        - [ResNet](https://huggingface.co/microsoft/swin-base-patch4-window7-224-in22k)
        - [ViT](https://huggingface.co/google/vit-base-patch16-224)
    - text_ckpts
        - Encoder-only
            - [BERT](https://huggingface.co/bert-base-uncased)
            - [SciBERT](https://huggingface.co/allenai/scibert_scivocab_uncased)
            - [RoBERTa](https://huggingface.co/roberta-base)
            - [ChemBERTa](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1)
        - Decoder-only
            - [GPT-2](https://huggingface.co/gpt2)
            - [GPTNEO](https://huggingface.co/EleutherAI/gpt-neo-125m)
            - [BioGPT](https://huggingface.co/microsoft/biogpt)
        - Encoder-Decoder
            - [BART](https://huggingface.co/facebook/bart-base)
            - [T5](https://huggingface.co/google/flan-t5-base)
            - [T511](https://huggingface.co/google/flan-t5-base)
            - [MolT5-base](https://huggingface.co/laituan245/molt5-base)

### `datasets`
- **ChEBI-20-MM**: A comprehensive multi-modal molecular benchmark dataset.
- **mpp**: The MoleculeNet benchmark dataset, widely used in molecular studies.

### `src`
- **`evaluations`**: Scripts for various evaluation metrics.
    - `fingerprint_metrics.py`: Evaluates molecular fingerprint metrics.
    - `text_translation_metrics.py`: Metrics for assessing text translation accuracy.
    - `mol_translation_metrics.py`: Measures the performance of molecular translation tasks.
- **`feature`**: Contains embedding methods and featurizers.
    - `base_featurizer.py`: Base class for feature embedding.
    - `graph_featurizer.py`: Specializes in graph-based molecular embeddings.
- **`models`**: Core models for single-modal and multi-modal tasks.
    - `molecule`: Houses models specific to single-modal molecular data.
    - `multimodal`: Contains models designed for multi-modal tasks.
    - `metric.py`: Facilitates the loading of various metrics.
    - `init.py`: Initializes model parameters and settings.
    - `model_manager.py`: Manages the loading and handling of models.
- **`utils`**: Utility functions and initializations.
    - `init.py`: General utility tool initialization.
    - `xutils.py`: Advanced and specialized utility tool initialization.
- **`tasks`**: Task-specific scripts and data loaders.
    - `dataset_manager.py`: DataLoader for the ChEBI-20-MM dataset.
    - `task_manager.py`: Manages text generation tasks.
    - `mol_retrieval.py`: Handles the retrieval task operations.
    - `MoleculeNet_loader.py`: DataLoader for the MoleculeNet dataset.
    - `splitters.py`: Implements various data splitting methods.
    - `MPP.py`: Code for molecular property prediction tasks.

**Detailed Parameter Explanations for Tasks**

### Common Command Parameters
- `mode`: Select the operation mode. Options include `data_check`, `encoder_check`, and `eval`.
- `dataset_toy`: Use a smaller, "toy" dataset for quick testing. Set to `toy`.
- `graph_encoder`: Choose a graph encoder. Available options are `gin`, `gat`, `gcn`.
- `text_encoder`: Select a text encoder. Options are `bert`, `scibert`, `roberta`, `chemberta`, `bart`, `t5`, `t511`, `molt5`.
- `image_encoder`: Choose an image encoder from `swin`, `resnet`, `vit`.
- `batch_size`: Set the batch size. Valid choices are 2, 4, 6, 8, 12, 16, 32.

### Task-Specific Command Parameters
- Execute `python task_manager.py` with the following options:
    - `input_modal`: Define the input modality. Choices include `graph`, `SMILES`, `image`, `IUPAC`, `SELFIES`, `InChI`, `caption`.
    - `output_modal`: Specify the output modality. Options are `SMILES`, `caption`, `IUPAC`.
    - `task_name`: Select the task to perform. Available tasks are `molcap`, `mol2IUPAC`, `textmolgen`, `IUPAC2mol`, `image2smi`.
    - `fusion_net`: Choose a fusion network strategy. Options include `add`, `weight_add`, `self_attention`.
    - `decoder`: Select the decoder to use. Choices are `molt5`, `biogpt`, `gpt2`, `gptneo`.

- For molecular retrieval, execute `python mol_retrieval.py`:
    - The `input_modal` and `output_modal` parameters are the same as in `task_manager.py`.

- For executing `python MPP.py` (Molecular Property Prediction):
    - `input_modal`: Define the input modality. Choices include `graph`, `SMILES`, `SELFIES`, `InChI`.
    - `dataset_name`: Specify the dataset for property prediction. Options include `tox21`, `bace`, `bbbp`, `toxcast`, `sider`, `clintox`, `esol`, `lipophilicity`, `freesolv`.
    - `split`: Choose the data splitting method. Options are `scaffold` or `random`.
    - `pool`: Determine the pooling strategy. Choose between `avg` and `max`.

## Citation
```

```
