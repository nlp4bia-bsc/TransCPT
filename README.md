# TransCPT

# Installation

Python version: 3.10.15

Load Anaconda module:
    
```bash
load module anaconda/2023.07
 ```

Create a new environment from the `environment.yml` file:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate cardioberta2
```

After installing things, export the environment:

```bash
conda env export --no-builds | grep -v "^prefix: " > environment.yaml
```

# Train a Masked Language Model

1. Create your default accelerator configuration file depending on your device
    
```bash 
accelerate config
```

2. [Temporarily]: Run scripts/train/CPT.py adjusting the parameters and paths



Source: https://github.com/ayoolaolafenwa/TrainNLP/blob/main/train_masked_language_model.py

**Acceleterate guide**: https://huggingface.co/docs/accelerate/usage_guides/explore