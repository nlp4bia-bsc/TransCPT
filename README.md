# TransCPT

This repository use Cosmos Model library to execute a CPT of huggingface models

# Installation
```bash
pip install -r requirements.txt
 ```

# Train a Masked Language Model

1. Create your default accelerator configuration file depending on your device
    
```bash 
accelerate config
```

2. Adapt the variables you want to train the model, you can see the variables to adapt in the training_pipeline function of /trans_cpt/training.py

3. To monitor the logs run tensorboard:

```bash
tensorboard --logdir models/MODEL_NAME/
``` 

4. To execute the training you have to execute both cell in the main.ipynb

Source: https://github.com/ayoolaolafenwa/TrainNLP/blob/main/train_masked_language_model.py

**Acceleterate guide**: https://huggingface.co/docs/accelerate/usage_guides/explore