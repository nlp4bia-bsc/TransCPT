#!/usr/bin/env python

from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM
from torch.optim import AdamW
from accelerate import Accelerator
from transformers.optimization import get_scheduler
from transformers import default_data_collator
from tqdm.auto import tqdm
import math
import os

import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import sys

sys.path.append("/home/bsc/bsc830651/tmp/roberta_training/")
from src.utils import concat_chunk_dataset, insert_random_mask, fill_mask, set_seed, worker_init_fn


current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

seed = 42
set_seed(seed) # Set random seed to guarantee reproducibility

mlm_probability = 0.15 # Probability of masking a token
pad_to_multiple_of = 8 # This value depends on the numeric types that are used. For float16 -> 8, float8 -> 16
num_train_epochs = 20
batch_size = 64
data_path = "data/cardio_data/proc"
model_checkpoint = "/gpfs/projects/bsc14/hf_models/roberta-base-biomedical-clinical-es"
test_ratio = 0.2
learning_rate = 1e-4
n_warmup_steps = 0
model_name = "CardioBERTa"
model_output = os.path.join("models", model_name, current_time)   # Directory to save the models

# writer_train = SummaryWriter(log_dir=os.path.join(model_output, "runs/train"))
# writer_eval = SummaryWriter(log_dir=os.path.join(model_output, "runs/eval"))
# accelerator = Accelerator(log_with="tensorboard", project_dir='./output')

from accelerate.utils import LoggerType
# Initialize accelerator for training
accelerator = Accelerator(log_with="tensorboard", project_dir=model_output)
accelerator.init_trackers("runs")

# Create a logger
logging_level = logging.INFO if accelerator.is_main_process else logging.WARNING
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging_level)
logger = logging.getLogger(__name__)

#################################################### LOAD DATA #########################################################

logger.info("="*50)
logger.info("Loading the dataset")
logger.info("="*50)

logger.info(f"Loading dataset from {data_path}")
data = load_from_disk(data_path)

#################################################### TOKENIZATION #########################################################
logger.info("="*50)
logger.info("Tokenizing the dataset")
logger.info("="*50)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenize_dataset = data.map(lambda x: tokenizer(x["text"]), batched=True, remove_columns=["text"], batch_size=50)
processed_dataset = tokenize_dataset.map(concat_chunk_dataset, batched=True)

#################################################### DATA SPLITTING #########################################################
logger.info("="*50)
logger.info("Splitting the dataset")
logger.info("="*50)

train_size = int(len(processed_dataset["train"]) * (1 - test_ratio))
logger.info(f"TRAIN SIZE: {train_size}")

test_size = int(test_ratio * train_size)
logger.info(f"TEST SIZE: {test_size}")

split_dataset = processed_dataset["train"].train_test_split(train_size=train_size, test_size=test_size, seed=seed)

#################################################### DATA MASKING #########################################################

# Apply random masking once on the whole test data
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, 
                                                mlm_probability=mlm_probability,
                                                pad_to_multiple_of=pad_to_multiple_of)

mask_function = lambda x: insert_random_mask(x, data_collator)

eval_dataset = split_dataset["test"].map(mask_function, batched=True, remove_columns=split_dataset["test"].column_names)
eval_dataset = eval_dataset.rename_columns({"masked_input_ids": "input_ids", "masked_attention_mask": "attention_mask", "masked_labels": "labels"})

for k, v in eval_dataset[0].items():
    logger.info(f"{k}: {v[:20]}")


train_dataloader = DataLoader(split_dataset["train"], shuffle=True, 
                              batch_size=batch_size, collate_fn=data_collator, 
                              worker_init_fn=worker_init_fn, generator=torch.Generator().manual_seed(seed))

eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, 
                             collate_fn=default_data_collator, worker_init_fn=worker_init_fn, 
                             generator=torch.Generator().manual_seed(seed))

#################################################### MODEL LOADING #########################################################

logger.info("="*50)
logger.info("Loading the model")
logger.info("="*50)

logger.info(f"Loading model from {model_checkpoint}")
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

# Set the number of epochs
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch // accelerator.num_processes

# Set the optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, 
                             num_warmup_steps=n_warmup_steps, 
                             num_training_steps=num_training_steps)

# DP Parallelization: 
model, optimizer, train_dataloader, eval_dataloader =\
    accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)

logger.info(f"Using {accelerator.num_processes} device(s).")

progress_bar = tqdm(range(num_training_steps))

# Training loop
for epoch in range(num_train_epochs):
    model.train()
    total_train_loss = 0
    # print(len(train_dataloader))

    for batch in train_dataloader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        total_train_loss += loss.item()
        accelerator.backward(loss)
        optimizer.step()
        # lr_scheduler.step()
        progress_bar.update(1)

    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_train_perplexity = math.exp(avg_train_loss)

    # Only the main process should log and save
    # if accelerator.is_local_main_process:
    accelerator.print(f">>> TRAIN: Epoch {epoch} -- Loss: {avg_train_loss:.6f} -- Average Training Perplexity: {avg_train_perplexity:.6f}")
    accelerator.log({"Loss":{"train": avg_train_loss}}, step=epoch)
    accelerator.log({"Perplexity": {"train": avg_train_perplexity}}, step=epoch)

        # # Evaluation (only in main process)
        # unwrapped_model = accelerator.unwrap_model(model)
        # unwrapped_model.eval()
    
    # accelerator.wait_for_everyone()
    model.eval()

    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss.item()
        losses.append(loss)


    # After gathering losses, calculate average loss and perplexity
    if losses:
        losses = torch.tensor(losses)  # Convert to tensor if needed
        avg_eval_loss = torch.mean(losses)
        eval_perplexity = math.exp(avg_eval_loss.item())  # Use .item() to convert to scalar
    else:
        avg_eval_loss = float('inf')  # Or handle as needed
        eval_perplexity = float('inf')

    # if accelerator.is_main_process:
        # if accelerator.is_local_main_process:
    accelerator.print(f">>> EVAL: Epoch {epoch} -- Loss: {avg_eval_loss:.6f} -- Perplexity: {eval_perplexity:.6f}")

    accelerator.log({"Loss": {"valid": avg_eval_loss}}, step=epoch)
    accelerator.log({"Perplexity": {"valid": eval_perplexity}}, step=epoch)

    # # Save model
    # # accelerator.wait_for_everyone()  # Ensure all processes are synchronized before saving

    # if accelerator.is_main_process:
    # unwrapped_model = accelerator.unwrap_model(model)
    # unwrapped_model.save_pretrained(model_output, save_function=accelerator.save)
    # tokenizer.save_pretrained(model_output, save_function=accelerator.save)

        # Ensure only the main process closes the TensorBoard writer

accelerator.end_training()

# logger.info("Training completed successfully!")

# logger.info("="*50)
# logger.info("Model Evaluation")
# logger.info("="*50)

# text = "Con el diagnóstico de endocarditis infecciosa sobre válvula protésica por Bacteroides fragilis,"\
#         "se comenzó tratamiento con metronidazol 500 mg/8 horas y amoxicilina-clavulánico 1000 mg/200mg/8 "\
#         "horas intravenoso. La paciente permaneció <mask> durante todo el ingreso, senegativizaron los hemocultivos "\
#         "de forma precoz y evolucionó de forma favorables de su ligera descompensación cardiaca con tratamiento"\
#         "diurético. Tras 6 semanas de tratamiento antibiótico intravenoso dirigido, estando estable hemodinámicamente "\
#         "y en buena clase funcional se dio de alta hospitalaria."

# preds = fill_mask(text, model_output)

# # if accelerator.is_local_main_process:
# for pred in preds:
#     logger.info(f">>> {pred['token_str']}({pred['score']}): {pred['sequence']}")

# logger.info("THE END")
