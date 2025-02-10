#!/usr/bin/env python

import datasets  # type: ignore
import math
import torch  # type: ignore

from datetime import datetime
from torch.optim import AdamW  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
from transformers import (  # type: ignore
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    default_data_collator,
)
from transformers.optimization import get_scheduler  # type: ignore
from tqdm.auto import tqdm  # type: ignore

from trans_cpt.utils import (
    concat_chunk_dataset,
    get_accelerator,
    get_logger,
    insert_random_mask,
    set_seed,
    worker_init_fn,
    fill_mask,
)

accelerator = None
logger = None


def get_dataset(data_path):
    print(f"Loading dataset from {data_path}")
    data = datasets.load_dataset(data_path)
    return data


def dataset_tokenization(data, model_checkpoint, chunk_size):
    print("Tokenizing the dataset")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer_dataset = data.map(
        lambda x: tokenizer(x["text"]),
        batched=True,
        remove_columns=["text"],
        batch_size=50
    )
    processed_dataset = tokenizer_dataset.map(
        lambda x: concat_chunk_dataset(x, chunk_size=chunk_size),
        batched=True
    )
    return processed_dataset, tokenizer


def dataset_split(processed_dataset, test_ratio, seed):
    print("Splitting the dataset")

    train_size = int(len(processed_dataset["train"]) * (1 - test_ratio))
    print(f"Train size: {train_size}")

    test_size = int(test_ratio * train_size)
    print(f"Test size: {test_size}")

    return processed_dataset["train"].train_test_split(
        train_size=train_size,
        test_size=test_size,
        seed=seed
    )


def dataset_masking(
    split_dataset,
    tokenizer,
    mlm_probability,
    pad_to_multiple_of,
    batch_size,
    seed
):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=mlm_probability,
        pad_to_multiple_of=pad_to_multiple_of
    )
    mask_function = (lambda x: insert_random_mask(x, data_collator))

    eval_dataset = split_dataset["test"].map(
        mask_function,
        batched=True,
        remove_columns=split_dataset["test"].column_names
    )
    eval_dataset = eval_dataset.rename_columns({
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels"
    })

    if "token_type_ids" in eval_dataset.column_names:
        eval_dataset = eval_dataset.remove_columns("token_type_ids")

    for k, v in eval_dataset[0].items():
        print(f"{k}: {v[:20]}")

    train_dataloader = DataLoader(
        split_dataset["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
        worker_init_fn=worker_init_fn, generator=torch.Generator().manual_seed(seed),
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=default_data_collator,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed),
    )

    return train_dataloader, eval_dataloader


def preprocessing(
    data,
    model_checkpoint,
    test_ratio,
    mlm_probability,
    pad_to_multiple_of,
    batch_size,
    seed,
    chunk_size,
):
    processed_dataset, tokenizer = dataset_tokenization(data, model_checkpoint, chunk_size)
    split_dataset = dataset_split(processed_dataset, test_ratio, seed)
    train_dataloader, eval_dataloader = dataset_masking(
        split_dataset,
        tokenizer,
        mlm_probability,
        pad_to_multiple_of,
        batch_size,
        seed,
    )
    return train_dataloader, eval_dataloader, tokenizer


def load_model(model_checkpoint):
    print(f"Loading model from {model_checkpoint}")
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    return model


def get_training_config(
    model,
    train_dataloader,
    eval_dataloader,
    learning_rate,
    n_warmup_steps,
    num_train_epochs,
):
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch // accelerator.num_processes

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=n_warmup_steps,
        num_training_steps=num_training_steps,
    )

    model, optimizer, train_dataloader, eval_dataloader =\
        accelerator.prepare(
            model,
            optimizer,
            train_dataloader,
            eval_dataloader,
        )

    print(f"Using {accelerator.num_processes} device(s).")

    return (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
        num_training_steps
    )


def training(
    model,
    optimizer,
    train_dataloader,
    eval_dataloader,
    num_training_steps,
    num_train_epochs
):
    accelerator.wait_for_everyone()
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_train_epochs):
        model.train()
        total_train_loss = 0

        for batch in train_dataloader:
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            total_train_loss += loss.item()
            accelerator.backward(loss)
            optimizer.step()
            progress_bar.update(1)

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_perplexity = math.exp(avg_train_loss)

        accelerator.print(
            f">>> TRAIN: Epoch {epoch} -- Loss: {avg_train_loss:.6f} -- "
            f"Average Training Perplexity: {avg_train_perplexity:.6f}"
        )
        accelerator.log({"Loss": {"train": avg_train_loss}}, step=epoch)
        accelerator.log({"Perplexity": {"train": avg_train_perplexity}}, step=epoch)

        model.eval()

        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss.item()
            losses.append(loss)

        if losses:
            losses = torch.tensor(losses)
            avg_eval_loss = torch.mean(losses)
            eval_perplexity = math.exp(avg_eval_loss.item())
        else:
            avg_eval_loss = float('inf')
            eval_perplexity = float('inf')

        accelerator.print(
            f">>> EVAL: Epoch {epoch} -- Loss: {avg_eval_loss:.6f} -- "
            f"Perplexity: {eval_perplexity:.6f}"
        )
        accelerator.log({"Loss": {"valid": avg_eval_loss}}, step=epoch)
        accelerator.log({"Perplexity": {"valid": eval_perplexity}}, step=epoch)

    return model


def monitor_resources():
    import os
    import psutil
    import torch.distributed as dist

    process = psutil.Process(os.getpid())
    cpu_percent = process.cpu_percent(interval=1)
    memory_percent = process.memory_percent()
    gpu_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    rank = dist.get_rank() if dist.is_initialized() else 0

    print(f"Rank {rank}: CPU: {cpu_percent:.1f}%, Memoria: {memory_percent:.1f}%, GPU: {gpu_memory} bytes")


def save_model(model, tokenizer, model_output):
    print("Saving model")
    monitor_resources()
    accelerator.wait_for_everyone()  # Ensure all processes are synchronized before saving
    monitor_resources()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(model_output, save_function=accelerator.save)
        tokenizer.save_pretrained(model_output, save_function=accelerator.save)
    monitor_resources()
    accelerator.wait_for_everyone()
    # import torch.distributed as dist
    # if dist.is_initialized():
    #     dist.barrier()  # Ensure all ranks hit this before exiting
    #     dist.destroy_process_group()

    print(f"Model saved in {model_output}")


def inference_pipeline(vars):
    model_path = vars.get("model_path")
    text = vars.get("text")

    if not model_path:
        print("Not model_path assigned")
        return

    if not text:
        print("No text to predict")
        return

    preds = fill_mask(text, model_path)

    for pred in preds:
        print((f">>> {pred['token_str']}({pred['score']}): {pred['sequence']}"))


def training_pipeline(vars):
    import os       
    ld_library_path = os.environ.get('LD_LIBRARY_PATH')
    print(f"LD_LIBRARY_PATH: {ld_library_path}")
    print(f"Input variables: {vars}")

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    chunk_size = 512
    pad_to_multiple_of = chunk_size

    # Get variables
    seed = vars.get("seed", 42)
    mlm_probability = vars.get("mlm_probability", 0.15)  # Probability of masking a token
    num_train_epochs = vars.get("num_train_epochs", 20)
    batch_size = vars.get("batch_size", 64)
    data_path = vars.get("data_path", "./data/trans_cpt_data/proc")
    model_checkpoint = vars.get(
        "model_checkpoint",
        "/gpfs/projects/bsc14/hf_models/roberta-base-biomedical-clinical-es"
    )
    test_ratio = vars.get("test_ratio", 0.2)
    learning_rate = vars.get("learning_rate", 1e-4)
    n_warmup_steps = vars.get("n_warmup_steps", 0)
    model_name = vars.get("model_name", "CardioBERTa")  # TODO: Change name of project
    model_output = f"/gpfs/projects/bsc14/storage/models/transcpt/{model_name}_{current_time}"
    logs_folder = "./training_logs"

    # Set random to guarantee reproducibility
    set_seed(seed)

    # Initialize accelerator and logger for training
    global accelerator, logger
    accelerator = get_accelerator(log_with="tensorboard", project_dir=logs_folder)
    logger = get_logger(accelerator)

    # 1. Get data
    data = get_dataset(data_path)

    # 2. Load Model
    model = load_model(model_checkpoint)

    # 3. Preprocessing
    train_dataloader, eval_dataloader, tokenizer = preprocessing(
        data,
        model_checkpoint,
        test_ratio,
        mlm_probability,
        pad_to_multiple_of,
        batch_size,
        seed,
        chunk_size,
    )

    # 4. Get model config
    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
        num_training_steps
    ) = get_training_config(
        model, train_dataloader, eval_dataloader, learning_rate, n_warmup_steps, num_train_epochs
    )

    # 5. Training Model
    model = training(
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        num_training_steps,
        num_train_epochs,
    )

    # 6. Save Model
    save_model(model, tokenizer, model_output)

    accelerator.end_training()
