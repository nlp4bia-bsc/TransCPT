import numpy as np  # type: ignore
import logging
import random
import torch  # type: ignore

from accelerate import Accelerator  # type: ignore
from transformers import pipeline  # type: ignore


def set_seed(seed: int):
    """
    Set a seed to guarantee reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_accelerator(log_with, project_dir):
    accelerator = Accelerator(log_with=log_with, project_dir=project_dir)
    accelerator.init_trackers("runs")
    return accelerator


def get_logger(accelerator):
    logging_level = logging.INFO if accelerator.is_main_process else logging.WARNING
    logging.basicConfig(
        format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
        level=logging_level
    )
    logger = logging.getLogger(__name__)
    return logger


def concat_chunk_dataset(data, chunk_size=512):
    """
    Receives data in the form of a dictionary with keys as the column names and values
    as the list of tokenized text. It concatenates the tokenized text and labels into chunks
    of size chunk_size.

    Output: A dictionary with keys as the column names and values as the list of tokenized
            text in chunks of size chunk_size. In key "label" the values are the same as the
            input_ids of the processed text data for later masking.

    **Important**: Chunk size should be less than or equal to 512 and conditions the context
    window size of the model.
    """

    # concatenate tokens id and labels
    concatenated_sequences = {k: sum(data[k], []) for k in data.keys()}

    # compute length of concatenated texts
    total_concat_length = len(concatenated_sequences[list(data.keys())[0]])

    # drop the last chunk if is smaller than the chunk size
    total_length = (total_concat_length // chunk_size) * chunk_size

    # split the concatenated sentences into chunks using the total length
    result = {
        k: [
            t[i: i + chunk_size] for i in range(0, total_length, chunk_size)
        ] for k, t in concatenated_sequences.items()
    }

    # we create a new labels column which is a copy of the input_ids of the processed text data,
    # the labels column serve as ground truth for our masked language model to learn from.

    result["labels"] = result["input_ids"].copy()

    return result


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def insert_random_mask(batch, data_collator):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)

    return {f"masked_{k}": v.numpy() for k, v in masked_inputs.items()}


def fill_mask(text, model):
    pred_model = pipeline("fill-mask", model=model, device="cuda")

    preds = pred_model(text)
    return preds
