from datasets import load_from_disk
from transformers import AutoTokenizer
from datasets import DatasetDict

imdb_data = load_from_disk("/home/bsc/bsc830651/tmp/roberta_training/data/imdb")

# use bert model checkpoint tokenizer
model_checkpoint = "/home/bsc/bsc830651/.cache/huggingface/hub/models--distilbert-base-uncased/snapshots/12040accade4e8a0f71eabdb258fecc2e7e948be/"
# word piece tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

#define tokenize function to tokenize the dataset
def tokenize_function(data):
    result = tokenizer(data["text"])
    return result

# batched is set to True to activate fast multithreading!
tokenize_dataset = imdb_data.map(tokenize_function, batched = True, remove_columns = ["text", "label"],
                                 batch_size=2000)

tokenize_dataset.save_to_disk("data/imdb_tok")