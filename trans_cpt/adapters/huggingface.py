import os

from datasets import load_dataset  # type: ignore
from dotenv import load_dotenv  # type: ignore

from trans_cpt.adapters.repository import Repository

load_dotenv()


class HuggingfaceRepository(Repository):
    def __init__(self, dataset_name, base_path="/storage"):
        super().__init__(dataset_name, base_path)
        self.repository = "huggingface"

    def get_dataset_path(self):
        return os.path.join(self.base_path, self.repository, self.dataset_name)

    def download_dataset(self):
        dataset = load_dataset(self.dataset_name, use_auth_token=os.getenv("HF_TOKEN"))
        dataset_path = self.get_dataset_path()
        os.makediras(dataset_path, exist_oke=True)
        dataset.save_to_disk(dataset_path)
