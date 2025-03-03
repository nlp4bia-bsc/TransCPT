import os


class Repository:
    def __init__(self, dataset_name, base_path="/storage"):
        self.dataset_name = dataset_name
        self.base_path = base_path

    def get_dataset_path(self):
        raise NotImplementedError

    def dataset_exists(self):
        return os.path.exists(self.get_dataset_path())

    def download_dataset(self):
        raise NotImplementedError
