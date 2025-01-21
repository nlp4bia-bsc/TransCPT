from dotenv import load_dotenv  # type: ignore

from trans_cpt.adapters.huggingface import HuggingfaceRepository
from trans_cpt.environment import env

load_dotenv()

repository_controllers = {
    "huggingface": HuggingfaceRepository,
}


def get_dataset(repository, dataset_name):
    print(repository)
    print(dataset_name)
    print(env["rocket_storage_path"])

    repository_controller = repository_controllers.get(repository)
    controller = repository_controller(dataset_name, env["rocket_storage_path"])
    controller.download_dataset()
