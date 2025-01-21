import os
import yaml  # type: ignore

script_path = os.path.dirname(os.path.abspath(__file__))

env_file = os.path.join(script_path, 'application-prod.yml')


def read_yaml(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as exc:
            print(f"Error reading YAML file: {exc}")
            return None


env = read_yaml(env_file)
