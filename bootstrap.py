from pathlib import Path
from datasets import load_dataset
import urllib.request
import yaml


def download_tasks(config):

    load_dataset(
        config["dataset"]["repository"],
        config["dataset"]["tasks"],
        split="test",
        cache_dir=config["dataset"]["tasks_directory"],
    )


def download_tool_descriptions(config):

    tools_directory = Path(config["dataset"]["tools_directory"])

    tools_directory.mkdir(parents=True, exist_ok=True)

    for dataset_name in config["dataset"]["tools"]:

        tool_desc_url = (
            f"https://huggingface.co/datasets/"
            f"{config['dataset']['repository']}/"
            f"resolve/main/data_{dataset_name}/tool_desc.json"
        )

        output_file = (tools_directory / f"{dataset_name}_tool_desc.json")

        urllib.request.urlretrieve(tool_desc_url, output_file)


def main():

    with open("config/config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    download_tasks(config)
    download_tool_descriptions(config)


if __name__ == "__main__":
    main()