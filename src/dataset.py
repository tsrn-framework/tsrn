import json
from pathlib import Path

from datasets import load_dataset


def load_tasks(repository, dataset_name, cache_dir):
        
    dataset = load_dataset(
        repository,
        dataset_name,
        split="test",
        cache_dir=cache_dir,
    )

    dataset = dataset.filter(lambda sample: sample["type"] == "chain")

    return dataset


def load_tools(tools_directory, tool_datasets):

    tools = {"nodes": []}

    tools_directory = Path(tools_directory)

    for dataset_name in tool_datasets:

        tools_file = (tools_directory / f"{dataset_name}_tool_desc.json")

        with open(tools_file, "r", encoding="utf-8") as file:

            data = json.load(file)
            for tool in data["nodes"]:
                tools["nodes"].append(tool)

    return tools