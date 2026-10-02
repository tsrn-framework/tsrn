from src.execute import execute
from src.evaluate import evaluate

import yaml


def main():

    with open("config/config.yaml", "r", encoding="utf-8") as file:
	    config = yaml.safe_load(file)

    execute(config)
    evaluate(config)


if __name__ == "__main__":
    main()
