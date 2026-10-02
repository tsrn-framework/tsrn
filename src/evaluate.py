import csv
import json
from pathlib import Path

from src.dataset import load_tasks
from src.experiments import generate_experiments

from statistics import stdev

from tqdm import tqdm
import sys



# Helper
def extract_expected_tools(task):
    tool_nodes = json.loads(task["tool_nodes"])
    return [tool["task"] for tool in tool_nodes]

# Helper
def extract_predicted_tools(result):
    return [tool["tool"] for tool in result["predicted_tools"]]













# --------------------------------------------------
# Prediction Metrics
# --------------------------------------------------

def compute_tool_exact_match(results_file, expected_tools):

    exact_matches = 0
    total_tasks = 0

    with open(results_file, "r", encoding="utf-8") as file:

        for line in file:

            result = json.loads(line)

            task_expected_tools = expected_tools[result["task_id"]]
            task_predicted_tools = extract_predicted_tools(result)

            if (set(task_expected_tools) == set(task_predicted_tools)):
                exact_matches += 1

            total_tasks += 1

    if total_tasks == 0:
        tool_exact_match = 0.0
    else:
        tool_exact_match = exact_matches / total_tasks

    return {
        "tool_exact_match": tool_exact_match,
    }



def compute_tool_precision(results_file, expected_tools):

    precisions = []

    with open(results_file, "r", encoding="utf-8") as file:

        for line in file:

            result = json.loads(line)

            task_expected_tools = expected_tools[result["task_id"]]
            task_predicted_tools = extract_predicted_tools(result)

            expected = set(task_expected_tools)
            predicted = set(task_predicted_tools)

            if not predicted:
                precisions.append(0.0)
                continue

            precisions.append(len(expected & predicted) / len(predicted))

    if not precisions:
        tool_precision = 0.0
    else:
        tool_precision = sum(precisions) / len(precisions)

    return {
        "tool_precision": tool_precision,
    }



def compute_tool_recall(results_file, expected_tools):

    recalls = []

    with open(results_file, "r", encoding="utf-8") as file:

        for line in file:

            result = json.loads(line)

            task_expected_tools = expected_tools[result["task_id"]]
            task_predicted_tools = extract_predicted_tools(result)

            expected = set(task_expected_tools)
            predicted = set(task_predicted_tools)

            if not expected:
                recalls.append(0.0)
                continue

            recalls.append(len(expected & predicted) / len(expected))

    if not recalls:
        tool_recall = 0.0
    else:
        tool_recall = sum(recalls) / len(recalls)

    return {
        "tool_recall": tool_recall,
    }



def compute_tool_f1_score(results_file, expected_tools):

    f1_scores = []

    with open(results_file, "r", encoding="utf-8") as file:

        for line in file:

            result = json.loads(line)

            task_expected_tools = expected_tools[result["task_id"]]
            task_predicted_tools = extract_predicted_tools(result)

            expected = set(task_expected_tools)
            predicted = set(task_predicted_tools)

            if not predicted:
                precision = 0.0
            else:
                precision = len(expected & predicted) / len(predicted)

            if not expected:
                recall = 0.0
            else:
                recall = len(expected & predicted) / len(expected)

            if (precision + recall) == 0:
                f1_scores.append(0.0)
                continue

            f1_scores.append((2 * precision * recall) / (precision + recall))

    if not f1_scores:
        tool_f1_score = 0.0
    else:
        tool_f1_score = sum(f1_scores) / len(f1_scores)

    return {
        "tool_f1_score": tool_f1_score,
    }







# --------------------------------------------------
# Agent Metrics
# --------------------------------------------------
def compute_completion_rate(results_file):

    completed_tasks = 0
    total_tasks = 0

    with open(results_file, "r", encoding="utf-8") as file:

        for line in file:

            result = json.loads(line)

            if result["completed"]:
                completed_tasks += 1

            total_tasks += 1

    if total_tasks == 0:
        completion_rate = 0.0
    else:
        completion_rate = completed_tasks / total_tasks

    return {
        "completion_rate": completion_rate,
    }



def compute_max_steps_hit_rate(results_file):

    max_step_hits = 0
    total_tasks = 0

    with open(results_file, "r", encoding="utf-8") as file:

        for line in file:

            result = json.loads(line)

            if result["max_steps_hit"]:
                max_step_hits += 1

            total_tasks += 1

    if total_tasks == 0:
        max_steps_hit_rate = 0.0
    else:
        max_steps_hit_rate = max_step_hits / total_tasks

    return {
        "max_steps_hit_rate": max_steps_hit_rate,
    }



def compute_average_steps(results_file):

    steps = []

    with open(results_file, "r", encoding="utf-8") as file:

        for line in file:
            result = json.loads(line)
            steps.append(result["steps"])

    if not steps:
        average_steps = 0.0
        stddev_steps = 0.0
    else:
        average_steps = sum(steps) / len(steps)
        stddev_steps = stdev(steps) if len(steps) > 1 else 0.0

    return {
        "average_steps": average_steps,
        "stddev_steps": stddev_steps,
    }






# --------------------------------------------------
# Execution Metrics
# --------------------------------------------------
def compute_execution_time(results_file):

    execution_times = []

    with open(results_file, "r", encoding="utf-8") as file:

        for line in file:

            result = json.loads(line)
            execution_times.append(result["execution_time"])

    if not execution_times:
        average_execution_time = 0.0
        stddev_execution_time = 0.0
    else:
        average_execution_time = sum(execution_times) / len(execution_times)
        stddev_execution_time = stdev(execution_times) if len(execution_times) > 1 else 0.0

    return {
        "execution_time": average_execution_time,
        "stddev_execution_time": stddev_execution_time,
    }






# --------------------------------------------------
# Reduction Metrics
# --------------------------------------------------
def compute_average_visible_tools(results_file):

    visible_tools = []

    with open(results_file, "r", encoding="utf-8") as file:

        for line in file:

            result = json.loads(line)
            visible_tools.extend(result["visible_tools_counts"])

    if not visible_tools:
        average_visible_tools = 0.0
        stddev_visible_tools = 0.0
    else:
        average_visible_tools = sum(visible_tools) / len(visible_tools)
        stddev_visible_tools = stdev(visible_tools) if len(visible_tools) > 1 else 0.0

    return {
        "average_visible_tools": average_visible_tools,
        "stddev_visible_tools": stddev_visible_tools,
    }













# builds the name of the result files (one for experiment)
def build_results_file(
    results_dir,
    safe_model_name,
    experiment_name,
    top_k,
    n_clusters,
):

    filename_parts = [safe_model_name, experiment_name]

    if top_k is not None:
        filename_parts.append(f"k{top_k}")

    if n_clusters is not None:
        filename_parts.append(f"c{n_clusters}")

    filename = "_".join(filename_parts) + ".jsonl"

    return Path(results_dir) / filename








# Different reduction methods use different parameters
# This function simplifies parameter retrieval
def get_experiment_parameters(experiment, config):

    # Retrieve top_k
    if (
        "semantic" in experiment
        or "concept" in experiment
        or "schema" in experiment
    ):
        top_k_values = config["reduction_parameters"]["top_k"]

    else:
        top_k_values = [None]


    # Retrieve n_clusters
    if "category" in experiment:
        n_cluster_values = config["reduction_parameters"]["n_clusters"]

    else:
        n_cluster_values = [None]

    return (top_k_values, n_cluster_values)




# Returns the header for the results file
def build_metrics_header():

    return [

        # Experiment Metadata
        "dataset",
        "model",
        "experiment",
        "top_k",
        "n_clusters",

        # Prediction Metrics
        "tool_exact_match",
        "tool_precision",
        "tool_recall",
        "tool_f1_score",

        # Agent Metrics
        "completion_rate",
        "max_steps_hit_rate",
        "average_steps",
        "stddev_steps",

        # Execution Metrics
        "execution_time",
        "stddev_execution_time",

        # Reduction Metrics
        "average_visible_tools",
        "stddev_visible_tools",
    ]


# Builds a row of the results file
def build_metrics_row(
    dataset_name,
    model_name,
    experiment_name,
    top_k,
    n_clusters,
    metrics,
):

    return [

        # Experiment Metadata
        dataset_name,
        model_name,
        experiment_name,
        top_k,
        n_clusters,

        # Prediction Metrics
        metrics["tool_exact_match"],
        metrics["tool_precision"],
        metrics["tool_recall"],
        metrics["tool_f1_score"],

        # Agent Metrics
        metrics["completion_rate"],
        metrics["max_steps_hit_rate"],
        metrics["average_steps"],
        metrics["stddev_steps"],

        # Execution Metrics
        metrics["execution_time"],
        metrics["stddev_execution_time"],

        # Reduction Metrics
        metrics["average_visible_tools"],
        metrics["stddev_visible_tools"],
    ]






def evaluate(config):

    # Load tasks
    tasks = load_tasks(
        repository=config["dataset"]["repository"],
        dataset_name=config["dataset"]["tasks"],
        cache_dir=config["dataset"]["tasks_directory"],
    )


    # Load tools
    expected_tools = {}

    for task in tasks:
        expected_tools[task["id"]] = extract_expected_tools(task)

    # Generate experiment combinations
    experiments = generate_experiments(config["reductions"], config["combination_sizes"])

    # Count experiment evaluations
    total_evaluations = 0
    for experiment in experiments:
        top_k_values, n_cluster_values = get_experiment_parameters(experiment, config)
        total_evaluations += len(top_k_values) * len(n_cluster_values)
    total_evaluations *= len(config["agent"]["models"])

    # Prepare output file
    metrics_file = (Path(config["results"]["directory"]) / "metrics.csv")

    # Prepare progress bar
    progress = tqdm(total=total_evaluations, desc="Evaluating")

    with open(metrics_file, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        # Header
        writer.writerow(build_metrics_header())

        # Iterate over models
        for model_name in config["agent"]["models"]:

            safe_model_name = (
                model_name
                .replace(".", "_")
                .replace(":", "_")
                .replace("-", "_")
            )

            # Iterate over experiments
            for experiment in experiments:

                # Naming the experiment
                experiment_name = "_".join(experiment)

                # Parameter loops
                top_k_values, n_cluster_values = get_experiment_parameters(experiment, config)

                for top_k in top_k_values:

                    for n_clusters in n_cluster_values:

                        # Opens result file for this experiment
                        results_file = (
                            build_results_file(
                                results_dir=config["results"]["directory"],
                                safe_model_name=safe_model_name,
                                experiment_name=experiment_name,
                                top_k=top_k,
                                n_clusters=n_clusters,
                            )
                        )

                        if not results_file.exists():
                            print(f"Missing {results_file}")
                            continue



                        metrics = {}

                        # Prediction Metrics
                        metrics.update(compute_tool_exact_match(results_file, expected_tools))
                        metrics.update(compute_tool_precision(results_file, expected_tools))
                        metrics.update(compute_tool_recall(results_file, expected_tools))
                        metrics.update(compute_tool_f1_score(results_file, expected_tools))

                        # Agent Metrics
                        metrics.update(compute_completion_rate(results_file))
                        metrics.update(compute_max_steps_hit_rate(results_file))
                        metrics.update(compute_average_steps(results_file))

                        # Execution Metrics
                        metrics.update(compute_execution_time(results_file))
                        
                        # Reduction Metrics
                        metrics.update(compute_average_visible_tools(results_file))




                        # Records metrics for this experiment
                        writer.writerow(
                            build_metrics_row(
                                dataset_name=config["dataset"]["tasks"],
                                model_name=model_name,
                                experiment_name=experiment_name,
                                top_k=top_k,
                                n_clusters=n_clusters,
                                metrics=metrics,
                            )
                        )

                        # Update the progress bar
                        progress.update(1)

    # Close the progress bar
    progress.close()





