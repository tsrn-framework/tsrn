import json
import time
from pathlib import Path

from tqdm import tqdm
import sys

from src.dataset import load_tasks
from src.dataset import load_tools
from src.agent import build_client
from src.agent import call_llm

from src.experiments import generate_experiments

from src.semantic import reduce_semantic
from src.category import reduce_category
from src.concept import reduce_concept
from src.schema import reduce_schema

from openai import InternalServerError


def build_openai_tools(tool_descriptions):

    tools = []

    for tool in tool_descriptions["nodes"]:

        properties = {}
        required = []

        for parameter in tool.get("parameters", []):

            properties[parameter["name"]] = {
                "type": parameter["type"],
                "description": parameter["desc"],
            }

            required.append(parameter["name"])

        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool["id"],
                    "description": tool["desc"],
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
        )

    return tools


def execute(config):

    # Load tasks and tool descriptions
    tasks = load_tasks(
        repository=config["dataset"]["repository"],
        dataset_name=config["dataset"]["tasks"],
        cache_dir=config["dataset"]["tasks_directory"],
    )

    tool_descriptions = load_tools(
        config["dataset"]["tools_directory"],
        config["dataset"]["tools"],
    )

    tools = build_openai_tools(tool_descriptions)


    # Limit the number of tasks if requested
    max_tasks = config["dataset"]["max_tasks"]

    if max_tasks is not None:
        tasks = tasks.select(range(min(max_tasks, len(tasks))))


    # Prepare experiments (combinations)
    experiments = generate_experiments(config["reductions"], config["combination_sizes"])


    # Prepare results directory
    results_dir = Path(config["results"]["directory"])
    results_dir.mkdir(exist_ok=True)


    # --------------------------------------------------
    # Experimental loops
    # --------------------------------------------------

    # Iterate over models
    for model_name in config["agent"]["models"]:

        # Create an agent for each task


        # Iterate over experimental configurations
        for experiment in experiments:


            # Naming the experiment
            experiment_name = "_".join(experiment)


            # Semantic and Concept use top-k
            if (
                "semantic" in experiment
                or "concept" in experiment
                or "schema" in experiment
            ):
                top_k_values = config["reduction_parameters"]["top_k"]
            else:
                top_k_values = [None]

            # Category uses n_clusters
            if "category" in experiment:
                n_cluster_values = config["reduction_parameters"]["n_clusters"]
            else:
                n_cluster_values = [None]


            # Iterate over k values
            for top_k in top_k_values:

                # Iterate over n_cluster values
                for n_clusters in n_cluster_values:


                    # Build result filename
                    safe_model_name = (model_name
                        .replace(".", "_")
                        .replace(":", "_")
                        .replace("-", "_")
                    )

                    filename_parts = [
                        safe_model_name,
                        experiment_name,
                    ]

                    if top_k is not None:
                        filename_parts.append(f"k{top_k}")
                    if n_clusters is not None:
                        filename_parts.append(f"c{n_clusters}")

                    results_filename = "_".join(filename_parts) + ".jsonl"


                    results_file = (results_dir / results_filename)

                    # deletes file if it exists
                    if results_file.exists(): results_file.unlink()



                    # ------------------------------------------
                    # Execute all tasks
                    # ------------------------------------------

                    interactive = sys.stdout.isatty()
                    for task in tqdm(tasks,
                        ncols=None if interactive else 80,
                        desc=results_filename,
                    ):




                        # Create the client
                        # One agent for each task
                        client = build_client(
                            base_url=config["agent"]["base_url"],
                            api_key=config["agent"]["api_key"],
                        )






                        # Record system message and user request
                        messages = [
                            {
                                "role": "system",
                                "content":
                                    "You are a ReAct agent. "
                                    "You select only one tool "
                                    "at a time."
                            },
                            {
                                "role": "user",
                                "content": task["instruction"],
                            },
                        ]

                        # prediction metrics
                        executed_tools = []

                        # agent metrics
                        completed = False
                        max_steps_hit = True

                        # execution metrics
                        start_time = time.perf_counter()

                        # reduction metrics
                        visible_tools_counts = []

                        # --------------------------------------
                        # Agentic loop
                        # --------------------------------------
                        for step in range(config["agent"]["max_steps"]):

                            tool_sets = []

                            if "semantic" in experiment:

                                tool_sets.append(
                                    reduce_semantic(
                                        instruction=task["instruction"],
                                        tools=tools,
                                        top_k=top_k,
                                    )
                                )

                            if "category" in experiment:

                                tool_sets.append(
                                    reduce_category(
                                        instruction=task["instruction"],
                                        tools=tools,
                                        n_clusters=n_clusters
                                    )
                                )

                            if "concept" in experiment:

                                tool_sets.append(
                                    reduce_concept(
                                        instruction=task["instruction"],
                                        tools=tools,
                                        top_k=top_k,
                                    )
                                )

                            if "schema" in experiment:

                                tool_sets.append(
                                    reduce_schema(
                                        instruction=task["instruction"],
                                        tools=tools,
                                        top_k=top_k,
                                    )
                                )


                            # Intersection of active reductions
                            available_tools = tools

                            if tool_sets:

                                available_tools = [
                                    tool
                                    for tool in tools
                                    if all(
                                        tool in tool_set
                                        for tool_set in tool_sets
                                    )
                                ]


                            # reduction metric
                            visible_tools_counts.append(len(available_tools))



                            # Ask the model for the next action
                            try:
                                response = call_llm(
                                    client=client,
                                    model=model_name,
                                    messages=messages,
                                    tools=available_tools,
                                    n_ctx=config["agent"]["n_ctx"],
                                    max_tokens=config["agent"]["max_tokens"],
                                    temperature=config["agent"]["temperature"],
                                )
                            except InternalServerError as e:
                                print(f"InternalServerError on task {task['id']}: {e}")
                                max_steps_hit = False
                                break

                            # Stop condition
                            finish_reason = response.choices[0].finish_reason

                            if finish_reason == "stop":
                                completed = True
                                max_steps_hit = False
                                break

                            if finish_reason != "tool_calls":
                                max_steps_hit = False
                                break


                            # Store assistant response
                            assistant_message = response.choices[0].message
                            # ReAct: one tool call per iteration
                            tool_call = assistant_message.tool_calls[0]
                            

                            messages.append({
                                "role": "assistant",
                                "content": assistant_message.content,
                                "tool_calls": [
                                    {
                                        "id": tool_call.id,
                                        "type": tool_call.type,
                                        "function": {
                                            "name": tool_call.function.name,
                                            "arguments": tool_call.function.arguments,
                                        },
                                    }
                                ],
                            })

                            

                            # prediction metrics
                            executed_tools.append(
                                {
                                    "tool": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                }
                            )

                            # Simulate tool execution
                            observation = {
                                "tool": tool_call.function.name,
                                "status": "success",
                            }

                            # Store tool output
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": json.dumps(observation),
                                }
                            )


                        # some execution information for evaluation
                        # agent metrics
                        steps = len(executed_tools)
                        if (steps >= config["agent"]["max_steps"]):
                            max_steps_hit = True

                        # execution metrics
                        execution_time = time.perf_counter() - start_time

                        # --------------------------------------
                        # Save task result
                        # --------------------------------------

                        result = {
                            "task_id": task["id"],
                            "instruction": task["instruction"],

                            # Prediction Metrics
                            "predicted_tools": executed_tools,

                            # Agent Metrics
                            "completed": completed,
                            "max_steps_hit": max_steps_hit,
                            "steps": len(executed_tools),

                            # Execution Metrics
                            "execution_time": execution_time,

                            # Reduction Metrics
                            "visible_tools_counts": visible_tools_counts,
                        }

                        with open(results_file, "a", encoding="utf-8") as file:
                            file.write(json.dumps(result, ensure_ascii=False))
                            file.write("\n")

