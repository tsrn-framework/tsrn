# TSRN: Tool-Space Reduction and Navigation for Tool-Using LLM Agents

TSRN (Tool-Space Reduction and Navigation) is a framework for studying tool reduction in tool-using Large Language Model (LLM) agents. As tool ecosystems continue to grow, agents must reason over increasingly large tool catalogs, which can negatively affect tool selection and task completion. TSRN investigates whether reducing the set of candidate tools presented to an agent can improve performance. The framework implements and evaluates multiple reduction strategies, including semantic, schema-based, concept-based, and category-based reductions, across different model families, reduction configurations, and tool-catalog sizes.

---

# Installation

## 1. Create a Virtual Environment

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

**Linux / macOS**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

---

## 2. Install Dependencies

Install all project dependencies:

```bash
pip install -r requirements.txt
```

---

## 3. Install Ollama

TSRN uses Ollama to serve local language models through an OpenAI-compatible API.

Install Ollama by following the instructions for your operating system:

https://ollama.com

After installation, verify that Ollama is running:

```bash
ollama serve
```

The default configuration expects Ollama to be accessible at:

```text
http://localhost:11434/v1
```

This endpoint can be modified in `config.yaml` if needed.

---

## 4. Download Models

Before running experiments, download the models required by the selected configuration.

For example:

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
```

Additional models used in the paper can be downloaded in the same manner:

```bash
ollama pull qwen2.5:0.5b-instruct-q4_K_M
ollama pull qwen2.5:1.5b-instruct-q4_K_M
ollama pull qwen2.5:3b-instruct-q4_K_M
ollama pull qwen2.5:7b-instruct-q4_K_M

ollama pull qwen3.5:0.8b
ollama pull qwen3.5:2b
ollama pull qwen3.5:4b

ollama pull gemma4:e2b
ollama pull gemma4:e4b
ollama pull gemma4:12b

ollama pull gpt-oss:20b
ollama pull devstral:24b
```

Only the models enabled in `config.yaml` need to be installed.

---

## 5. Download and Prepare the Dataset

The experiments rely on datasets obtained from Microsoft TaskBench.

Run:

```bash
python bootstrap.py
```

This script downloads and prepares the required datasets under:

```text
datasets/
├── tasks/
└── tools/
```

---

# Configuration

All experimental settings are controlled through:

```text
config.yaml
```

The configuration file allows users to specify:

- Task and tool datasets
- Models to evaluate
- Reduction strategies
- Reduction parameters
- Reduction combinations
- Agent settings
- Output locations

Example options include:

```yaml
agent:
  models:
    - qwen2.5:7b-instruct-q4_K_M

combination_sizes: [0,1]

reductions:
  semantic: true
  category: true
  concept: true
  schema: true
```

The default configuration is intentionally lightweight and intended for quick artifact validation.

To reproduce the experiments reported in the paper, uncomment the corresponding models and parameters described in the configuration file.

---

# Running Experiments

Execute experiments using:

```bash
python main.py
```

The framework will:

1. Load the configured datasets.
2. Apply the selected reduction strategies.
3. Execute the configured agent(s).
4. Collect evaluation metrics.
5. Store results on disk.

---

# Results

Experimental results are stored in:

```text
results/
```

The repository includes the complete set of results used in the paper. This allows reviewers and users to inspect the raw experimental outputs without rerunning the full evaluation.

To reproduce the experiments from scratch, delete the contents of the `results/` directory and execute:

```bash
python main.py
```

using the desired configuration specified in `config.yaml`.

New execution outputs will be automatically written to the `results/` directory.

---

# Repository Structure

```text
.
├── bootstrap.py
├── main.py
├── requirements.txt
│
├── config/
│   └── config.yaml
│
├── datasets/
│   ├── tasks/
│   └── tools/
│
├── results/
│
└── src/
```

<!--

# Citation

If you use this framework, please cite the accompanying paper.

```bibtex
% @inproceedings{tsrn,
%   title     = {TSRN: Tool-Space Reduction and Navigation for Tool-Using LLM Agents},
%   author    = {},
%   booktitle = {Proceedings of the ACM International Conference on the Foundations of Software Engineering (FSE)},
%   year      = {2027},
%   publisher = {ACM}
% }
```
-->
