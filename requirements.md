# QAEvolver
Copyright © 2026 440 Labs LLC

QAEvolver is an AI system built in DSPY for answering questions on multihop QA benchmarks. QAEvolver's purpose is to serve as a testing ground for the effectiveness of CodeEvolver.

## CodeEvolver
CodeEvolver offers autonomous coding agents for high reliability AI systems. It uses GEPA optimization to evolve your AI system code until it performs optimally for a given dataset and outcome metric.

This combines several mechanisms:
- **Optimizer algorithm:** GEPA is a reflective language model algorithm that makes point mutations to the code base, over many iterations, and the best solution is selected, based on a dataset and a reward metric.
- **Coding agents**: Autonomous agents execute code changes that are requested by the optimizer. 
- **Git branching:** A git process manages evolving code across many git worktrees  
- **Sandboxing for security:** Coding agents are a big cyber risk without sandboxing, network policies, etc. 

### Optimizer
The optimizer is handled by a separate repository, which will later be loaded into this repository, as defined by specs/gepa_plan.md. This repository will create a /optimize endpoint to run GEPA optimization orchestration, and Will interface with that package as defined in gepa_plan.md. 

### Coding Agents
CodeEvolver agents uses Claude Agents SDK in a fully autonomous, dangerously-skip-permissions mode, which uses a Modal sandbox execution environment for modifying code, running code, and executing bash / grep / glob. After code changes are made, the app needs to run a mutated version of the code, and return the output. 

Code changes will be made in the context of GEPA optimization - i.e., an evolutionary, 100+ step process. Speed and parallel execution of coding changes is important. The AI worfklow code needs to be edited over 100 times. Each mutation is small, but costs will add up. Do not worry about cost right now.

CodeEvolver repository: https://github.com/julianghadially/CodeEvolver
CodeEvolver requirements: github repo with module path, metric path, and dataset. No main function required. 

### Git branching
Users (like QAEvolver) connect their code with the CodeEvolver GitHub app, which adds our organization as a contributor to their GitHub.

## Initial QAEvolver Architecture
QAEvolver in its initial architecture will mirror the GEPA paper architecture (reconstructed to the best of our ability). CodeEvolver will take this starting point and optimize it. These are the GEPA HotpotQA architecture:
- **Inital query module:** Generate a query to search
- **Additional query module:** Generate additional queries to search (Reference summary of prior evidence)
- **Retriever module:** Execute search queries
- **Evidence summarizer module:** Summarize evidence based on the question and page result(s).
- **Answer module:**

### Prompts:
- Initial retriever prompt: Given the field question, produce the field query to find evidence for the question (Not confirmed)
- Additional retriever prompt: Given the fields question, summary 1, produce the fields query (published)

### Constraints:
- Do NOT search more than two times per question. This is a hard requirement.
- Do NOT visit more than one page per query
- Do NOT use the HotpotQA dataset as context. 

### Considerations
- The data set is designed to contain information from Wikipedia. However, the optimizer is not required to stay on Wikipedia only.
- The optimizer is allowed to create or remove modules, dynamic prompts, tool calls, etc.
- The optimizer is allowed to change the module types (e.g., dspy.ReAct for tool calling, dspy.ChainOfThought, dspy.Predict, etc.)
- Available services: Firecrawl and serper.dev. 
- There is no limit on the number of search results to display per query

### QAAgent dspy.ReAct module
QAAgent should be armed (initially) with web search and web scraping tools, which leverage firecrawl and serper.dev services.

## Data Sources
- **HotpotQA:** A multihop QA dataset with 113k examples designed to piece together related facts from multiple Wikipedia pages. The dataset provides supporting facts as gold label answers along with gold label supporting context, but supporting facts and gold label answers should not be provided to the system. The system is supposed to find those supporting facts via its retrieval pipeline.
- **NovelHopQA:** 1–4 hop reasoning over long narrative contexts (up to ~128 k tokens)
- **DocHop-**QA: multimodal scientific document QA dataset
- **PluriHop:** exhaustive and recall-sensitive multi-hop QA
- **BMGQ:** Bottom-up Multi-step Graph QA. A dataset creating complex multi-step reasoning paths from semi-structured data
- **DeepAmbigQA:** Contains ~3,600 difficult multi-hop questions with name ambiguity explicitly embedded — testing completeness of answer sets and robustness. Evaluates not just hop reasoning but disambiguation under multi-stage inference
- **MMQA:** multi-table reasoning where evidence must be drawn from interconnected tabular datasets (foreign/primary key reasoning) beyond text

## Experiment
We will be replicating the GEPA experiment on HotpotQA with CodeEvolver, which allows for prompt and architecture optimization.

We use a training and validation and testing set from HotpotQA (available on hugging face https://huggingface.co/datasets/hotpotqa/hotpot_qa). We will use a data pipeline that saves a random sampler to obtain an excerpt of the data set:
- **Training set:** **800** (sampled from official training set)
- **Validation set:** **200** (sampled from official validation set)
- **Test set:** **1000** (sampled from official test set)
