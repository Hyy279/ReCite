# ReCite: Agentic Reasoning for Faithful Citation

## Overview

ReCite consists of three main components:

1. **Agent (Master Brain)** - Orchestrates citation search and retrieval using multi-threaded processing with S2 API integration
2. **CiteLocator** - SFT-trained marking engine for identifying and localizing citations within documents
3. **QueryPlanner** - GRPO-optimized intent engine for planning and refining search queries

## Architecture

- **Agent**: Central coordinator with thread-safe rate limiting and caching mechanisms
- **CiteLocator**: Supervised fine-tuning on citation detection tasks
- **QueryPlanner**: Reinforcement learning optimization for query strategy selection

## Data Structure

The `Data/` directory contains evaluation datasets and training materials:
- `evaluation/` - Benchmark datasets for system evaluation
- `location_perception/` - Citation localization training data
- `query_planning/` - Query strategy training data
- `reflective_trajectories/` - Training trajectories for model optimization

## Installation

```bash
pip install -r requirements.txt
```

## Starting the Services

Start the three model servers in separate terminals. Replace `/your/path/to/` with the actual model paths:

**Terminal 1: CiteLocator (Port 8001)**
```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /your/path/to/CiteLocator \
  --served-model-name CiteLocator \
  --port 8001 \
  --gpu-memory-utilization 0.4 \
  --max-model-len 16384
```

**Terminal 2: QueryPlanner (Port 8002)**
```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /your/path/to/QueryPlanner \
  --served-model-name QueryPlanner \
  --port 8002 \
  --gpu-memory-utilization 0.4 \
  --max-model-len 16384
```

**Terminal 3: MasterBrain Agent (Port 8003)**
```bash
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model /your/path/to/MasterBrain \
  --served-model-name MasterBrain \
  --port 8003 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 60000
```

## Quick Start

Once all services are running, configure API keys and initialize the agent:

```python
from Agent.agent import CiteAgent

agent = CiteAgent(
    s2_api_keys=["key1", "key2"],
    device="cuda"
)
```

## File Structure

**Agent (Master Brain)**
- `agent.py` - Main orchestrator coordinating citation retrieval across three model components
- `evaluation_api.py` - Evaluation using multiple LLM APIs (OpenAI, DeepSeek, Qwen, etc.)
- `evaluation_local.py` - Local model inference and evaluation
- `lenient_evaluation_and_position.py` - Alternative evaluation metrics
- `train_sft.sh` - Training script for Agent model with distributed training setup
- `tools/model_workers.py` - Interface for communicating with remote model servers

**CiteLocator**
- `train_sft.py` - Fine-tune model for citation position detection
- `evaluation.py` - Benchmark citation localization performance

**QueryPlanner**
- `train_sft.py` - Foundation model fine-tuning
- `train_grpo.py` - GRPO training for query keyword generation

**Data**
- `evaluation/` - Test datasets for benchmarking
- `location_perception/` - Citation detection training data
- `query_planning/` - Query planning training data
- `reflective_trajectories/` - GRPO training trajectories

## Requirements

See `requirements.txt` for dependencies.

## License

Please refer to the project documentation for licensing information.
