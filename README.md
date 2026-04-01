# Constrained Clarification: Training LLM Agents to Ask Better Questions Under Budget Constraints

This project trains a large language model (LLM) to ask optimal clarifying questions when given ambiguous coding problems — but under a strict budget on how many questions it can ask. It uses **Reinforcement Learning with PPO-Lagrangian** on top of **Qwen2.5-Coder-7B-Instruct** with LoRA adapters, evaluated on the **HumanEvalComm** benchmark.

---

## Current Status (2026-04-01)

**Phase: Training sanity check (in progress)**

Completed:
- Pipeline validated end-to-end (smoke tests with Llama 3B, 8B, and Qwen 7B)
- Switched from Llama-3.1-8B to Qwen2.5-Coder-7B (stronger coding baseline)
- Baseline eval shows Qwen scores ~40-80% on degraded specs without training (3 runs, 10 problems each)
- Early baseline numbers are deflated due to executor bugs that have since been fixed:
  - String outputs were unquoted in test assertions (correct code scored 0)
  - Function name mismatch (agent wrote `candidate`, tests called original name)
  - Helper functions (e.g., `poly`) missing from test programs
  - Template test relations (`$demo$`, `$input$`) not expanded
- Baseline needs re-running with fixed code to get accurate numbers
- Training sanity check (3 iterations, 16 episodes) is being debugged — prior runs timed out due to KV cache being disabled during generation (gradient checkpointing conflict). Fix applied, awaiting results.

Next steps:
1. Get training sanity check to complete successfully
2. Re-run baseline eval with all fixes
3. Full training: d1=0 and d1=1 (80 iterations each, ~11 hours per setting)
4. Evaluate and compare trained policies

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Hardware Requirements](#2-hardware-requirements)
3. [Installation](#3-installation)
4. [API Keys](#4-api-keys)
5. [Project Structure](#5-project-structure)
6. [Dataset](#6-dataset)
7. [System Prompts](#7-system-prompts)
8. [Training](#8-training)
9. [Evaluation](#9-evaluation)
10. [Trained Model Checkpoints](#10-trained-model-checkpoints)
11. [Configuration Reference](#11-configuration-reference)
12. [Key Design Decisions](#12-key-design-decisions)

---

## 1. What This Project Does

When an LLM is given a vague coding task like *"Return list with elements incremented by a number"*, it can either:
- **Guess** and write code (fast, but may be wrong)
- **Ask** a clarifying question like *"What number should each element be incremented by?"* (gets better information, but costs the user a turn)

This project trains the model to make that decision **optimally** — asking only when it is truly worth it — under a configurable budget `d₁` that caps the average number of questions per problem.

We train two policies:
- **d₁ = 0** — the agent learns to never ask; it must guess from the degraded spec alone
- **d₁ = 1** — the agent may ask at most 1 question on average per problem

These two policies form the beginning of a **Pareto frontier**: a family of policies ranging from "never ask" to "ask freely", each optimal for its budget.

The training algorithm is **PPO-Lagrangian**, where a Lagrange multiplier `λ₁` automatically learns how expensive each question should be to satisfy the budget. No manual penalty tuning is needed.

---

## 2. Hardware Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| GPUs | 2× A100-40GB | 2× A100-40GB |
| CPU RAM | 64 GB | 128 GB |
| Disk | 50 GB free | 100 GB free |
| Internet | Required (HuggingFace + OpenAI API) | — |

**GPU layout:**
- `cuda:0` — Policy training (Qwen2.5-Coder-7B + LoRA + value heads + optimizer, ~19 GB)
- `cuda:1` — Rollout inference + frozen reference model (~17 GB)

To change GPU assignment, edit `model.train_device` and `model.rollout_device` in `configs/default.yaml`.

---

## 3. Installation

```bash
# Clone the repository
git clone <repo-url>
cd rl_llm_multiturn_project

# Create and activate a virtual environment (Python 3.10+)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Install CUDA-enabled PyTorch separately if needed** (the version in `requirements.txt` is CPU-safe but may not match your CUDA driver):
```bash
# Example for CUDA 12.1
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```

**Verify GPU access:**
```bash
python3 -c "import torch; print(torch.cuda.device_count(), 'GPUs available')"
```

**Download the base model:**
The base model (Qwen2.5-Coder-7B-Instruct) is openly available on HuggingFace — no access request or token needed:
```bash
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-7B-Instruct'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-7B-Instruct', torch_dtype='bfloat16')"
```

---

## 4. API Keys

This project uses two external APIs:

### OpenAI API (required for training)

The **user simulator** (GPT-4o-mini) answers the agent's clarifying questions during every training episode. This is the only API cost during training (~$5–10 total for a full run).

```bash
export OPENAI_API_KEY="sk-..."
```

Or add it to a `.env` file in the project root (never commit this file):
```
OPENAI_API_KEY=sk-...
```

Then load it before running:
```bash
source .env  # if using a .env file
```

**Cost estimate:** GPT-4o-mini charges ~$0.15 per 1M input tokens. With 256 episodes/iteration × 80 iterations × ~2 questions/episode × ~200 tokens/call ≈ ~8M tokens ≈ **~$1.20 per d₁ setting**.

### HuggingFace Token (not required)

Qwen2.5-Coder-7B-Instruct is not gated — no HuggingFace token is needed to download it.

---

## 5. Project Structure

```
rl_llm_multiturn_project/
│
├── configs/
│   └── default.yaml          # All hyperparameters (model, training, constraints)
│
├── data/
│   ├── raw/                  # Downloaded raw data (auto-populated, gitignored)
│   └── processed/            # Preprocessed problem files (gitignored)
│
├── src/
│   ├── data/
│   │   ├── dataset.py        # Problem dataclass + HumanEvalComm loader
│   │   └── augmentation.py   # MBPP degradation via GPT-4o (one-time script)
│   │
│   ├── environment/
│   │   ├── env.py            # ClarificationEnv: the RL state machine
│   │   ├── user_simulator.py # GPT-4o-mini wrapper (async, with atomic question counting)
│   │   └── code_executor.py  # Sandboxed Python runner → pass@1 score
│   │
│   ├── models/
│   │   ├── agent.py          # Qwen2.5-Coder-7B + LoRA: generate() and score()
│   │   └── value_heads.py    # Three MLP value heads (reward, q_cost, t_cost)
│   │
│   ├── training/
│   │   ├── rollout.py        # Async episode collection + RolloutBuffer
│   │   ├── ppo.py            # PPO loss, GAE, KL penalty, entropy bonus
│   │   ├── lagrangian.py     # Lagrange multiplier dual update
│   │   └── trainer.py        # PPOLagrangianTrainer: main training loop
│   │
│   └── evaluation/
│       └── evaluator.py      # pass@1 metrics, Pareto frontier, plots
│
├── scripts/
│   ├── train.py              # Training entry point
│   ├── evaluate.py           # Evaluation entry point
│   ├── smoke_test.py         # End-to-end pipeline validation
│   ├── baseline_eval.py      # Base model coding ability test
│   ├── smoke_test.sbatch     # SLURM job script for smoke test
│   ├── baseline_eval.sbatch  # SLURM job script for baseline eval
│   └── train_sanity.sbatch   # SLURM job script for short training check
│
├── checkpoints/              # Saved model checkpoints (gitignored)
│   ├── d1_0/
│   │   ├── iter_0019/        # Checkpoint every 20 iterations
│   │   ├── iter_0039/
│   │   └── final/            # Final checkpoint after all iterations
│   └── d1_1/
│       ├── iter_0019/
│       └── final/
│
├── outputs/                  # Eval results, plots (gitignored)
│
├── ACTION_PLAN.md            # Full project design document
├── DATASET_REFERENCE.md      # HumanEvalComm dataset analysis and notes
└── requirements.txt
```

---

## 6. Dataset

The dataset is **HumanEvalComm** — 164 Python coding problems from HumanEval, each with multiple degraded versions of the problem specification.

**It is automatically downloaded from HuggingFace** the first time you run training or evaluation. No manual download is needed.

**What the degradations look like:**

| Type | Field | What changes |
|---|---|---|
| Ambiguity | `prompt1a` | Specific values replaced with vague terms ("by 1" → "by a number") |
| Inconsistency | `prompt1c` | Examples contradict the description |
| Incompleteness | `prompt1p` | All examples and details stripped; only a stub remains |
| Ambiguity + Inconsistency | `prompt2ac` | Both combined |
| Ambiguity + Incompleteness | `prompt2ap` | Both combined |
| Inconsistency + Incompleteness | `prompt2cp` | Both combined |
| All three | `prompt3acp` | Ambiguity + Inconsistency + Incompleteness |

**Train/test split:** The split is **stratified** at the base problem level — problems are grouped by their rarest available variant, then each group is split proportionally (~60% eval, ~40% train). This guarantees all 7 degradation types appear in both train and eval sets. All variants of a base problem go to the same set (no leakage). Enforced in `src/data/dataset.py`.

**Which variants are used for training** is controlled by `data.use_variants` in the config:
```yaml
data:
  use_variants:
    - prompt1a
    - prompt1c
    - prompt1p
    - prompt2ac
    - prompt2ap
    - prompt2cp
    - prompt3acp
```

---

## 7. System Prompts

There are two system prompts in this project. Both are defined in source code, not config files.

### Agent System Prompt

Located in `src/environment/env.py` — this is what the agent sees at every turn. Prompts are formatted using the model's chat template (`<|im_start|>/<|im_end|>` for Qwen):

```
You are a coding assistant. Given a coding task below, you must either:
  - Ask a clarifying question by responding with [ASK] followed by your question.
  - Write your Python solution by responding with [ANSWER] followed by the code.

Important rules:
  - Respond with ONLY one action per turn ([ASK] or [ANSWER]).
  - When you have enough information, write the code.
  - Do not explain your reasoning — just output the action directly.
```

The full conversation history is appended below this prompt at each turn.

### User Simulator System Prompt

Located in `src/environment/user_simulator.py` — this is what GPT-4o-mini receives:

**Mode: `count` (default)** — the simulator counts atomic questions and returns `QUESTION_COUNT: N`:
```
You are a helpful assistant who holds the complete specification for a coding problem.

Full specification:
{original_prompt}

Note: The agent may refer to the function by a different name (e.g., "candidate").
Treat any function name the agent uses as referring to the function described above.

Rules:
1. Answer ONLY the specific question(s) the agent asks. Do not volunteer extra information.
2. Do not reveal test cases or the full solution.
3. Keep your answer brief and factual.
4. After your answer, on a new line write EXACTLY:
   QUESTION_COUNT: N
   where N is the number of distinct atomic questions you identified in the agent's message.

Counting rules for N:
- Each distinct piece of information being requested = 1.
- "What is X and what is Y?" = 2.
- Conjunctions like "and", "also", "additionally" between separate requests = separate questions.
```

**Mode: `truncate`** — the simulator only sees the first question, `cost_q` is always 1:
```
You are a helpful assistant who holds the complete specification for a coding problem.
Answer ONLY the single question below. Do not volunteer extra information.
```

Switch between modes via `environment.multi_question_mode` in `configs/default.yaml`.

**Why two modes?** The agent may learn to pack multiple questions into one `[ASK]` turn. In `count` mode, each atomic question is charged separately (2 questions in one turn = `cost_q = 2`). In `truncate` mode, only the first question is answered and `cost_q = 1` always. See `ACTION_PLAN.md` for the full design discussion.

---

## 8. Training

### Quick Start

```bash
# Train d₁=0 policy (agent learns to never ask)
python scripts/train.py --d1 0

# Train d₁=1 policy (agent may ask ≤1 question on average)
python scripts/train.py --d1 1
```

Train both sequentially:
```bash
python scripts/train.py --d1 0 && python scripts/train.py --d1 1
```

### Resume from a Checkpoint

```bash
python scripts/train.py --d1 1 --resume checkpoints/d1_1/iter_0039
```

### Override Config Values at the Command Line

```bash
# Reduce iterations for a quick smoke-test
python scripts/train.py --d1 1 training.n_iterations=5 training.rollout_batch_size=16

# Change the multi-question mode
python scripts/train.py --d1 1 environment.multi_question_mode=truncate

# Increase the question budget
python scripts/train.py --d1 2 constraint.d1=2
```

### What Happens During Training

Each **iteration** consists of:

1. **Rollout collection** (~5–6 min): 256 episodes run in parallel. The agent sees a degraded problem spec, generates `[ASK]` or `[ANSWER]` actions, and the environment responds. API calls to GPT-4o-mini happen asynchronously. Code execution happens in a subprocess sandbox.

2. **Advantage computation**: GAE advantages are computed for three return streams: reward (pass@1), question cost, and turn cost.

3. **PPO update** (~3 min): 4 epochs over the buffer in mini-batches of 16. The Lagrangian advantage `A_lag = A_reward - λ₁·A_q - λ₂·A_t` is used. Only LoRA weights are updated.

4. **Lagrange update**: `λ₁ += 0.01 × (avg_questions - d₁)`. If the agent asked too many questions, `λ₁` rises — making questions more expensive next iteration.

5. **Logging**: Per-iteration stats printed to stdout.

**Expected training output:**
```
iter=   0 | reward=0.3210 | q=1.84 (budget=1) | λ₁=0.0084 | ppo=0.0421 | vf=0.3120 | kl=0.0023 | t=521s
iter=   1 | reward=0.3450 | q=1.62 (budget=1) | λ₁=0.0146 | ppo=0.0380 | vf=0.2890 | kl=0.0019 | t=498s
...
iter=  79 | reward=0.6820 | q=0.98 (budget=1) | λ₁=0.2310 | ppo=0.0120 | vf=0.0430 | kl=0.0008 | t=487s
```

**Total training time estimate:**
- ~8–9 min/iteration × 80 iterations = ~11 hours per d₁ setting
- d₁=0 and d₁=1 = ~22 hours total (run sequentially)

### Training Logs

Per-iteration logs are saved alongside each checkpoint as `log.json`:
```
checkpoints/d1_1/iter_0079/log.json
```

---

## 9. Evaluation

### Evaluate a Single Checkpoint

```bash
python scripts/evaluate.py --checkpoint checkpoints/d1_1/final --d1 1
```

Results are saved to `outputs/eval/results_d1_1.json`.

### Evaluate Both Policies and Plot the Pareto Frontier

After both d₁=0 and d₁=1 are trained:

```bash
python scripts/evaluate.py --sweep --output_dir outputs/pareto
```

This will:
1. Find `checkpoints/d1_0/final` and `checkpoints/d1_1/final`
2. Run each on the 100 held-out eval problems (greedy decoding, temperature=0)
3. Print a results table
4. Save `outputs/pareto/pareto_frontier.png`

**Example output table:**
```
  d1    pass@1    avg_q  avg_turns       n
----  --------  -------  ---------  ------
   0    0.4120     0.00       1.00     400
   1    0.5830     0.97       1.92     400
```

---

## 10. Trained Model Checkpoints

### Location

All checkpoints are saved under `checkpoints/` (gitignored):

```
checkpoints/
├── d1_0/
│   ├── iter_0019/          # Saved every 20 iterations
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors   ← LoRA weights (~80 MB)
│   │   ├── tokenizer.json
│   │   ├── value_heads.pt              ← Three MLP heads
│   │   ├── dual_variables.pt           ← λ₁, λ₂ values
│   │   ├── train_state.pt              ← Optimizer + scheduler state
│   │   └── log.json                    ← Training log up to this point
│   ├── iter_0039/
│   └── final/              ← Best checkpoint to use for evaluation
└── d1_1/
    ├── iter_0019/
    └── final/
```

### What is Saved

| File | Contents | Size |
|---|---|---|
| `adapter_model.safetensors` | LoRA adapter weights (the only trained LLM parameters) | ~80 MB |
| `adapter_config.json` | LoRA configuration (rank, alpha, target modules) | <1 KB |
| `tokenizer.json` | Tokenizer files (copied from base model) | ~10 MB |
| `value_heads.pt` | Three MLP value heads (reward, q_cost, t_cost) | ~50 MB |
| `dual_variables.pt` | Current λ₁ and λ₂ values | <1 KB |
| `train_state.pt` | Optimizer and LR scheduler state (for resuming) | ~500 MB |
| `log.json` | Per-iteration training metrics | ~100 KB |

**Note:** The base Qwen2.5-Coder-7B weights (~14 GB) are NOT saved — they are downloaded from HuggingFace and remain frozen. Only the LoRA adapter (~80 MB) is saved. To deploy a checkpoint, you need both the base model and the LoRA adapter.

### Loading a Checkpoint for Inference

```python
from omegaconf import OmegaConf
from src.models.agent import Agent

cfg = OmegaConf.load("configs/default.yaml")
agent = Agent(cfg)
agent.load_lora("checkpoints/d1_1/final")

# Generate a response
action_text, _, _, _ = agent.generate(prompt)
print(action_text)  # "[ASK] What number should each element be incremented by?"
```

---

## 11. Configuration Reference

All configuration lives in `configs/default.yaml`. Every value can be overridden at the command line using OmegaConf dot-notation (e.g., `training.n_iterations=40`).

### Model

```yaml
model:
  name: Qwen/Qwen2.5-Coder-7B-Instruct      # HuggingFace model ID
  dtype: bfloat16                            # bf16 for A100s
  lora_rank: 16                              # LoRA rank (~40M trainable params)
  lora_alpha: 32                             # LoRA scaling factor
  lora_dropout: 0.05
  lora_target_modules:                       # Which linear layers to apply LoRA to
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  gradient_checkpointing: true               # Reduces activation memory from ~10GB to ~3GB
  train_device: cuda:0                       # GPU for PPO update
  rollout_device: cuda:1                     # GPU for rollout inference
```

### Environment

```yaml
environment:
  max_turns: 6                               # Hard cap on conversation length
  max_new_tokens: 512                        # Max tokens per agent action
  max_seq_len: 1536                          # Max total prompt length (tokens)
  rollout_temperature: 0.8                   # Exploration temperature during rollout
  multi_question_mode: count                 # "count" or "truncate" (see Section 7)
  efficiency_alpha: 0.025                    # Small bonus for fewer turns (tiebreaker vs waste)
  efficiency_beta: 0.025                     # Small bonus for fewer questions
```

### User Simulator

```yaml
user_simulator:
  model: gpt-4o-mini                         # OpenAI model for the user simulator
  temperature: 0.0                           # Deterministic responses
  max_tokens: 300
  max_concurrent_api: 50                     # Max parallel API calls (rate limit safety)
```

### Training

```yaml
training:
  rollout_batch_size: 256                    # Episodes per iteration
  ppo_epochs: 4                              # PPO update passes per batch
  ppo_mini_batch_size: 16                    # Mini-batch size per GPU
  clip_epsilon: 0.2                          # PPO clip range
  gamma: 1.0                                 # Discount factor (1.0 = no discounting)
  gae_lambda: 0.95                           # GAE smoothing
  kl_coeff: 0.05                             # KL penalty (keeps policy near reference)
  entropy_coeff: 0.01                        # Entropy bonus (prevents collapse)
  lr_policy: 5.0e-6                          # LoRA learning rate
  lr_value: 1.0e-4                           # Value head learning rate
  optimizer: adamw_8bit                      # 8-bit AdamW (saves ~8 GB vs full)
  warmup_steps: 20
  n_iterations: 80                           # Training iterations per d₁ setting
  save_interval: 20                          # Save checkpoint every N iterations
  eval_interval: 10                          # Run eval every N iterations
```

### Constraint (CMDP)

```yaml
constraint:
  d1: 1                                      # Question budget (set to 0 or 1 for this run)
  lambda_init: 0.0                           # Starting value for λ₁
  lambda_max: 10.0                           # Maximum value for λ₁
  lr_lambda: 0.01                            # Lagrange multiplier step size
  d2: 4                                      # Turn budget (soft, secondary constraint)
  lambda2_init: 0.0
  lambda2_max: 5.0
  lr_lambda2: 0.005
```

### Data

```yaml
data:
  hf_dataset: jie-jw-wu/HumanEvalComm       # HuggingFace dataset identifier
  eval_size: 100                             # Held-out base problems for evaluation
  seed: 42
  use_variants:                              # Degradation types used for training
    - prompt1a
    - prompt1c
    - prompt1p
    - prompt2ac
    - prompt2ap
    - prompt2cp
    - prompt3acp
```

---

## 12. Key Design Decisions

**Why Qwen2.5-Coder-7B and not Llama-3.1-8B?**
The HumanEvalComm paper shows that code-specialized models (CodeQwen, DeepSeek Coder) significantly outperform general-purpose models on degraded specs. Qwen2.5-Coder-7B scores ~70% on standard HumanEval vs ~55% for Llama-3.1-8B. It's also similar size (~14GB bf16), same LoRA config, and not gated on HuggingFace. We verified that Llama-3.1-8B scored 0% on smoke test episodes; Qwen Coder provides a much stronger coding baseline for PPO to build on.

**Why LoRA and not full fine-tuning?**
Full fine-tuning on 7B with AdamW would require ~56 GB for optimizer states alone. LoRA limits trainable parameters to ~40M, reducing optimizer memory to ~800 MB. The frozen base weights also prevent catastrophic forgetting of Python syntax knowledge.

**Why constrained prefix decoding?**
The base model sometimes outputs code without the required `[ASK]` or `[ANSWER]` prefix, resulting in malformed actions and zero reward. Constrained prefix decoding forces every generation to start with one of the two valid prefixes. The model still chooses which prefix by comparing their log-probs given the prompt — so the decision is learned, not random. This eliminates wasted training iterations on formatting errors.

**Why PPO-Lagrangian and not fixed-penalty RL?**
A fixed penalty requires hand-tuning — you don't know in advance how large the penalty needs to be to achieve exactly d₁=1 question on average. PPO-Lagrangian finds this value automatically via dual ascent: if the agent asks too many questions, λ₁ rises until it stops; if it asks too few, λ₁ falls. This also enables sweeping multiple d₁ values without re-tuning.

**Why two Lagrange multipliers?**
`λ₁` enforces the question budget `d₁`. `λ₂` is a soft secondary constraint on turns. Turn cost exists because asking many focused questions across many turns is still expensive, even if question count is low.

**Multi-question handling (the `multi_question_mode` setting):**
When the agent writes `[ASK] What is X? And what is Y?`, it is asking two questions in one turn. In `count` mode (default), the user simulator counts 2 atomic questions and the environment charges `cost_q = 2`. This prevents the agent from exploiting the budget by batching questions. In `truncate` mode, only the first question is answered and `cost_q = 1` always. The default `count` mode is more principled but depends on GPT-4o-mini counting accurately; `truncate` mode is simpler but restricts the agent's action space. Switch with `environment.multi_question_mode=truncate`.

**Function name handling in code execution:**
Degraded specs sometimes rename functions to `candidate`, but test cases use the original name. The code executor aliases the **last** top-level function in the agent's code to the expected `entry_point`. This avoids breaking helper functions (e.g., `is_prime` defined before `is_multiply_prime`). Additionally, helper functions from the degraded spec (e.g., `poly` for `find_zero`) are automatically extracted and prepended to the test program so tests can reference them.

**String output quoting in test assertions:**
Some test case outputs are bare strings (e.g., `fdcb` not `'fdcb'`). The executor detects these and wraps them in `repr()` so assertions compare against the correct type. Template-based test relations (using `$demo$` and `$input$` placeholders) are also expanded correctly.

**Function name handling in the user simulator:**
The simulator's system prompt tells GPT-4o-mini to treat any function name the agent uses (e.g., `candidate`) as referring to the function in the original spec. Without this, the simulator would say "I don't know about `candidate`" when the original spec defines `decode_cyclic`.

**Why partial credit for code evaluation?**
Binary pass/fail gives a flat reward landscape. Partial credit (fraction of assertions passing) gives smoother gradients — an agent that gets 8/10 tests right receives reward 0.8, not 0. This significantly stabilises PPO training.



**Why a stratified train/eval split?**
Rare degradation types (`prompt3acp` = 12 problems, `prompt2cp` = 35 problems) could end up entirely in one set with a random split. The stratified split groups problems by their rarest variant and splits each group proportionally, guaranteeing all 7 types appear in both sets.

**Why `ast.literal_eval` and not `json.loads` for test cases?**
HumanEvalComm stores test cases as Python-literal strings (single-quoted dicts), not JSON. `json.loads` will fail on them. Always use `ast.literal_eval`.
