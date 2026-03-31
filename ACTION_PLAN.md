# Action Plan: Constrained Clarification RL Pipeline

## Overview

We are building a reinforcement learning pipeline that trains an LLM agent (**Qwen2.5-Coder-7B-Instruct** with LoRA) to ask optimal clarifying questions when given ambiguous coding problems, under a budget constraint on how many questions it can ask. The environment is a multi-turn conversation loop: the agent sees a degraded problem spec, decides to either ask a clarifying question (`[ASK]`) or submit code (`[ANSWER]`), a GPT-4o-mini user simulator answers questions using the hidden full spec, and a sandboxed code executor scores the submitted code against a hidden test suite (pass@1). This interaction is formulated as a Constrained MDP with two cost signals — questions asked and turns used — and trained using PPO-Lagrangian, where a Lagrange multiplier automatically learns how much to penalize question-asking to satisfy the budget constraint `d₁`. By training separate policies at `d₁ ∈ {0,1}`, we compare a "never ask" policy against one that may ask at most 1 question on average.

**Model change (2026-03-29):** Switched from Llama-3.1-8B-Instruct to Qwen2.5-Coder-7B-Instruct based on HumanEvalComm paper results showing code-specialized models score ~20% higher on degraded specs. Llama 8B scored 0% on initial smoke tests; Qwen Coder provides a stronger coding baseline. Same LoRA config and memory budget apply.

## Hardware & Model

We have 2× A100-40GB GPUs (80GB total). **Qwen2.5-Coder-7B-Instruct** — its bf16 weights occupy ~14GB, leaving ~26GB headroom per GPU for activations, KV cache, and optimizer states. GPU 0 handles PPO training (model weights + LoRA gradients + 8-bit AdamW optimizer + value heads + gradient checkpointing to cap activation memory at ~3GB). GPU 1 holds a frozen inference copy of the model for rollout collection and the reference policy for KL penalty. The agent uses LoRA rank=16 on all attention and MLP projection layers (~40M trainable params); three small MLP value heads predict expected future reward, question cost, and turn cost from the LLM's 3584-dim hidden state. Prompts use Qwen's chat template (`<|im_start|>/<|im_end|>`) via `tokenizer.apply_chat_template()`. Constrained prefix decoding forces every generation to start with `[ASK]` or `[ANSWER]`, preventing malformed outputs.

## Pipeline Components

| Component | Role | Trained? |
|---|---|---|
| Problem Bank | JSON of (degraded_spec, original_spec, test_suite) tuples from HumanEvalComm + MBPP | No |
| User Simulator | GPT-4o-mini API call; answers agent questions using original_spec only | No |
| Code Executor | Runs agent code in a subprocess sandbox against test assertions; returns pass@1 | No |
| Agent | Qwen2.5-Coder-7B + LoRA; generates `[ASK]`/`[ANSWER]` actions (constrained prefix) | **Yes** |
| Value Heads | 3 MLPs predicting V_reward, V_q_cost, V_t_cost from hidden state | **Yes** |

## Training Loop (one iteration)

1. **Rollout**: Run 256 episodes in parallel. Each episode: reset with a problem → agent generates action → environment steps (simulator or executor) → store (state, action, log_prob, reward, cost_q, cost_t) in buffer.
2. **Advantage computation**: Use GAE with `γ=1.0, λ=0.95` on all three return streams. Combine into `A_lagrangian = A_reward - λ₁·A_q_cost - λ₂·A_t_cost`.
3. **PPO update**: 4 epochs over the buffer in mini-batches of 16. Clip ratio at 0.2. Add KL penalty (`coeff=0.05`) and entropy bonus (`coeff=0.01`). Backprop only through LoRA weights.
4. **Lagrange update**: `λ₁ += lr_λ × (avg_questions_this_batch - d₁)`. If agent asked too many questions, `λ₁` rises, making questions more expensive next iteration — and vice versa.
5. Repeat for 80 iterations per `d₁` setting.

## Key Hyperparameters

```yaml
# Hardware: 2× A100-40GB
model:           Qwen/Qwen2.5-Coder-7B-Instruct     # ~14GB bf16; fits on 40GB with room
lora_rank:       16
lora_alpha:      32
lora_targets:    [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
gradient_checkpointing: true    # keeps activation memory ~3GB instead of ~10GB

rollout_batch_size:    256      # reduced from 512 (40GB vs 80GB)
max_turns:             6
max_new_tokens:        512      # enough for longest solutions (~500 tokens)
max_seq_len:           1536     # reduced from 2048; episodes rarely exceed this
temperature:           0.8      # agent during rollout
user_sim_temperature:  0.0      # GPT-4o-mini (deterministic environment)

ppo_epochs:            4
ppo_mini_batch_size:   16       # reduced from 32 (fits comfortably in 40GB)
clip_epsilon:          0.2
gamma:                 1.0
gae_lambda:            0.95
kl_coeff:              0.05
entropy_coeff:         0.01
lr_policy:             5e-6
lr_value:              1e-4
lr_lambda:             0.01
optimizer:             adamw_8bit  # saves ~8GB vs full-precision AdamW

d1_values:             [0, 1]
n_iterations:          80
```

**Memory budget per GPU (approximate):**
```
GPU 0 (training):
  Model weights (bf16):         16.0 GB
  Activations (grad checkpoint): 3.0 GB
  LoRA gradients:                0.2 GB
  8-bit AdamW states (LoRA):     0.4 GB
  Value heads + optimizer:       0.1 GB
  Misc / fragmentation:          1.5 GB
  ─────────────────────────────────────
  Total:                        ~21 GB   (19 GB headroom on 40 GB)

GPU 1 (rollout + reference):
  Model weights (bf16):         16.0 GB
  KV cache (32 concurrent eps):  2.0 GB
  Misc:                          1.0 GB
  ─────────────────────────────────────
  Total:                        ~19 GB   (21 GB headroom on 40 GB)
```

## Phase Schedule

- **Weeks 1-2**: Build and test all pipeline components (env, simulator, executor, data loader). Run sanity checks with the base model (no training).
- **Weeks 3-4**: Run baselines — Never-Ask, Always-Ask-k, SFT on GPT-4o generated gold trajectories, BED prompting.
- **Weeks 5-7**: PPO-Lagrangian Phase 1. Train `d₁=2` first to debug and tune, then full Pareto sweep. ~11 hours per setting × 6 = ~66 GPU-hours.
- **Week 8**: Analyze emergent allocation — does the constrained policy naturally ask more on ambiguous problems?
- **Weeks 9-10**: Phase 3 — assign per-problem budgets based on ambiguity scores, train one adaptive policy, compare against best fixed-budget policy.
- **Weeks 11-12**: Plots, paper.

## Expected File Structure

```
rl_llm_multiturn_project/
├── configs/default.yaml
├── data/
│   ├── raw/                  # downloaded HumanEvalComm + MBPP
│   └── processed/            # parsed Problem objects
├── src/
│   ├── data/
│   │   ├── dataset.py        # Problem dataclass + loader
│   │   └── augmentation.py   # MBPP degradation via GPT-4o
│   ├── environment/
│   │   ├── env.py            # ClarificationEnv (state machine)
│   │   ├── user_simulator.py # GPT-4o-mini async wrapper
│   │   └── code_executor.py  # subprocess sandbox + pass@1
│   ├── models/
│   │   ├── agent.py          # Qwen2.5-Coder-7B + LoRA + generation/scoring
│   │   └── value_heads.py    # 3 MLP value heads
│   ├── training/
│   │   ├── rollout.py        # RolloutBuffer + collect_rollouts()
│   │   ├── ppo.py            # PPO loss + GAE
│   │   ├── lagrangian.py     # Lagrange multiplier dual update
│   │   └── trainer.py        # Main PPOLagrangianTrainer loop
│   └── evaluation/
│       └── evaluator.py      # pass@1, Pareto frontier, allocation curves
├── scripts/
│   ├── train.py              # entry point
│   └── evaluate.py
└── requirements.txt
```

---

## End-to-End Episode Example

**Problem sampled from training pool:**
```
task_id:        HumanEval/42
entry_point:    incr_list
degradation:    1a (ambiguity — specific value removed)
degraded_spec:  "Return list with elements incremented by a number."
original_spec:  "Return list with elements incremented by 1."   ← agent never sees this
test_suite:     assert incr_list([1,2,3]) == [2,3,4]
                assert incr_list([]) == []
                assert incr_list([-1,0,1]) == [0,1,2]
budget d₁:      2  (agent may average at most 2 questions per problem)
```

---

**Turn 1 — Agent input:**
```
System: You are a coding assistant. Given the task below, you can either:
- Ask a clarifying question by responding with [ASK] followed by your question
- Write your solution by responding with [ANSWER] followed by Python code

Task:
def incr_list(l: list):
    """Return list with elements incremented by a number."""

Conversation so far: (none)

Your action:
```

**Turn 1 — Agent output:**
```
[ASK] What number should each element be incremented by?
```

**Turn 1 — User simulator receives:**
```
System: You hold the full specification: "Return list with elements incremented by 1."
        Answer ONLY what is asked. Do not volunteer extra information.
User:   What number should each element be incremented by?
```

**Turn 1 — User simulator responds:**
```
Each element should be incremented by 1.
```

**Turn 1 — Transition recorded:**
```
reward  = 0.0   (no code submitted yet)
cost_q  = 1.0   (asked a question)
cost_t  = 1.0   (used a turn)
done    = False
```

---

**Turn 2 — Agent input:**
```
System: (same system prompt)

Task:
def incr_list(l: list):
    """Return list with elements incremented by a number."""

Conversation so far:
[Turn 1] You asked: "What number should each element be incremented by?"
[Turn 1] User said: "Each element should be incremented by 1."

Your action:
```

**Turn 2 — Agent output:**
```
[ANSWER]
def incr_list(l: list):
    return [x + 1 for x in l]
```

**Turn 2 — Code executor runs:**
```python
def incr_list(l: list):
    return [x + 1 for x in l]

assert incr_list([1, 2, 3]) == [2, 3, 4]   # PASS
assert incr_list([]) == []                  # PASS
assert incr_list([-1, 0, 1]) == [0, 1, 2]  # PASS
```

**Turn 2 — Transition recorded:**
```
reward  = 1.0   (3/3 tests passed → pass@1 = 1.0)
cost_q  = 0.0   (no question asked)
cost_t  = 1.0   (used a turn)
done    = True
```


---

**Episode Summary:**

| Turn | Action | reward | cost_q | cost_t |
|------|--------|--------|--------|--------|
| 1    | [ASK]  | 0.0    | 1.0    | 1.0    |
| 2    | [ANSWER] | 1.029 | 0.0    | 1.0    |
| **Total** | | **1.029** | **1.0** | **2.0** |

---

**Reward & Constraint Formulation:**

```
Episode return (reward):     R = 1.0         (pass@1)
Episode cost (questions):    C_q = 1.0       (1 question asked)
Episode cost (turns):        C_t = 2.0       (2 turns used)

Budget constraint:           E[C_q] ≤ d₁ = 2.0   (soft, enforced on batch average)

Lagrangian advantage at Turn 1 ([ASK]):
  A_lagrangian = A_reward - λ₁ · A_q_cost - λ₂ · A_t_cost
               = (+0.6)   - (0.3) · (+0.8) - (0.05) · (+0.5)
               = 0.6 - 0.24 - 0.025
               = +0.335   → this action gets reinforced (positive advantage)

Lagrangian advantage at Turn 2 ([ANSWER]):
  A_lagrangian = (+0.4) - (0.3) · (-0.8) - (0.05) · (-0.5)
               = 0.4 + 0.24 + 0.025
               = +0.665   → also reinforced (submitting correct code is good)

After this batch, if avg_questions_across_256_episodes = 2.4 > d₁ = 2.0:
  λ₁ ← λ₁ + lr_λ × (2.4 - 2.0) = λ₁ + 0.01 × 0.4 = λ₁ + 0.004
  → questions become slightly more expensive next iteration
```

The agent in this episode asked exactly the right question and got full reward. Under a tighter budget (e.g. `d₁=0`), `λ₁` would be large enough that the cost of asking outweighs the expected reward gain, so the policy would learn to guess directly instead.


## Dumpgroud / Caveats

 we are concerend about the scenario where the policy llm asks multiple questions at once but we only increase the constraint by 
 one. What do you think about this.                                                                                               

⏺ This is a real and important loophole. Here's the full breakdown:

  The Problem

  The budget constraint tracks turns with [ASK], not questions asked within a turn. So if the agent learns to pack multiple
  questions into one [ASK] action:

  [ASK] What is the increment value? Should I handle empty lists?
        Also, what type should the output be?

  It pays cost_q = 1 but extracts 3 answers from the simulator. Under a tight budget like d₁=1, this completely defeats the
  constraint — the agent can ask arbitrarily many questions for free by batching them. Since LLMs naturally produce compound
  questions, this will almost certainly emerge during training.

  Options

  Option A — User simulator only answers the first question
  The simplest fix. Instruct GPT-4o-mini to answer only the first question if multiple are detected, ignore the rest. No cost
  function changes needed. The agent gets no benefit from batching, so the behavior never gets reinforced.

  System prompt addition:
  "If the agent asks more than one question, answer ONLY the first question.
   Do not acknowledge or answer subsequent questions."

  Downside: the agent might still waste a turn asking questions that don't get answered.

  Option B — Count ? marks and charge proportionally
  Parse the [ASK] text, count question marks, charge cost_q = n_questions. Principled but gameable — the agent can rephrase
  multi-part questions without ? marks ("Tell me the increment value and whether to handle negatives."), which is semantically
  still a compound question.

  Option C — Hard format enforcement
  Add to the system prompt: "Ask exactly ONE question per [ASK] turn." Then post-process: if the output contains more than one ?,
  truncate at the first one before passing to the simulator. This means the agent's extra questions are silently dropped — it
  never benefits from trying.

  Option D — Charge cost proportionally via a question-count classifier
  Use a lightweight heuristic (sentence splitting + ? detection) to count distinct questions in the [ASK] text and set cost_q = n.
   Most principled for the CMDP, but complex to implement reliably.

  ---
  Recommendation

  Combine Option A + Option C. These two together close the loophole cleanly:

  1. System prompt explicitly says "ask exactly ONE question per turn"
  2. Post-processing: truncate [ASK] output at the first ? + sentence end before passing to the simulator
  3. User simulator instructed to answer only the first question as a backstop

  This way the agent is never rewarded for batching — it physically cannot extract multiple answers — and the cost function stays
  clean (cost_q ∈ {0,1} per turn, no edge cases). The CMDP formulation remains mathematically clean.

  There's also a deeper question worth noting: is asking one precise multi-part question better or worse than asking two separate
  targeted ones? Under A+C, the agent is forced to ask sequentially, which is actually the behavior you want to measure — it must
  decide which single piece of information is most valuable to ask next. That's exactly the skill the budget constraint is
  designed to train.

  Worth updating the ACTION_PLAN.md and/or DATASET_REFERENCE.md with this decision so the implementation is consistent. Want me to
   add it?


What if we tell the user llm to also keep track of atomic questions in one response from the policy llm. and increase the       
constraint accordingly?

 How It Would Work

  The user simulator does two things per turn instead of one:

  Agent sends:  [ASK] What is the increment value? Should I handle empty lists?

  Simulator:
    1. Counts atomic questions → 2
    2. Answers both
    3. Returns response + QUESTION_COUNT: 2

  Environment sets:  cost_q = 2  (not 1)

  The CMDP constraint E[C_q] ≤ d₁ still holds mathematically — cost_q per turn is now variable (1, 2, 3...) instead of binary. The Lagrangian update is
  unchanged, it just sees larger cost values when the agent batches questions.

  ---
  Why This Is Better Than Truncation

  - The agent's action space is unrestricted — no silent dropping of questions
  - The constraint becomes semantically meaningful: it genuinely tracks atomic information requests, not just turn counts
  - The agent can make a real tradeoff: asking 2 questions in one turn costs cost_q=2 but only cost_t=1. Asking them separately costs cost_q=2, cost_t=2.
   So batching saves turns but not the question budget — which is exactly the right incentive structure
  - No need for format enforcement or post-processing hacks

  ---
  The Real Risks

  1. What is "atomic" is ambiguous.
  This is the biggest problem. Consider:

  "What is the increment value and should it handle negative numbers?"  → 2? or 1?
  "Can you describe the full expected behavior including edge cases?"    → 1? but extracts everything
  "What type should the output be?"                                     → 1, clearly

  GPT-4o-mini will make judgment calls here. At temperature=0 it'll be consistent for the same phrasing, but the agent could learn to phrase compound
  questions in ways that get counted as 1 ("describe the complete behavior" vs "what is X and what is Y").

  2. Cost signal noise across problems.
  Even at temperature=0, the same structure phrased differently might get counted differently. A noisy cost signal directly destabilizes the Lagrangian
  update — λ₁ is chasing a moving target.

  3. The agent can still game it, just differently. Instead of asking "What is X? What is Y?", it learns to ask "Tell me everything about X including Y"
  — one atomic question that extracts the same information. You've closed one loophole and opened a softer one.

  ---
  Mitigation: Make the Counting Prompt Explicit

  You can tighten this significantly by defining "atomic" precisely in the simulator prompt:

  After your answer, on a new line write exactly:
  QUESTION_COUNT: N

  Rules for counting N:
  - Each distinct piece of information being requested = 1 question
  - "What is X and Y?" = 2
  - "What is X?" = 1
  - "Can you describe X?" where X is a single concept = 1
  - Conjunctions (and, also, additionally) between requests = separate questions

  At temperature=0, this gives you deterministic counting for most cases. Borderline cases will still exist but will be rare.

  ---
  Verdict

  It's a sound idea worth implementing, with one important caveat: the counting must be treated as part of the environment spec (like a rule, not a
  judgment call), defined clearly enough that GPT-4o-mini applies it consistently. The residual gaming risk (rephrasing compound questions as single
  ones) is acceptable — it's actually semantically closer to what a well-formed clarifying question should look like anyway.

  The alternative (truncation + single-question enforcement) is simpler and noiseless, but it constrains the agent's action space in a way that's
  slightly artificial. Your proposed approach lets the agent naturally discover that batching questions is expensive, which is a more interesting
  emergent behavior to study.


## Future Direction: Dynamic Budget via Learned Ambiguity Assessment

Currently `d1` is a static global constant — the agent has the same average question budget regardless of how degraded the spec is. A natural extension is to let the agent allocate its budget dynamically based on how ambiguous each spec is. Three approaches, from simplest to most ambitious:

### Approach A: Per-variant fixed budgets (no model changes)

Hand-assign budgets per degradation type:
```python
budget_map = {"1a": 0.5, "1c": 0.5, "1p": 2, "2ac": 1, "2ap": 2, "2cp": 2, "3acp": 3}
```
The dual update computes violation relative to the per-variant budget. Easy to implement but the agent doesn't learn to assess ambiguity — it's told the difficulty.

### Approach B: Learned budget head (explicit predictor)

Add a small MLP head on the same hidden state the value heads use:
```
budget_head: hidden_state → predicted_d1 (scalar)
```
Train it with a self-supervised signal: for solved episodes (pass@1 > 0.5), the actual number of questions asked is a reasonable target. The per-episode constraint becomes `questions_asked ≤ budget_head(state)`. Adds a chicken-and-egg problem early in training since the budget head is random.

### Approach C: Implicit dynamic budget (what we already have)

**Key insight:** with a global `d1=1` across all variants, the agent is already learning a dynamic budget implicitly. The policy reads the degraded spec through Qwen's full language understanding and outputs `P([ASK])` vs `P([ANSWER])`. That probability IS the model's learned ambiguity estimate. The Lagrangian doesn't constrain individual episodes — it constrains the expectation. So the agent is free to ask 0 questions on easy specs (`1a`, `1c`) and 2 on hard ones (`1p`), as long as the average stays ≤ 1.

The difference from a "true" dynamic budget is that the agent is constrained by what else is in the batch. If you sampled all `3acp` problems, the agent can't ask 3 each — it's still pulled toward the global average.

### Analysis for the paper

To tell this story without building anything new, analyze the trained policy's behavior:
- Plot `P([ASK])` vs degradation type — this is an empirical ambiguity detector
- Show that the agent asks more on `1p`/`3acp` and skips on `1a`/`1c` — emergent budget allocation
- Compare pass@1 breakdown by variant between constrained (`d1=1`) and unconstrained (no Lagrangian) agents
- Run an ablation: at inference time, remove the constraint and let the agent ask freely on hard variants — does pass@1 on `3acp` improve? That gap measures the cost of a tight global budget

If the emergent allocation result is strong, Approach C is the most compelling story: the agent learns when to ask without being told problem difficulty. If it's weak (agent just never asks or always asks), then Approach A or B would be the next step.

