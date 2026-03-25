# Action Plan: Constrained Clarification RL Pipeline

## Overview

We are building a reinforcement learning pipeline that trains an LLM agent (Llama-3.1-8B-Instruct with LoRA) to ask optimal clarifying questions when given ambiguous coding problems, under a budget constraint on how many questions it can ask. The environment is a multi-turn conversation loop: the agent sees a degraded problem spec, decides to either ask a clarifying question (`[ASK]`) or submit code (`[ANSWER]`), a GPT-4o-mini user simulator answers questions using the hidden full spec, and a sandboxed code executor scores the submitted code against a hidden test suite (pass@1). This interaction is formulated as a Constrained MDP with two cost signals — questions asked and turns used — and trained using PPO-Lagrangian, where a Lagrange multiplier automatically learns how much to penalize question-asking to satisfy the budget constraint `d₁`. By training six separate policies at `d₁ ∈ {0,1,2,3,4,5}`, we produce a Pareto frontier of policies ranging from "never ask" to "ask freely", each optimal for its budget — something SFT cannot provide.

## Hardware & Model

We have 2× A100-80GB GPUs. GPU 1 handles rollout collection (running 512 episodes in parallel with async GPT-4o-mini API calls), while GPU 0 handles the PPO training update. The agent is Llama-3.1-8B-Instruct with LoRA (rank=16) applied to all attention and MLP projection layers — only ~40M parameters are trained, the rest of the 8B model is frozen. Three small MLP value heads sit on top of the LLM's hidden state and predict expected future reward, question cost, and turn cost respectively; these are used to compute GAE advantages for PPO. A frozen copy of the initial model serves as the reference policy for a KL penalty that prevents the agent from drifting too far from sensible language.

## Pipeline Components

| Component | Role | Trained? |
|---|---|---|
| Problem Bank | JSON of (degraded_spec, original_spec, test_suite) tuples from HumanEvalComm + MBPP | No |
| User Simulator | GPT-4o-mini API call; answers agent questions using original_spec only | No |
| Code Executor | Runs agent code in a subprocess sandbox against test assertions; returns pass@1 | No |
| Agent | Llama-3.1-8B + LoRA; generates `[ASK]`/`[ANSWER]` actions | **Yes** |
| Value Heads | 3 MLPs predicting V_reward, V_q_cost, V_t_cost from hidden state | **Yes** |

## Training Loop (one iteration)

1. **Rollout**: Run 512 episodes in parallel. Each episode: reset with a problem → agent generates action → environment steps (simulator or executor) → store (state, action, log_prob, reward, cost_q, cost_t) in buffer.
2. **Advantage computation**: Use GAE with `γ=1.0, λ=0.95` on all three return streams. Combine into `A_lagrangian = A_reward - λ₁·A_q_cost - λ₂·A_t_cost`.
3. **PPO update**: 4 epochs over the buffer in mini-batches of 32. Clip ratio at 0.2. Add KL penalty (`coeff=0.05`) and entropy bonus (`coeff=0.01`). Backprop only through LoRA weights.
4. **Lagrange update**: `λ₁ += lr_λ × (avg_questions_this_batch - d₁)`. If agent asked too many questions, `λ₁` rises, making questions more expensive next iteration — and vice versa.
5. Repeat for 80 iterations per `d₁` setting.

## Key Hyperparameters

```yaml
model:           meta-llama/Llama-3.1-8B-Instruct
lora_rank:       16
lora_alpha:      32
lora_targets:    [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

rollout_batch_size:    512
max_turns:             6
max_new_tokens:        300
temperature:           0.8        # agent during rollout
user_sim_temperature:  0.0        # GPT-4o-mini (deterministic environment)

ppo_epochs:            4
ppo_mini_batch_size:   32
clip_epsilon:          0.2
gamma:                 1.0
gae_lambda:            0.95
kl_coeff:              0.05
entropy_coeff:         0.01
lr_policy:             5e-6
lr_value:              1e-4
lr_lambda:             0.01

d1_values:             [0, 1, 2, 3, 4, 5]
n_iterations:          80
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
│   │   ├── agent.py          # Llama-3.1-8B + LoRA + generation/scoring
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
| 2    | [ANSWER] | 1.0  | 0.0    | 1.0    |
| **Total** | | **1.0** | **1.0** | **2.0** |

---

**Reward & Constraint Formulation:**

```
Episode return (reward):     R = 1.0         (pass@1 from final code)
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

After this batch, if avg_questions_across_512_episodes = 2.4 > d₁ = 2.0:
  λ₁ ← λ₁ + lr_λ × (2.4 - 2.0) = λ₁ + 0.01 × 0.4 = λ₁ + 0.004
  → questions become slightly more expensive next iteration
```

The agent in this episode asked exactly the right question and got full reward. Under a tighter budget (e.g. `d₁=0`), `λ₁` would be large enough that the cost of asking outweighs the expected reward gain, so the policy would learn to guess directly instead.
