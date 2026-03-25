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
