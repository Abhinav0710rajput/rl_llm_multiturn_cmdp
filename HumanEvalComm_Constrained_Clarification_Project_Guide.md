# Constrained Clarification: Training LLM Agents to Ask Better Questions Under Budget Constraints

## Project Overview for Team Members

## 1. The Problem

When you give an LLM a vague coding request like "Return list with elements incremented by a number," current models just guess. They might assume the number is 1, or 2, or add a parameter — but they rarely ask "What number should I increment by?" before writing code.

Humans do this naturally. If your manager gives you a vague spec, you ask questions before coding. But questions have a cost — your manager is busy, users are impatient. So you need to decide: is this ambiguity worth asking about, or should I just make a reasonable guess?

Our project trains an LLM to make this decision optimally using reinforcement learning with budget constraints.

## 2. The Core Idea

We formulate clarification question-asking as a **Constrained Markov Decision Process (CMDP)**. The agent (an LLM) interacts with a user to solve ambiguous coding problems. It can either ask a clarifying question or submit code.

- **Reward:** How correct is the final code? (measured by running unit tests)
- **Constraint:** The average number of questions asked per problem must stay within a budget **d₁**

The budget creates a real tension: asking questions helps write better code, but you can't ask unlimited questions. The agent must learn which questions are worth asking and when to stop asking and just write code.

We train using **PPO-Lagrangian**, where a Lagrange multiplier automatically learns how much to penalize each question given the current budget. By training separate policies at different budgets (d₁ = 0, 1, 2, 3, 4, 5), we produce a **Pareto frontier** — a menu of policies ranging from "never ask" to "ask freely," each optimal for its budget level. This is a deployment-time knob that SFT cannot provide.

## 3. What the Dataset Looks Like

We use **HumanEvalComm**, built on top of OpenAI's HumanEval benchmark. It has 164 base Python coding problems. Each problem has:

**The original (full) specification S\* — what the user simulator knows:**

```python
def incr_list(l: list):
    """Return list with elements incremented by 1.
    >>> incr_list([1, 2, 3])
    [2, 3, 4]
    >>> incr_list([5, 3, 5, 2, 3, 9, 0, 123])
    [6, 4, 6, 3, 4, 4, 10, 1, 124]
    """
```

**The test suite T\* — how we score the code (never shown to agent):**

```python
assert incr_list([1, 2, 3]) == [2, 3, 4]
assert incr_list([]) == []
assert incr_list([-1, 0, 1]) == [0, 1, 2]
```

**The degraded specification S — what the agent sees:**

```python
def incr_list(l: list):
    """Return list with elements incremented by a number."""
```

The degradation is done in multiple ways for each problem:

- **Ambiguity (1a):** "by 1" → "by a number"
- **Incompleteness (1p):** Examples removed entirely
- **Inconsistency (1c):** Examples contradict the description (e.g., examples show increment by 2 but correct answer is increment by 1)
- **Combinations (2ac, 2ap, 2cp):** Two types combined

This gives ~762 total variants, but only 164 truly independent problems (variants share the same test suite). To increase diversity, we augment with **MBPP** (Mostly Basic Python Problems) — 974 additional independent problems that we programmatically degrade using GPT-4o.

## 4. The RL Environment — Complete Walkthrough

### 4.1 Three Components (Only the Agent is Trained)

**Component 1 — Problem Bank (JSON file on disk):** The HumanEvalComm + MBPP­Augmented problems. Each entry has the degraded spec, original spec, and test suite.

**Component 2 — User Simulator (GPT-4o-mini API call, NOT trained):** An LLM that holds the original spec S\* and answers the agent's questions. It only reveals what is specifically asked — it doesn't volunteer extra info.

**Component 3 — Code Executor (Python sandbox, NOT trained):** Runs the agent's generated code against the test suite. Returns pass@1 (fraction of tests that pass).

**The Agent — Llama-3.2-3B-Instruct with LoRA adapters (THIS is what we train):** Takes in the degraded spec + conversation history, outputs either a clarifying question or code.

### 4.2 What One Episode Looks Like

Let's trace through problem #42 with the degraded spec "Return list with elements incremented by a number."

**EPISODE A: Agent asks a good question**

**Turn 1 — Agent sees this prompt:**

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

**Turn 1 — Agent generates:** `[ASK] What number should each element be incremented by?`

**Turn 1 — User simulator is prompted:**

```
System: You hold the full specification for this problem:
"Return list with elements incremented by 1."
Answer ONLY what is asked. Do not volunteer extra info.

Agent asks: "What number should each element be incremented by?"

Your response:
```

**Turn 1 — Simulator responds:** "Each element should be incremented by 1."

**Turn 1 — We record:** reward = 0, cost c₁ = 1 (asked a question), cost c₂ = 1 (one turn)

**Turn 2 — Agent now sees the conversation history appended to the prompt and generates:**

```
[ANSWER]
def incr_list(l: list):
    return [x + 1 for x in l]
```

**Turn 2 — Code executor runs tests:** All 3 tests pass.

**Turn 2 — We record:** reward = 1.0 (pass@1), cost c₁ = 0, cost c₂ = 1

**Episode A totals:** reward = 1.0, questions asked = 1, turns = 2

---

**EPISODE B: Agent guesses without asking**

**Turn 1 — Agent generates:**

```
[ANSWER]
def incr_list(l: list):
    return [x + 2 for x in l]
```

**Turn 1 — Code executor:** 2 of 3 tests fail.

**Episode B totals:** reward = 0.33, questions asked = 0, turns = 1

---

**EPISODE C: Agent asks a wasteful question**

**Turn 1 — Agent:** `[ASK] Can you give me an example input?` **Simulator:** "For example, [1, 2, 3]." (This doesn't reveal the increment value!) **Record:** reward = 0, c₁ = 1, c₂ = 1

**Turn 2 — Agent:** `[ASK] What should the output be for [1, 2, 3]?` **Simulator:** "[2, 3, 4]." (Now the agent can infer increment = 1, but it took 2 questions) **Record:** reward = 0, c₁ = 1, c₂ = 1

**Turn 3 — Agent:** `[ANSWER] return [x + 1 for x in l]` — all tests pass. **Record:** reward = 1.0, c₁ = 0, c₂ = 1

**Episode C totals:** reward = 1.0, questions asked = 2, turns = 3

### 4.3 How PPO-Lagrangian Learns From These Episodes

After collecting 256 such episodes, here's what happens:

**Step 1 — Compute advantages.**

For each action in each episode, we compute:

```
A_lagrangian = A_reward - λ₁ × A_cost1 - λ₂ × A_cost2
```

In Episode A, the ASK at Turn 1 has positive reward advantage (it led to correct code) but also positive cost advantage (it costs 1 question). If λ₁ is small, the reward gain dominates → action is reinforced. If λ₁ is large, the cost dominates → action is discouraged.

In Episode C, the first ASK ("give me an example") has zero reward advantage — it didn't help produce the answer. But it costs 1 unit. So its Lagrangian advantage is negative. The policy learns this was a bad question.

**Step 2 — PPO weight update.**

For each (state, action) pair, we compute:

```
ratio = (current policy probability of this action) / (old probability when we collected it)
loss = -min(ratio × advantage, clip(ratio, 0.8, 1.2) × advantage)
loss.backward() → gradients flow into LoRA weights only
optimizer.step() → LoRA weights nudged
```

The clipping prevents the policy from changing too much in one step. Actions with positive advantage become more likely. Actions with negative advantage become less likely.

**Step 3 — Update Lagrange multiplier.**

```python
avg_questions = mean([total_questions for episode in batch])  # e.g., 2.4
λ₁ = max(0, λ₁ + lr_lambda × (avg_questions - d₁))          # e.g., d₁ = 2
# If avg > d₁: λ₁ increases → questions become more expensive next iteration
# If avg < d₁: λ₁ decreases → questions become cheaper next iteration
```

This is the automatic penalty tuning. No hand-tuning needed — λ₁ finds the right value on its own.

**Step 4 — Repeat.**

Go back to Step 1 with the updated policy and updated λ₁. Over 80 iterations, the policy gradually learns to ask the right questions while respecting the budget.

## 5. Key Concepts

**What is the budget d₁?**

It's a soft constraint on the average number of questions across all episodes in a batch. d₁ = 2 does NOT mean "you can ask at most 2 questions per problem." It means "across all 256 episodes in a batch, the average should be ≤ 2."

This means the agent can ask 0 questions on easy problems and 4 on hard ones, as long as it averages out. This flexibility is what allows adaptive allocation to emerge.

**What are test suites?**

The assert statements that check code correctness. For problem #42:

```python
assert incr_list([1, 2, 3]) == [2, 3, 4]
assert incr_list([]) == []
assert incr_list([-1, 0, 1]) == [0, 1, 2]
```

You run the agent's code, then run these assertions. If all pass, pass@1 = 1.0. If 2/3 pass, pass@1 = 0.67. This gives a clean, automatic reward signal — no human judgment needed.

**What is a rollout?**

One complete episode — the agent starts with a problem, plays through the conversation (asking or answering), and the episode ends when it submits code. Each rollout produces one trajectory: a sequence of (state, action, reward, cost) tuples.

**What is an iteration?**

One cycle of the training loop: collect 256 rollouts → compute advantages → PPO update → update value heads → update λ. Then repeat.

**Why not just use SFT?**

- SFT produces one fixed behavior. The CMDP gives a family of policies indexed by budget.
- SFT can't learn when to stop asking — the early-termination strategy never appears in training data.
- SFT can't learn from bad questions — if the agent asks a useless question, SFT has no signal to discourage it.
- SFT can't learn adaptive allocation — spending more questions on hard problems, fewer on easy ones.

## 6. The Model

**Agent:** `meta-llama/Llama-3.2-3B-Instruct`

We use the smallest feasible Llama instruction-tuned model. This keeps iteration fast (3x faster than 8B), fits on a single A100, and still generates decent Python code. We can scale to 8B for final results if time permits.

**Adaptation:** LoRA (rank 16) on attention layers. The base 3B weights are frozen. Only the small LoRA adapter matrices (~20-40M parameters) are updated by PPO.

**Value heads:** Three small MLPs (separate from the LLM) that predict expected future reward, future question cost, and future turn cost from the current state. Used for computing advantages. Trained via regression alongside the policy.

**User simulator:** GPT-4o-mini via API (temperature 0 for reproducibility). Not trained — it's part of the environment.

## 7. Three-Phase Experimental Design

### Phase 1: Fixed Global Budget — The Pareto Frontier (Weeks 5-7)

Train 6 separate policies with d₁ ∈ {0, 1, 2, 3, 4, 5}. Evaluate each on the held-out test set. Plot pass@1 vs. average questions asked. Each point is a different policy. This is the core result.

Compare against baselines:

- **Never-Ask:** Base LLM, no questions
- **Always-Ask-k:** Prompted to ask exactly k questions
- **SFT:** Fine-tuned on gold clarification trajectories
- **Fixed-penalty RL:** PPO with hand-tuned question penalty, no Lagrangian
- **BED prompting:** Bayesian Experimental Design (Kobalczyk et al., 2025)

### Phase 2: Emergent Allocation Analysis (Week 8)

No new training. Analyze the Phase 1 policies: does the d₁ = 2 policy naturally ask more questions on ambiguous problems and fewer on clear ones? Measure ambiguity by sampling 10 solutions from the base model and counting distinct functional clusters.

### Phase 3: Instance-Adaptive Budgets (Weeks 9-10)

Use the ambiguity estimator from Phase 2 to assign per-problem budgets:

- Low ambiguity → d₁(τ) = 0
- Medium → d₁(τ) = 2
- High → d₁(τ) = 4

Train one policy with these per-problem budgets. Compare against the best fixed-budget policy from Phase 1 at the same average question count.

## 8. Implementation Plan

### Weeks 1-2: Environment Setup

- Download HumanEvalComm, parse into (degraded\_spec, original\_spec, test\_suite) format
- Run MBPP augmentation: prompt GPT-4o to degrade specs, validate ambiguity
- Build user simulator with prompt template
- Build code executor using HumanEval's sandboxed test harness
- Build the rollout collection loop (agent ↔ simulator ↔ executor)
- Train/test split: hold out 100 problems for evaluation

### Weeks 3-4: Baselines

- Run Never-Ask, Always-Ask-k, BED prompting on test set
- Generate gold clarification trajectories using GPT-4o, fine-tune SFT baseline
- Establish performance bounds

### Weeks 5-7: Phase 1 — PPO-Lagrangian

- Implement PPO with LoRA on Llama-3.2-3B-Instruct using TRL or OpenRLHF
- Add three value heads, Lagrangian dual update
- Train d₁ = 2 first (debug and tune hyperparameters)
- Full Pareto sweep: d₁ ∈ {0, 1, 2, 3, 4, 5}
- ~10-15 hours per d₁ setting on A100

### Week 8: Phase 2 — Allocation Analysis

- Compute ambiguity scores for all problems
- Plot questions asked vs. ambiguity for each Phase 1 policy

### Weeks 9-10: Phase 3 — Adaptive Budgets

- Assign per-problem budgets based on ambiguity bins
- Train one adaptive policy with per-problem d₁
- Compare against best fixed-budget policy

### Weeks 11-12: Analysis and Paper

- Generate all plots: Pareto frontier, constraint tracking, allocation curves
- Question quality audit (manual inspection of 50 questions)
- Write paper

## 9. Expected Results

1. **Pareto frontier** showing that different d₁ values produce different, useful policies. The frontier should dominate fixed-penalty baselines and SFT.
2. **Constraint satisfaction plot** showing PPO-Lagrangian tracks the specified budget d₁ within ±0.3, while fixed-penalty baselines deviate by ±1.0+.
3. **Adaptive allocation** — the globally-constrained policy naturally spends more questions on ambiguous problems without being told to.
4. **Instance-adaptive budgets** outperform the best fixed budget by 5-10% pass@1 at the same average question count.
5. **Question quality** improves under tighter budgets — the agent learns to ask more targeted questions, not just fewer.

## 10. Compute Requirements

- **Model:** Llama-3.2-3B-Instruct with LoRA (rank 16)
- **Hardware:** 1× A100-80GB (or equivalent)
- **Per d₁ setting:** ~80 iterations × ~10 min/iteration = ~13 hours
- **Full Phase 1 sweep (6 settings):** ~80 hours
- **Phase 3 (1 adaptive setting):** ~13 hours
- **User simulator API cost:** ~$5-10 total (GPT-4o-mini is cheap)
- **Total estimate:** ~100 GPU-hours + ~$10 API costs

## 11. Key References

- **HumanEvalComm** — Li et al., ACM TOSEM 2025. The benchmark we use.
- **Active Task Disambiguation** — Kobalczyk et al., ICLR 2025 Spotlight. Bayesian question selection (our BED baseline).
- **Learning to Clarify (ACT)** — ICLR 2025. Contrastive RL for clarification.
- **Modeling Future Conversation Turns** — ICLR 2025. RLHF for clarifying questions.
- **SWEET-RL** — Zhou et al., 2025. Multi-turn RL framework from Meta.
- **SAGE-Agent / ClarifyBench** — 2025. POMDP formulation for tool-call clarification.
- **PPO** — Schulman et al., 2017. The base RL algorithm.
- **LMRL-Gym** — Abdulhai et al., ICLR 2025. Multi-turn RL benchmarks for LLMs.
