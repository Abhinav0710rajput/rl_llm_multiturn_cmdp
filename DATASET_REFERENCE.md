# HumanEvalComm Dataset Reference

Source: https://huggingface.co/datasets/jie-jw-wu/HumanEvalComm
Paper: https://arxiv.org/abs/2406.00215
License: Apache 2.0

---

## Loading

```python
from datasets import load_dataset

ds = load_dataset("jie-jw-wu/HumanEvalComm", split="train")  # only split is "train"
rows = list(ds)   # 164 rows
```

---

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `name` | str | Problem ID, e.g. `"HumanEval/42"` |
| `entry_point` | str | Function name, e.g. `"incr_list"` |
| `prompt` | str | Original full spec (function signature + complete docstring + examples) |
| `prompt1a` | str | Ambiguity degradation: key specifics replaced with vague terms |
| `prompt1c` | str | Inconsistency degradation: examples contradict the description |
| `prompt1p` | str | Incompleteness degradation: docstring stripped to minimal stub |
| `prompt2ac` | str | Ambiguity + Inconsistency combined |
| `prompt2ap` | str | Ambiguity + Incompleteness combined |
| `prompt2cp` | str | Inconsistency + Incompleteness combined |
| `prompt3acp` | str | All three combined (rarest) |
| `solution` | str | Full reference Python solution |
| `test_case` | str | Python literal (list of dicts): `[{input, output, relation}, ...]` |

**Note:** `test_case` is stored as a Python-literal string — use `ast.literal_eval()`, not `json.loads()`.

```python
import ast
tests = ast.literal_eval(row['test_case'])
# tests[0] → {'input': '[1, 2, 3]', 'output': '[2, 3, 4]', 'relation': '=='}
```

---

## Dataset Size & Coverage

| Split | Rows |
|-------|------|
| train (only split) | **164** base problems |

| Degradation Field | Non-null Count | % Coverage |
|---|---|---|
| `prompt` (original) | 164 | 100% |
| `prompt1a` (ambiguity) | 164 | 100% |
| `prompt1c` (inconsistency) | 163 | 99% |
| `prompt1p` (incompleteness) | 164 | 100% |
| `prompt2ac` | 162 | 99% |
| `prompt2ap` | 71 | 43% |
| `prompt2cp` | 35 | 21% |
| `prompt3acp` | 12 | 7% |
| **Total variants** | **~835** | — |

The three core single-degradation types (1a, 1c, 1p) are fully covered. Combined variants thin out quickly — only 12 problems have the triple-degraded `prompt3acp`. For training we primarily use `1a`, `1c`, `1p` and `2ac`; the sparser types are supplementary.

---

## Prompt Length Statistics

### Characters

| Field | Min | Max | Mean | Median |
|-------|-----|-----|------|--------|
| `prompt` (original) | 115 | 1360 | 451 | 396 |
| `prompt1a` | 143 | 1341 | 456 | 409 |
| `prompt1c` | 114 | 1360 | 449 | 400 |
| `prompt1p` | 62 | 695 | 204 | 173 |
| `prompt2ac` | 142 | 1341 | 458 | 414 |
| `prompt2ap` | 67 | 629 | 209 | 181 |
| `prompt2cp` | 82 | 525 | 264 | 262 |

### Words

| Field | Min | Max | Mean | Median |
|-------|-----|-----|------|--------|
| `prompt` (original) | 17 | 249 | 68 | 60 |
| `prompt1p` (most stripped) | 8 | 119 | 31 | 26 |

**Key insight:** `prompt1p` is on average **48% the size** of the original prompt (range: 19%–97%). This is the degradation that most radically strips context — the agent must ask to recover examples and specifics.

---

## Solution Length Statistics

| Metric | Value |
|--------|-------|
| Min | 132 chars / 20 words |
| Max | 1993 chars / 327 words |
| Mean | 631 chars / 92 words |
| Median | 572 chars / 84 words |

The typical solution is ~85 words of Python. The longest problems involve grid/path algorithms; the shortest are one-liners.

---

## Test Case Statistics

Test cases are stored as Python-literal lists. Each entry has `input` (string), `output` (string), and `relation` (`"=="` or custom float-tolerance expression).

| Metric | Value |
|--------|-------|
| Min per problem | 1 |
| Max per problem | 105 |
| Mean per problem | 9.2 |
| Median per problem | 7 |

**Relation types across all test cases:**
- `==` (exact equality): **1,411** test cases (93%)
- `abs(...)` (float tolerance): **105** test cases (7%)

### Extreme Test Case Counts

| Most tests | Count | Fewest tests | Count |
|---|---|---|---|
| HumanEval/53 `add` | 105 | HumanEval/34 `unique` | 1 |
| HumanEval/32 `find_zero` | 100 | HumanEval/35 `max_element` | 2 |
| HumanEval/38 `decode_cyclic` | 100 | HumanEval/29 `filter_by_prefix` | 2 |
| HumanEval/50 `decode_shift` | 100 | HumanEval/113 `odd_count` | 3 |
| HumanEval/141 `file_name_check` | 26 | HumanEval/160 `do_algebra` | 3 |

---

## Longest & Shortest Problems

### Longest Original Prompts

| Problem | Function | Chars | Words |
|---------|----------|-------|-------|
| HumanEval/129 | `minPath` | 1360 | 249 |
| HumanEval/109 | `move_one_ball` | 1265 | 185 |
| HumanEval/68 | `pluck` | 1167 | 180 |
| HumanEval/153 | `Strongest_Extension` | 1053 | 155 |
| HumanEval/115 | `max_fill` | 1050 | 148 |

### Shortest Original Prompts

| Problem | Function | Chars | Words |
|---------|----------|-------|-------|
| HumanEval/53 | `add` | 115 | 20 |
| HumanEval/55 | `fib` | 130 | 17 |
| HumanEval/23 | `strlen` | 133 | 18 |
| HumanEval/45 | `triangle_area` | 138 | 20 |
| HumanEval/34 | `unique` | 149 | 27 |

### Longest Solutions

| Problem | Function | Chars | Words |
|---------|----------|-------|-------|
| HumanEval/129 | `minPath` | 1993 | 327 |
| HumanEval/81 | `numerical_letter_grade` | 1899 | 198 |
| HumanEval/109 | `move_one_ball` | 1557 | 204 |
| HumanEval/153 | `Strongest_Extension` | 1490 | 223 |
| HumanEval/95 | `check_dict_case` | 1359 | 136 |

### Shortest Solutions

| Problem | Function | Chars | Words |
|---------|----------|-------|-------|
| HumanEval/53 | `add` | 132 | 24 |
| HumanEval/23 | `strlen` | 156 | 20 |
| HumanEval/45 | `triangle_area` | 161 | 26 |
| HumanEval/34 | `unique` | 181 | 29 |
| HumanEval/27 | `flip_case` | 208 | 25 |

---

## Duplicate Entry Points

6 function names appear in more than one problem (different problems, same function name):

```
triangle_area, add, correct_bracketing, solve, sort_array, sum_squares
```

**Implication:** When parsing results or building a lookup by `entry_point`, use `name` (HumanEval/N) as the primary key, not `entry_point`.

---

## Example: All Degradations for HumanEval/42 (`incr_list`)

```python
# prompt (original)
def incr_list(l: list):
    """Return list with elements incremented by 1.
    >>> incr_list([1, 2, 3])
    [2, 3, 4]
    >>> incr_list([5, 3, 5, 2, 3, 3, 9, 0, 123])
    [6, 4, 6, 3, 4, 4, 10, 1, 124]
    """

# prompt1a (ambiguity — "by 1" → "by a number")
def incr_list(l: list):
    """Return list with elements incremented by a number.
    >>> incr_list([1, 2, 3])
    [2, 3, 4]
    ...
    """

# prompt1c (inconsistency — examples show +2 but correct answer is +1)
def incr_list(l: list):
    """Return list with elements incremented by 1.
    >>> incr_list([1, 2, 3])
    [3, 4, 5]    ← wrong (shows +2)
    ...
    """

# prompt1p (incompleteness — all specifics stripped)
def incr_list(l: list):
    """Return list with elements incremented.
    """

# prompt2ac (ambiguity + inconsistency)
def incr_list(l: list):
    """Return list with elements incremented by a number.
    >>> incr_list([1, 2, 3])
    [3, 4, 5]    ← shows +2, description says "a number"
    ...
    """
```

**Test cases for HumanEval/42** (3 total, all exact equality):
```
input=[]               → output=[]
input=[3, 2, 1]        → output=[4, 3, 2]
input=[5,2,5,2,3,3,9,0,123] → output=[6,3,6,3,4,4,10,1,124]
```

---

## Implications for Our Pipeline

| Observation | What it means for us |
|---|---|
| Max prompt: 1360 chars / ~250 words | `max_seq_len=1536` is sufficient; even with 6-turn history the context fits |
| `prompt1p` is 48% size on average | Incompleteness variant is most under-specified — expect most questions here |
| Median test cases: 7 | pass@1 granularity is 1/7 ≈ 14% increments for typical problem; reward signal has reasonable resolution |
| 105 test cases for `add` | A few problems have very dense test suites — partial credit scoring (fraction passed) matters more for these |
| 1516 total test assertions across 164 problems | Code executor will run ~9 assertions per episode on average |
| 6 duplicate `entry_point` names | Use `name` field (HumanEval/N) as primary key in all data structures |
| `prompt2ap`, `prompt2cp`, `prompt3acp` are sparse | All 7 types are included in training. Stratified split ensures they appear in both train and eval sets |
| Degraded specs rename functions to `candidate` | Code executor renames agent's function to match `entry_point`. Simulator prompt tells GPT-4o-mini to treat any name as the spec'd function |
