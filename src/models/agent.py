"""
Agent: LLM with LoRA adapters for the clarification task.

Responsibilities:
  generate()  → sample an action during rollout (returns text + log_prob + hidden_state)
  score()     → re-score a stored (state, action) pair during PPO update
  kl_penalty()→ KL divergence between current policy and frozen reference model
  save/load LoRA checkpoints
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType, PeftModel


class Agent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_device = torch.device(cfg.model.train_device)
        self.rollout_device = torch.device(cfg.model.rollout_device)
        self.max_new_tokens = cfg.environment.max_new_tokens
        self.max_seq_len = cfg.environment.max_seq_len
        self.rollout_temperature = cfg.environment.rollout_temperature
        self.dtype = torch.bfloat16 if cfg.model.dtype == "bfloat16" else torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.name,
            padding_side="left",
            truncation_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── Policy model (trainable, LoRA) on GPU 0 ──────────────────────────
        base = AutoModelForCausalLM.from_pretrained(
            cfg.model.name,
            torch_dtype=self.dtype,
            device_map=cfg.model.train_device,
        )
        if cfg.model.gradient_checkpointing:
            base.gradient_checkpointing_enable()

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.model.lora_rank,
            lora_alpha=cfg.model.lora_alpha,
            lora_dropout=cfg.model.lora_dropout,
            target_modules=list(cfg.model.lora_target_modules),
            bias="none",
        )
        self.policy = get_peft_model(base, lora_cfg)
        self.policy.print_trainable_parameters()

        # ── Reference model (frozen base, no LoRA) on GPU 1 ─────────────────
        self.reference = AutoModelForCausalLM.from_pretrained(
            cfg.model.name,
            torch_dtype=self.dtype,
            device_map=cfg.model.rollout_device,
        )
        for p in self.reference.parameters():
            p.requires_grad = False
        self.reference.eval()

        # ── Rollout copy (policy weights synced before each rollout) ─────────
        # We re-use the reference model device for rollout inference to avoid
        # having three full models in memory. The rollout copy is just the policy
        # with LoRA weights copied over, running in eval mode on GPU 1.
        # Updated via sync_rollout_model() before each rollout batch.
        self._rollout_model = None

        # ── Prefix token IDs for constrained decoding ────────────────────────
        # Force every generation to start with either [ASK] or [ANSWER].
        # The model chooses which prefix by comparing their log-probs given the
        # prompt, then generates freely after the prefix.
        self._ask_ids = self.tokenizer.encode("[ASK] ", add_special_tokens=False)
        self._answer_ids = self.tokenizer.encode("[ANSWER] ", add_special_tokens=False)

    def sync_rollout_model(self):
        """
        Copy current LoRA weights to GPU 1 for rollout inference.
        Called once per training iteration before collect_rollouts().
        """
        if self._rollout_model is None:
            # First sync: create the rollout model on GPU 1
            base1 = AutoModelForCausalLM.from_pretrained(
                self.cfg.model.name,
                torch_dtype=self.dtype,
                device_map=self.cfg.model.rollout_device,
            )
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.cfg.model.lora_rank,
                lora_alpha=self.cfg.model.lora_alpha,
                lora_dropout=self.cfg.model.lora_dropout,
                target_modules=list(self.cfg.model.lora_target_modules),
                bias="none",
            )
            self._rollout_model = get_peft_model(base1, lora_cfg)

        # Copy LoRA weights from policy (GPU 0) to rollout model (GPU 1)
        policy_state = {
            k: v.to(self.rollout_device)
            for k, v in self.policy.state_dict().items()
            if "lora_" in k
        }
        self._rollout_model.load_state_dict(policy_state, strict=False)
        self._rollout_model.eval()

    @torch.no_grad()
    def generate(
        self, prompt: str, constrain_prefix: bool = True,
    ) -> Tuple[str, List[int], float, torch.Tensor]:
        """
        Generate one action during rollout (runs on GPU 1).

        Args:
            prompt:            full text prompt string
            constrain_prefix:  if True, force output to start with [ASK] or [ANSWER].
                               Set False for non-env prompts (e.g. direct code generation).

        Returns:
            action_text:  raw generated text (e.g. "[ASK] What is X?")
            action_ids:   list of token IDs for the generated action
            action_logp:  sum of log-probs of action tokens (scalar float)
            state_hidden: hidden state of last prompt token (for value heads)
                          shape: (hidden_dim,) on CPU
        """
        model = self._rollout_model or self.policy
        model.eval()
        device = self.rollout_device if self._rollout_model else self.train_device

        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_seq_len,
            truncation=True,
            padding=False,
            add_special_tokens=False,  # chat template already includes special tokens
        ).to(device)

        prompt_len = enc.input_ids.shape[1]

        # Forward pass on prompt to get hidden state
        with torch.no_grad():
            outputs = model(
                **enc,
                output_hidden_states=True,
                use_cache=True,
            )
        last_hidden = outputs.hidden_states[-1][0, -1, :].cpu().float()

        if not constrain_prefix:
            # ── Unconstrained generation (e.g. direct code prompt) ────────
            gen_out = model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.rollout_temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
            action_ids = gen_out.sequences[0, prompt_len:].tolist()
            action_text = self.tokenizer.decode(action_ids, skip_special_tokens=True).strip()
            action_logp = _compute_action_logp(gen_out.scores, action_ids)
            return action_text, action_ids, action_logp, last_hidden

        # ── Constrained prefix selection ─────────────────────────────────
        # Use the prompt's last-position logits to score the first token of
        # each prefix, then continue autoregressively for remaining tokens.
        prompt_logits = outputs.logits[0, -1, :]  # logits predicting first token after prompt

        ask_logp = _score_prefix_from_logits(
            model, prompt_logits, outputs.past_key_values, self._ask_ids, device
        )
        ans_logp = _score_prefix_from_logits(
            model, prompt_logits, outputs.past_key_values, self._answer_ids, device
        )

        # Sample prefix proportional to exp(logp) (i.e., softmax over the two)
        logps = torch.tensor([ask_logp, ans_logp])
        probs = torch.softmax(logps / self.rollout_temperature, dim=0)
        choice = torch.multinomial(probs, 1).item()

        if choice == 0:
            prefix_ids = self._ask_ids
            prefix_logp = ask_logp
        else:
            prefix_ids = self._answer_ids
            prefix_logp = ans_logp

        # ── Generate continuation after the prefix ───────────────────────
        prefix_tensor = torch.tensor([prefix_ids], device=device)
        full_input = torch.cat([enc.input_ids, prefix_tensor], dim=1)
        full_mask = torch.ones_like(full_input)

        gen_out = model.generate(
            input_ids=full_input,
            attention_mask=full_mask,
            max_new_tokens=self.max_new_tokens - len(prefix_ids),
            do_sample=True,
            temperature=self.rollout_temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # Combine prefix + generated continuation
        continuation_ids = gen_out.sequences[0, prompt_len + len(prefix_ids):].tolist()
        action_ids = prefix_ids + continuation_ids
        action_text = self.tokenizer.decode(action_ids, skip_special_tokens=True).strip()

        # Total log-prob = prefix log-prob + continuation log-prob
        continuation_logp = _compute_action_logp(gen_out.scores, continuation_ids)
        action_logp = prefix_logp + continuation_logp

        return action_text, action_ids, action_logp, last_hidden

    def score(
        self,
        prompt: str,
        action_ids: List[int],
    ) -> Tuple[float, torch.Tensor, float]:
        """
        Re-score a (prompt, action) pair during PPO update (runs on GPU 0).

        Args:
            prompt:     full text prompt string
            action_ids: token IDs of the action to score

        Returns:
            new_logp:     sum of log-probs under current policy (scalar tensor)
            state_hidden: hidden state of last prompt token (for value heads)
                          shape: (hidden_dim,) on train_device
            ref_logp:     sum of log-probs under frozen reference model (scalar float)
        """
        self.policy.train()

        enc_prompt = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_seq_len,
            truncation=True,
            padding=False,
            add_special_tokens=False,  # chat template already includes special tokens
        ).to(self.train_device)

        prompt_ids = enc_prompt.input_ids  # (1, prompt_len)
        action_tensor = torch.tensor(action_ids, device=self.train_device).unsqueeze(0)
        full_ids = torch.cat([prompt_ids, action_tensor], dim=1)
        attention_mask = torch.ones_like(full_ids)
        prompt_len = prompt_ids.shape[1]

        # Policy forward pass
        policy_out = self.policy(
            input_ids=full_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Hidden state at last prompt token
        state_hidden = policy_out.hidden_states[-1][0, prompt_len - 1, :]  # (hidden_dim,)

        # Log-probs of action tokens
        logits = policy_out.logits[0, prompt_len - 1: -1, :]  # (action_len, vocab)
        new_logp = _token_logp_sum(logits, action_tensor[0])

        # Reference model log-probs (no grad, GPU 1)
        with torch.no_grad():
            ref_enc_prompt = enc_prompt.to(self.rollout_device)
            ref_action = action_tensor.to(self.rollout_device)
            ref_full = torch.cat([ref_enc_prompt.input_ids, ref_action], dim=1)
            ref_mask = torch.ones_like(ref_full)
            ref_out = self.reference(input_ids=ref_full, attention_mask=ref_mask)
            ref_logits = ref_out.logits[0, prompt_len - 1: -1, :].to(self.train_device)
            ref_logp = _token_logp_sum(ref_logits, action_tensor[0]).detach()

        return new_logp, state_hidden, ref_logp

    def save_lora(self, path: str):
        """Save LoRA adapter weights to disk."""
        self.policy.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_lora(self, path: str):
        """Load LoRA adapter weights from disk into the policy model."""
        self.policy = PeftModel.from_pretrained(
            self.policy.base_model.model,
            path,
            device_map=self.cfg.model.train_device,
        )


# ── Helper functions ──────────────────────────────────────────────────────────

def _score_prefix_from_logits(
    model,
    prompt_logits: torch.Tensor,
    past_key_values,
    prefix_ids: List[int],
    device: torch.device,
) -> float:
    """
    Score a fixed prefix using the prompt's logits for the first token,
    then autoregressively for subsequent tokens.

    No KV cache mutation — we re-encode the prefix cheaply (only a few tokens).
    """
    total_logp = 0.0

    # Score first prefix token using the prompt's last logits
    log_probs = F.log_softmax(prompt_logits, dim=-1)
    total_logp += log_probs[prefix_ids[0]].item()

    if len(prefix_ids) == 1:
        return total_logp

    # Score remaining prefix tokens by feeding them through the model
    # Re-encode prompt + prefix[:i] to avoid cache mutation issues
    # Since prefixes are short (3-5 tokens), this is cheap
    prefix_tensor = torch.tensor([prefix_ids], device=device)
    for i in range(1, len(prefix_ids)):
        # Feed tokens up to position i, get logits predicting position i
        inp = prefix_tensor[:, :i]
        with torch.no_grad():
            out = model(input_ids=inp, past_key_values=past_key_values, use_cache=False)
        logits = out.logits[0, -1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        total_logp += log_probs[prefix_ids[i]].item()

    return total_logp


def _compute_action_logp(scores, action_ids: List[int]) -> float:
    """
    Compute sum of log-probs from model.generate() scores output.

    scores: tuple of (vocab_size,) tensors, one per generated token
    action_ids: list of token IDs that were generated
    """
    total = 0.0
    for score, tok_id in zip(scores, action_ids):
        log_probs = F.log_softmax(score[0], dim=-1)
        total += log_probs[tok_id].item()
    return total


def _token_logp_sum(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """
    Compute sum of log-probs for a sequence of token IDs.

    Args:
        logits:    (seq_len, vocab_size) — shifted so logits[i] predicts token_ids[i]
        token_ids: (seq_len,)

    Returns:
        scalar tensor (gradient-connected)
    """
    log_probs = F.log_softmax(logits, dim=-1)  # (seq_len, vocab_size)
    selected = log_probs[torch.arange(len(token_ids)), token_ids]  # (seq_len,)
    return selected.sum()
