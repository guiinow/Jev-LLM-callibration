"""Hugging Face `transformers` backend for the open models (needs the `open-models` extra).

Weights load on first use, so building a client (and `--mock` runs, tests, `analyze`) never
imports torch. One backend serves both readouts of a model, so "<model>_prob" and
"<model>_verb" always come from the same weights, dtype and quantization.

Untested against the real Gemma 4 / Qwen 3.8 checkpoints in this repository's CI (they need a
large GPU or Apple-silicon memory); `tests/test_open_models.py` covers the logic with a fake.
"""
from typing import Any


class HfBackend:
    def __init__(self, hf_id: str, revision: str, dtype: str = "bfloat16", auto_class: str = "AutoModelForCausalLM",
                 quantization: str | None = None, device_map: str = "auto") -> None:
        self.hf_id, self.revision = hf_id, revision
        self.dtype, self.auto_class, self.quantization, self.device_map = dtype, auto_class, quantization, device_map
        self.model_id = f"{hf_id}@{revision[:12]}"
        self._tokenizer = None
        self._model = None
        self._torch = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        import transformers
        self._torch = torch
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.hf_id, revision=self.revision)
        kwargs: dict[str, Any] = {"revision": self.revision, "dtype": getattr(torch, self.dtype),
                                  "device_map": self.device_map}
        if self.quantization in ("4bit", "8bit"):
            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=self.quantization == "4bit", load_in_8bit=self.quantization == "8bit",
                bnb_4bit_compute_dtype=getattr(torch, self.dtype))
        elif self.quantization is not None:
            raise ValueError(f"quantization must be null, '4bit' or '8bit', got {self.quantization!r}")
        self._model = getattr(transformers, self.auto_class).from_pretrained(self.hf_id, **kwargs).eval()

    def _inputs(self, user_text: str):
        self._load()
        text = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True,
            enable_thinking=False)  # thinking off: answer directly (extra kwarg is ignored by other templates)
        return self._tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self._model.device)

    def single_token_id(self, text: str) -> int | None:
        self._load()
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    def next_token_logprobs(self, user_text: str, token_ids: list[int]) -> tuple[list[float], int]:
        inputs = self._inputs(user_text)
        with self._torch.no_grad():
            logits = self._model(**inputs).logits[0, -1].float()
        logprobs = self._torch.log_softmax(logits, dim=-1)[token_ids]
        return logprobs.tolist(), int(inputs["input_ids"].shape[1])

    def generate(self, user_text: str, max_new_tokens: int) -> tuple[str, dict[str, Any]]:
        inputs = self._inputs(user_text)
        with self._torch.no_grad():
            output = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text, {"prompt_tokens": int(inputs["input_ids"].shape[1]), "completion_tokens": int(len(new_tokens))}
