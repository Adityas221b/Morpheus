"""Scorer backends — decouple the judge LLM from how it is served.

Mirrors the target abstraction: scorers call `backend.generate(prompt, images)`
and don't care whether the model lives in Ollama or is loaded in-process via
Hugging Face transformers.

Backends:
  OllamaScorerBackend : HTTP to an ollama daemon (`/api/generate`, `/api/chat`).
  HFScorerBackend     : in-process inference via transformers; supports both
                        causal LMs (text-only) and vision-language models
                        (image_gen scorer with images).

Pick via `scoring.provider` in settings.yaml; default is "ollama".
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import sys
import os
import time
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Optional

# Force UTF-8 encoding for all stdout/stderr on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


class ScorerBackend(ABC):
    """Send a scoring prompt (with optional images) and return the judge's text reply."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        images: Optional[list[bytes]] = None,
    ) -> str:
        ...

    async def warmup(self) -> None:
        """Eagerly load weights so the first scoring call doesn't pay the cost.

        No-op for HTTP-backed providers; HFScorerBackend overrides this.
        """
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Ollama
# ─────────────────────────────────────────────────────────────────────────────

class OllamaScorerBackend(ScorerBackend):
    """Calls an Ollama server. Uses /api/chat when images are present, else /api/generate."""

    def __init__(
        self,
        config: dict,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ):
        self.config = config
        self.model = config.get("model", "qwen3:8b")
        self.base_url = base_url
        self.timeout = timeout
        # `think: True` is an ollama-specific knob (qwen3 thinking mode). Default true
        # to match prior behavior; surface as a config flag for callers that want it off.
        self.think = bool(config.get("think", True))

    async def generate(
        self,
        prompt: str,
        images: Optional[list[bytes]] = None,
    ) -> str:
        encoded = [base64.b64encode(b).decode("utf-8") for b in (images or [])]

        if encoded:
            payload = json.dumps({
                "model": self.model,
                "messages": [{"role": "user", "content": prompt, "images": encoded}],
                "stream": False,
            }).encode("utf-8")
            endpoint = f"{self.base_url}/api/chat"
        else:
            payload = json.dumps({
                "model": self.model,
                "prompt": prompt,
                "think": self.think,
                "stream": False,
            }).encode("utf-8")
            endpoint = f"{self.base_url}/api/generate"

        def _do_request() -> str:
            req = urllib.request.Request(
                endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
                if encoded:
                    return data.get("message", {}).get("content", "").strip()
                return data.get("response", "").strip()

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _do_request)


# ─────────────────────────────────────────────────────────────────────────────
# Hugging Face (in-process)
# ─────────────────────────────────────────────────────────────────────────────

class HFScorerBackend(ScorerBackend):
    """Runs the judge model in-process via transformers.

    Auto-detects VLM vs causal LM from the model class (or from `model_class:`
    config). When images are passed and the model is a VLM, they are routed
    through the processor's chat template; otherwise images are dropped with
    a warning.

    Config keys (settings.yaml -> scoring):
      model              HF repo id (e.g. "Qwen/Qwen2.5-VL-7B-Instruct")
      device             "auto" | "cuda" | "cuda:0" | "cpu" | "mps"   (default "auto")
      dtype              "float16" | "bfloat16" | "float32"            (default float16 on GPU)
      trust_remote_code  bool                                          (default False)
      model_class        transformers class override                   (auto-detected)
      max_new_tokens     int                                           (default 512)
      temperature        float                                         (default 0.0 — judges should be deterministic)
      top_p              float                                         (default 1.0)
      do_sample          bool                                          (default False)
      attn_implementation  "eager" | "sdpa" | "flash_attention_2"  (default: transformers' choice)
                         Set to "eager" for Llama-Guard-3-Vision / Mllama if you
                         see "view size is not compatible" runtime errors.
      bnb                dict — on-the-fly bitsandbytes quantization (CUDA only):
                           bits: "4bit" | "8bit"  (default "4bit")
                           quant_type: "nf4" | "fp4"  (4-bit only, default "nf4")
                           double_quant: bool  (4-bit only, default True)
                         Requires `pip install bitsandbytes accelerate`.
    """

    def __init__(self, config: dict, timeout: float = 120.0):
        self.config = config
        self.model_id = config.get("model")
        if not self.model_id:
            raise ValueError("HFScorerBackend requires scoring.model to be set.")

        self.device_pref = config.get("device", "auto")
        self.dtype_str: Optional[str] = config.get("dtype")
        self.trust_remote_code = bool(config.get("trust_remote_code", False))
        self.timeout = timeout

        self._model: Any = None
        self._tokenizer: Any = None
        self._processor: Any = None
        self._device: Optional[str] = None
        self._dtype: Any = None
        self._is_vlm: bool = False
        self._load_lock = asyncio.Lock()

    # ── Loading ────────────────────────────────────────────────────────────

    def _resolve_device_dtype(self):
        import torch

        if self.device_pref == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = self.device_pref

        if self.dtype_str:
            dtype = getattr(torch, self.dtype_str)
        else:
            dtype = torch.float16 if device.startswith(("cuda", "mps")) else torch.float32
        return device, dtype

    async def warmup(self) -> None:
        if self._model is not None:
            return
        print(f"  [scorer:{self.model_id}] downloading + loading via huggingface on {self.device_pref} ...")
        start = time.monotonic()
        await self._ensure_loaded()
        device = self._device or self.device_pref
        kind = "VLM" if self._is_vlm else "causal-LM"
        print(f"  [scorer:{self.model_id}] ready as {kind} on {device} ({self._dtype}) — load took {time.monotonic() - start:.1f}s")

    async def _ensure_loaded(self):
        if self._model is not None:
            return
        async with self._load_lock:
            if self._model is not None:
                return
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self):
        import transformers as _tf
        from transformers import AutoConfig, AutoTokenizer, AutoProcessor

        device, dtype = self._resolve_device_dtype()
        self._device = device
        self._dtype = dtype

        kwargs: dict = {"torch_dtype": dtype}
        if self.trust_remote_code:
            kwargs["trust_remote_code"] = True
        kwargs["device_map"] = self.config.get("device_map", device)

        # SDPA (default in recent transformers) hits a `.view()` on a non-contiguous
        # tensor inside Mllama (Llama-Guard-3-Vision) cross-attention; 8-bit bnb
        # makes it deterministic. Set `attn_implementation: eager` for those models.
        attn_impl = self.config.get("attn_implementation")
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl

        # bitsandbytes on-the-fly quantization (CUDA only). 4-bit nf4 cuts an
        # 11B VLM (e.g. Llama-Guard-3-Vision) from ~22 GB to ~6 GB.
        bnb_cfg = self.config.get("bnb")
        if bnb_cfg:
            from transformers import BitsAndBytesConfig

            bits = str(bnb_cfg.get("bits", "4bit")).lower()
            if bits in ("4", "4bit", "int4", "nf4"):
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type=bnb_cfg.get("quant_type", "nf4"),
                    bnb_4bit_use_double_quant=bool(bnb_cfg.get("double_quant", True)),
                )
            elif bits in ("8", "8bit", "int8"):
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                raise ValueError(f"scoring.bnb.bits must be '4bit' or '8bit', got {bits!r}")

        forced = self.config.get("model_class")
        if forced:
            model_class = getattr(_tf, forced)
            self._is_vlm = (
                "Vision" in forced
                or "Image" in forced
                or "Mllama" in forced
                or forced.endswith("ForConditionalGeneration")
            )
        else:
            # Authoritative VLM detection: read `architectures` from the model's
            # config.json. Avoids the prior probe-and-fallback, which silently
            # demoted Mllama (Llama-Guard-3-Vision) to a text-only causal LM
            # whenever AutoProcessor.from_pretrained raised for any reason.
            cfg = AutoConfig.from_pretrained(
                self.model_id, trust_remote_code=self.trust_remote_code
            )
            archs = getattr(cfg, "architectures", None) or []
            arch = archs[0] if archs else ""
            self._is_vlm = bool(arch) and (
                "Vision" in arch
                or "Image" in arch
                or "Mllama" in arch
                or "VL" in arch
                or arch.endswith("ForConditionalGeneration")
            )

            if self._is_vlm:
                # AutoModelForImageTextToText (transformers >= 4.45) covers Mllama /
                # Llava-Next / Qwen-VL; AutoModelForVision2Seq is the older catch-all
                # (BLIP, GIT, etc.). Try the newer name first, fall back. Don't use
                # getattr-with-default — Python evaluates the default eagerly and
                # transformers' lazy-import raises if the fallback name is unknown.
                model_class = None
                for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
                    try:
                        model_class = getattr(_tf, name)
                        break
                    except (ImportError, AttributeError):
                        continue
                if model_class is None:
                    raise RuntimeError(
                        f"Detected {self.model_id} as a VLM (architecture={arch!r}) "
                        f"but this transformers version exposes neither "
                        f"AutoModelForImageTextToText nor AutoModelForVision2Seq. "
                        f"Upgrade with `pip install -U transformers` (need >= 4.45 for Mllama)."
                    )
            else:
                model_class = _tf.AutoModelForCausalLM

        if self._is_vlm:
            try:
                self._processor = AutoProcessor.from_pretrained(
                    self.model_id, trust_remote_code=self.trust_remote_code
                )
            except Exception as e:
                raise RuntimeError(
                    f"Detected {self.model_id} as a VLM (architecture={arch if not forced else forced!r}) "
                    f"but AutoProcessor.from_pretrained failed: {e}. "
                    f"Common causes: transformers too old (need >= 4.45 for Mllama), "
                    f"missing pillow, or gated-repo auth (set HF_TOKEN / huggingface-cli login)."
                ) from e
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=self.trust_remote_code
            )

        model = model_class.from_pretrained(self.model_id, **kwargs)
        model.eval()
        self._model = model

    # ── Inference ──────────────────────────────────────────────────────────

    def _gen_kwargs(self) -> dict:
        gen_kwargs: dict = {
            "max_new_tokens": int(self.config.get("max_new_tokens", 512)),
            "temperature": float(self.config.get("temperature", 0.0)),
            "top_p": float(self.config.get("top_p", 1.0)),
            "do_sample": bool(self.config.get("do_sample", False)),
        }
        if not gen_kwargs["do_sample"]:
            gen_kwargs.pop("temperature", None)
            gen_kwargs.pop("top_p", None)
        return gen_kwargs

    async def generate(
        self,
        prompt: str,
        images: Optional[list[bytes]] = None,
    ) -> str:
        await self._ensure_loaded()

        if images and not self._is_vlm:
            print(
                f"  [WARN] scorer {self.model_id}: {len(images)} image(s) provided but "
                f"model is not a VLM; dropping images and scoring on text only."
            )
            images = None

        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, self._infer_sync, prompt, list(images or [])),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError as e:
            raise RuntimeError(f"Scorer inference timed out after {self.timeout}s") from e

    def _infer_sync(self, prompt: str, image_bytes_list: list[bytes]) -> str:
        import torch
        from PIL import Image as PILImage

        target_device = self._device or "cpu"
        gen_kwargs = self._gen_kwargs()

        if self._is_vlm:
            pil_images = [PILImage.open(io.BytesIO(b)).convert("RGB") for b in image_bytes_list]

            content: list[dict] = []
            for _ in pil_images:
                content.append({"type": "image"})
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]

            try:
                prompt_str = self._processor.apply_chat_template(
                    messages, add_generation_prompt=True
                )
            except Exception:
                prompt_str = prompt

            proc_kwargs: dict = {"text": prompt_str, "return_tensors": "pt"}
            if pil_images:
                proc_kwargs["images"] = pil_images
            inputs = self._processor(**proc_kwargs).to(target_device)

            with torch.inference_mode():
                output_ids = self._model.generate(**inputs, **gen_kwargs)

            input_len = inputs["input_ids"].shape[-1]
            new_tokens = output_ids[0, input_len:]
            return self._processor.batch_decode(
                [new_tokens], skip_special_tokens=True
            )[0].strip()

        tokenizer = self._tokenizer
        try:
            prompt_str = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt_str = prompt

        inputs = tokenizer(prompt_str, return_tensors="pt").to(target_device)
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                **gen_kwargs,
            )
        input_len = inputs["input_ids"].shape[-1]
        new_tokens = output_ids[0, input_len:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    async def aclose(self):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._unload_sync)

    def _unload_sync(self):
        self._model = None
        self._tokenizer = None
        self._processor = None
        try:
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# NVIDIA NIM (Remote API)
# ─────────────────────────────────────────────────────────────────────────────

class NIMScorerBackend(ScorerBackend):
    """Calls NVIDIA NIM API (OpenAI-compatible) for scoring.

    Config keys (settings.yaml -> scoring):
      api_key    NVIDIA NIM API key (or env NIM_API_KEY)
      model      Model ID (default: meta/llama-3.2-11b-vision-instruct)
      base_url   API endpoint (default: https://integrate.api.nvidia.com/v1)
    """

    def __init__(self, config: dict, timeout: float = 120.0):
        self.config = config
        self.model = config.get("model", "meta/llama-3.2-11b-vision-instruct")
        self.base_url = config.get("base_url", "https://integrate.api.nvidia.com/v1")
        self.timeout = timeout
        self.api_key = config.get("api_key") or os.environ.get("NIM_API_KEY", "")
        if not self.api_key:
            raise ValueError("NIMScorerBackend requires scoring.api_key or NIM_API_KEY env var.")

    async def generate(
        self,
        prompt: str,
        images: Optional[list[bytes]] = None,
    ) -> str:
        import base64 as b64

        content: list[dict] = []
        if images:
            for img_bytes in images:
                b64_img = b64.b64encode(img_bytes).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                })
        content.append({"type": "text", "text": prompt})

        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 1024,
            "temperature": 0.0,
        }).encode("utf-8")

        def _do_request() -> str:
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
                return data["choices"][0]["message"]["content"].strip()

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _do_request)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def make_scorer_backend(
    config: dict,
    ollama_url: str = "http://localhost:11434",
    timeout: float = 120.0,
) -> ScorerBackend:
    """Build the scorer backend selected by `config.provider` (default 'ollama')."""
    provider = str(config.get("provider", "ollama")).lower()
    if provider in ("huggingface", "hf", "transformers"):
        return HFScorerBackend(config, timeout=timeout)
    if provider == "ollama":
        return OllamaScorerBackend(config, base_url=ollama_url, timeout=timeout)
    if provider == "nim":
        return NIMScorerBackend(config, timeout=timeout)
    raise ValueError(
        f"Unknown scoring.provider={provider!r}. Expected one of: ollama, huggingface, nim."
    )
