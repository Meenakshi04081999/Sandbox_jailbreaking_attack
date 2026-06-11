"""
TEACHER -- the black-box safeguard model.

Threat model: the attacker may submit an input and receives ONLY a binary
label (pass / not-pass). `classify()` is the attacker-facing interface and
returns exactly that one bit -- nothing else may leak to the attack pipeline.

`classify_verbose()` exposes the raw label + score, but it is for OUR
research-side analysis only and must never be consumed by the student /
attack code, or the binary-only constraint is violated.

Default safeguard is a non-gated prompt-injection classifier so the sandbox
runs locally with no auth. To use a Llama Guard checkpoint instead, change
MODEL_ID / BLOCK_LABEL below -- the interface stays identical.
"""
# Validate TLS against the OS (Windows) certificate store rather than certifi,
# so HF downloads work behind a TLS-intercepting proxy. No-op if not installed.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import torch
from transformers import pipeline

MODEL_ID = "protectai/deberta-v3-base-prompt-injection-v2"

# The raw label the safeguard uses to mean "blocked" (i.e. not-pass).
BLOCK_LABEL = "INJECTION"


class Teacher:
    def __init__(self, model_id=MODEL_ID, device=None):
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(
            "text-classification",
            model=model_id,
            truncation=True,
            max_length=512,
            device=device,
        )
        # Query budget matters in the threat model -- track how many times the
        # attacker has hit the teacher.
        self.query_count = 0

    def classify_verbose(self, text):
        """Research-only: full label + score. Do NOT feed to the attacker."""
        self.query_count += 1
        out = self.pipe(text)[0]
        passed = out["label"] != BLOCK_LABEL
        return {
            "passed": passed,
            "label": out["label"],
            "score": float(out["score"]),
        }

    def classify(self, text):
        """Attacker-facing interface: the ONLY signal that leaves the teacher.

        Returns True  -> safeguard let the input through (pass)
                False -> safeguard blocked it (not-pass)
        """
        return self.classify_verbose(text)["passed"]
