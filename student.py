"""
STUDENT -- the attacker's white-box surrogate.

A frozen DistilBERT encoder + a trainable linear head. Trained to reproduce the
TEACHER's binary decision (block vs pass) from the distillation labels.

Design notes:
  - The encoder is frozen: with only a few hundred labels, fine-tuning the whole
    transformer would overfit. A linear head on [CLS] embeddings is effectively
    logistic regression on fixed features -- fast on CPU, well-behaved.
  - It is still fully white-box and differentiable end-to-end. `forward_embeds`
    runs the encoder from input embeddings, so the soft/hard-prompt attack can
    later take gradients w.r.t. the input and transfer to the teacher.

Label convention: target y = 1 means BLOCK (teacher not-pass), y = 0 means PASS.
"""
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

ENCODER_ID = "distilbert-base-uncased"


class Student(nn.Module):
    def __init__(self, encoder_id=ENCODER_ID, freeze_encoder=True, device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_id)
        self.encoder = AutoModel.from_pretrained(encoder_id)
        if freeze_encoder:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.head = nn.Linear(self.encoder.config.hidden_size, 1)
        self.to(self.device)

    # --- tokenization ---
    def tokenize(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, max_length=128,
                              return_tensors="pt").to(self.device)

    # --- forward paths ---
    def forward(self, input_ids, attention_mask):
        """Logit for BLOCK, from token ids."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.head(cls).squeeze(-1)

    def forward_embeds(self, inputs_embeds, attention_mask):
        """Logit for BLOCK, from input embeddings -- the attack entry point."""
        out = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.head(cls).squeeze(-1)

    def input_embeddings(self):
        """The encoder's token embedding matrix (for hard-prompt token search)."""
        return self.encoder.get_input_embeddings()

    # --- feature extraction (frozen encoder => cache once for training) ---
    @torch.no_grad()
    def embed(self, texts, batch_size=16):
        """Return [CLS] embeddings (N, H), no grad -- used to train/eval the head."""
        chunks = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenize(texts[i:i + batch_size])
            out = self.encoder(**enc)
            chunks.append(out.last_hidden_state[:, 0].cpu())
        return torch.cat(chunks, 0)

    # --- prediction ---
    @torch.no_grad()
    def block_prob(self, texts):
        emb = self.embed(texts).to(self.device)
        return torch.sigmoid(self.head(emb).squeeze(-1)).cpu()

    def predicts_block(self, texts, threshold=0.5):
        return (self.block_prob(texts) >= threshold).tolist()

    # --- persistence (only the head changes) ---
    def save_head(self, path):
        torch.save(self.head.state_dict(), path)

    def load_head(self, path):
        self.head.load_state_dict(torch.load(path, map_location=self.device))
