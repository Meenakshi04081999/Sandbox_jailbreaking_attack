"""
Adversarial-suffix attacks on the white-box STUDENT.

We append an adversarial suffix to a base prompt the teacher/student BLOCKS and
optimize it to flip the decision to PASS (block-logit below threshold), keeping
the injection content fixed.

`AttackTarget` is the seam shared by both attacks on a fixed base prompt:
  - soft attack (here):  optimize continuous suffix embeddings  -> logit_from_embeds
  - hard attack (GCG, drops in next): optimize discrete suffix token ids using
    `onehot_grad` (rank token swaps) + `logit_from_ids` (evaluate candidates)

Label convention matches the student: block-logit high => BLOCK; target for the
attacker is PASS (logit low, p_block < 0.5).
"""
import torch
import torch.nn.functional as F


class AttackTarget:
    def __init__(self, student, prompt, n_suffix=10):
        self.student = student
        self.device = student.device
        self.prompt = prompt
        self.n_suffix = n_suffix
        self.word_emb = student.input_embeddings()           # nn.Embedding (V, H)

        enc = student.tokenize([prompt])
        self.base_ids = enc["input_ids"]                     # (1, L)
        self.base_mask = enc["attention_mask"]               # (1, L)
        self.base_embeds = self.word_emb(self.base_ids).detach()
        suffix_mask = torch.ones(1, n_suffix, device=self.device, dtype=self.base_mask.dtype)
        self.full_mask = torch.cat([self.base_mask, suffix_mask], dim=1)

    @property
    def embedding_matrix(self):
        return self.word_emb.weight                          # (V, H)

    # --- continuous path (soft attack) ---
    def logit_from_embeds(self, suffix_embeds):
        inp = torch.cat([self.base_embeds, suffix_embeds], dim=1)
        return self.student.forward_embeds(inp, self.full_mask)

    # --- discrete path (hard / GCG attack) ---
    def logit_from_ids(self, suffix_ids):
        suffix_embeds = self.word_emb(suffix_ids.view(1, -1))
        return self.logit_from_embeds(suffix_embeds)

    def onehot_grad(self, suffix_ids, target=0.0):
        """GCG primitive: gradient of the (toward-PASS) loss w.r.t. the one-hot
        suffix selection, shape (n_suffix, V). Tokens with the most negative
        gradient most reduce the loss if swapped in."""
        V = self.embedding_matrix.shape[0]
        oh = F.one_hot(suffix_ids.view(-1), V).to(self.embedding_matrix.dtype)
        oh.requires_grad_(True)
        suffix_embeds = (oh @ self.embedding_matrix).unsqueeze(0)   # (1, n, H)
        logit = self.logit_from_embeds(suffix_embeds)
        loss = F.binary_cross_entropy_with_logits(logit, torch.full_like(logit, target))
        loss.backward()
        return oh.grad.detach()


def soft_attack(target, steps=250, lr=0.05, seed=0):
    """Optimize a continuous suffix to flip the student block->pass."""
    g = torch.Generator().manual_seed(seed)
    V, _ = target.embedding_matrix.shape
    init_ids = torch.randint(0, V, (target.n_suffix,), generator=g).to(target.device)
    suffix = target.word_emb(init_ids).detach().clone().unsqueeze(0)   # (1, n, H)
    suffix.requires_grad_(True)

    opt = torch.optim.Adam([suffix], lr=lr)
    target_pass = None  # filled lazily
    success_step = None
    for t in range(steps):
        opt.zero_grad()
        logit = target.logit_from_embeds(suffix)
        if target_pass is None:
            target_pass = torch.zeros_like(logit)
        loss = F.binary_cross_entropy_with_logits(logit, target_pass)
        loss.backward()
        opt.step()
        if success_step is None and torch.sigmoid(logit.detach()).item() < 0.5:
            success_step = t

    with torch.no_grad():
        final_p = torch.sigmoid(target.logit_from_embeds(suffix.detach())).item()
    return {
        "suffix_embeds": suffix.detach(),
        "final_p_block": final_p,
        "success": final_p < 0.5,
        "success_step": success_step,
    }


@torch.no_grad()
def project_to_tokens(target, suffix_embeds):
    """Naive discretization: snap each soft vector to its nearest real token
    embedding (L2). This is the baseline the hard/GCG attack must beat."""
    s = suffix_embeds.squeeze(0)                 # (n, H)
    dists = torch.cdist(s, target.embedding_matrix)   # (n, V)
    return dists.argmin(dim=1)                   # (n,)
