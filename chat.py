print("Eğitim başlıyor...")

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- VERİ ---
with open('veriseti.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print(f"Veri seti yüklendi. Toplam karakter: {len(text)}")

# --- AYARLAR ---
batch_size        = 8
block_size        = 512
max_iters         = 20000
eval_interval     = 1000
learning_rate     = 3e-4
device            = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters        = 200
n_enmbd           = 768
n_head            = 12
n_layer           = 12
dropout           = 0.1
accumulation_steps = 4

torch.manual_seed(1337)

# --- TOKENİZER ---
print("Tokenizer yükleniyor...")
tokenizer = AutoTokenizer.from_pretrained("ytu-ce-cosmos/turkish-gpt2")
vocab_size  = len(tokenizer)
eos_token_id = tokenizer.eos_token_id

encode = lambda s: tokenizer.encode(s)
decode = lambda l: tokenizer.decode(l)

# --- TOKENİZASYON ---
chunk_size = 1_000_000
all_tokens = []
for i in range(0, len(text), chunk_size):
    chunk = text[i:i+chunk_size]
    all_tokens.extend(encode(chunk))
    yuzde = int((i / len(text)) * 100)
    print(f"  Tokenizasyon: %{yuzde}")

data = torch.tensor(all_tokens, dtype=torch.long)
print(f"Toplam token: {len(data)}")

n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

# --- BATCH ---
def get_batch(split):
    d  = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x  = torch.stack([d[i:i+block_size]   for i in ix])
    y  = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# --- LOSS TAHMİNİ ---
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- MİMARİ ---
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_enmbd, head_size, bias=False)
        self.query = nn.Linear(n_enmbd, head_size, bias=False)
        self.value = nn.Linear(n_enmbd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k   = self.key(x)
        q   = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj    = nn.Linear(n_enmbd, n_enmbd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_enmbd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_enmbd, 4 * n_enmbd),
            nn.GELU(),                        # ReLU yerine GELU — modern tercih
            nn.Linear(4 * n_enmbd, n_enmbd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_enmbd, n_head):
        super().__init__()
        head_size = n_enmbd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size)
        self.ffed = FeedForward(n_enmbd)
        self.ln1  = nn.LayerNorm(n_enmbd)
        self.ln2  = nn.LayerNorm(n_enmbd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffed(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_enmbd)
        self.position_embedding_table = nn.Embedding(block_size, n_enmbd)
        self.blocks = nn.Sequential(*[Block(n_enmbd, n_head) for _ in range(n_layer)])
        self.ln_f   = nn.LayerNorm(n_enmbd)
        self.lm_head = nn.Linear(n_enmbd, vocab_size)

    def forward(self, idx, target=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x       = tok_emb + pos_emb
        x       = self.blocks(x)
        x       = self.ln_f(x)
        logits  = self.lm_head(x)

        loss = None
        if target is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), target.view(B*T))

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.3):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits    = logits[:, -1, :]

            # Tekrar cezası
            if repetition_penalty > 1.0:
                for token_id in set(idx[0].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            logits = logits / temperature

            # Top-K
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-P
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove    = cum_probs > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0]  = 0
                indices_to_remove = remove.scatter(1, sorted_indices, remove)
                logits[indices_to_remove] = float('-inf')

            probs    = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # EOS gelirse dur
            if idx_next.item() == eos_token_id:
                break

            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# --- EĞİTİM ---
model     = BigramLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=1e-5)
scaler    = torch.amp.GradScaler('cuda')

print(f"\nModel parametresi: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print(f"Cihaz: {device}\n")

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    with torch.amp.autocast('cuda', dtype=torch.float16):
        logits, loss = model(xb, yb)
        loss = loss / accumulation_steps

    scaler.scale(loss).backward()

    if (iter + 1) % accumulation_steps == 0 or (iter + 1) == max_iters:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

print(f"\nEğitim tamamlandı!")

# --- KAYIT ---
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab_size': vocab_size,
    'n_enmbd':    n_enmbd,
    'n_head':     n_head,
    'n_layer':    n_layer,
    'block_size': block_size,
}, 'duman_gpt.pth')
print("Model kaydedildi: duman_gpt.pth")

# --- HIZLI TEST ---
print("\n--- Hızlı Test ---")
baslangic = "Kullanıcı: Merhaba, nasılsın?\nAsistan:"
ctx = torch.tensor([encode(baslangic)], dtype=torch.long, device=device)
model.eval()
with torch.no_grad():
    out = model.generate(ctx, max_new_tokens=100, temperature=0.7)[0].tolist()
    print(decode(out))