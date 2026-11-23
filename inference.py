import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer

# --- AYARLAR (eğitimle birebir aynı olmalı) ---
device     = 'cuda' if torch.cuda.is_available() else 'cpu'
n_enmbd    = 768
n_head     = 12
n_layer    = 12
block_size = 512
dropout    = 0.1

# --- TOKENİZER ---
print("Tokenizer yükleniyor...")
tokenizer    = AutoTokenizer.from_pretrained("ytu-ce-cosmos/turkish-gpt2")
vocab_size   = len(tokenizer)
eos_token_id = tokenizer.eos_token_id

encode = lambda s: tokenizer.encode(s)
decode = lambda l: tokenizer.decode(l)

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
            nn.GELU(),
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
        self.blocks  = nn.Sequential(*[Block(n_enmbd, n_head) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_enmbd)
        self.lm_head = nn.Linear(n_enmbd, vocab_size)

    def forward(self, idx, target=None):
        B, T    = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x       = tok_emb + pos_emb
        x       = self.blocks(x)
        x       = self.ln_f(x)
        logits  = self.lm_head(x)
        return logits, None

    def generate(self, idx, max_new_tokens, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.3):
        for _ in range(max_new_tokens):
            idx_cond  = idx[:, -block_size:]
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

            # EOS gelirse dur — model cevabı bitirdi
            if idx_next.item() == eos_token_id:
                break

            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# --- MODELİ YÜKLE ---
print("Model yükleniyor...")
model = BigramLanguageModel(vocab_size)

try:
    checkpoint = torch.load('duman_gpt.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model başarıyla yüklendi!\n")
except Exception as e:
    print(f"Hata: {e}")
    print("'duman_gpt.pth' dosyasının bu klasörde olduğundan emin ol.")
    exit()

model.to(device)
model.eval()

print("=" * 50)
print("  DUMAN GPT — Çıkmak için 'q' yazın")
print("=" * 50)

# --- SOHBET DÖNGÜSÜ ---
while True:
    user_input = input("\nSiz: ").strip()

    if user_input.lower() == 'q':
        print("Görüşmek üzere!")
        break

    if not user_input:
        continue

    prompt  = f"Kullanıcı: {user_input}\nAsistan:"
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

    with torch.no_grad():
        out       = model.generate(context, max_new_tokens=200,
                                   temperature=0.7, top_k=50,
                                   top_p=0.9, repetition_penalty=1.3)[0].tolist()
        tam_cevap = decode(out)

    # Asistan cevabını temiz ayıkla
    try:
        sadece_cevap = tam_cevap.split("Asistan:")[-1].split("Kullanıcı:")[0].strip()
        # EOS artık kodda kırpıldığı için bu genellikle temiz gelir
        sadece_cevap = sadece_cevap.replace("<|endoftext|>", "").strip()
        print(f"\nDuman GPT: {sadece_cevap}")
    except Exception:
        print(f"\nDuman GPT (Ham): {tam_cevap}")