print("başladı")
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Eski text kısmını sil, bunu ekle:
with open('veriseti.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Veri seti yüklendi. Toplam karakter sayısı: {len(text)}")

batch_size = 8
block_size = 256
max_iters=20000
eval_interval=1000
learning_rate=3e-4
device='cuda' if torch.cuda.is_available() else 'cpu'
eval_iters=200
n_enmbd=512
n_head=8
n_layer=8
dropout=0.1

torch.manual_seed(1337)
print("Türkçe BPE Tokenizer yükleniyor (İnternet bağlantısı gerektirir)...")
     
tokenizer = AutoTokenizer.from_pretrained("ytu-ce-cosmos/turkish-gpt2")
vocab_size = len(tokenizer)
encode = lambda s: tokenizer.encode(s)
decode = lambda l: tokenizer.decode(l) 

chunk_size = 1000000 
all_tokens = []

for i in range(0, len(text), chunk_size):
    chunk = text[i:i+chunk_size]
    all_tokens.extend(encode(chunk))
    
    # Ekrana ilerleme yüzdesini yazdıralım ki dondu sanmayalım
    yuzde = int((i / len(text)) * 100)
    print(f"İlerleme: %{yuzde} tamamlandı...")

data = torch.tensor(all_tokens, dtype=torch.long)

print(f"\nTokenizasyon TAMAMLANDI! Toplam üretilen kelime sayısı: {len(data)}")

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
train_data[:block_size+1]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out={}
    m.eval()
    for split in ['train','val']:
        losses=torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y=get_batch(split)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                inputs,loss=model(X,Y)
            losses[k]=loss.item()
        out[split]=losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_enmbd, head_size, bias=False)
        self.query = nn.Linear(n_enmbd, head_size, bias=False)
        self.value = nn.Linear(n_enmbd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))

        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
         
        return out
class FeedForward(nn.Module):
    def __init__(self, n_enmbd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_enmbd, 4 * n_enmbd),
            nn.ReLU(),
            nn.Linear(4 * n_enmbd, n_enmbd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_enmbd, n_enmbd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class Block(nn.Module):
    
    def __init__(self, n_enmbd, n_head):
        super().__init__()
        head_size = n_enmbd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffed = FeedForward(n_enmbd)
        self.ln1 = nn.LayerNorm(n_enmbd)
        self.ln2 = nn.LayerNorm(n_enmbd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffed(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_enmbd)
        self.position_embedding_table = nn.Embedding(block_size,n_enmbd)
        self.blocks = nn.Sequential(*[Block(n_enmbd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_enmbd)
        self.lm_head = nn.Linear(n_enmbd, vocab_size)
        
    def forward(self, idx, target=None):
        B,T = idx.shape
        
        tok_emb = self.token_embedding_table(idx) #(B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) #(B,T,vocab_size)
        
        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.3, top_k=50, top_p=0.9):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            
            logits = logits / temperature
            # --- TOP-K SAMPLING ---
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # --- TOP-P (NUCLEUS) SAMPLING ---
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Kümülatif olasılığı top_p'yi geçenleri maskele
                sorted_indices_to_remove = cumulative_probs > top_p
                # İlk tokenin kalmasını sağla (hiçbir şey seçmeme riskine karşı)
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Maskeyi orijinal logit dizilimine geri yay
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

    def attention_katmanı(self):
        tril = torch.tril(torch.ones(block_size,block_size))
        wei = torch.zeros(block_size,block_size)
        wei = wei.masked_fill(tril == 0, float('-inf'))
        wei = F.softmax(wei,dim=-1)
        return wei

model = BigramLanguageModel(vocab_size)
m=model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=1e-5)

accumulation_steps = 4
scaler = torch.amp.GradScaler('cuda')

for iter in range(max_iters):
    if iter%eval_interval==0:
        losses=estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb, yb = get_batch('train')
    
    with torch.amp.autocast('cuda',dtype=torch.float16):
        logits, loss = m(xb, yb)
        loss = loss / accumulation_steps
    
    scaler.scale(loss).backward()

    if (iter + 1) % accumulation_steps == 0 or (iter + 1) == max_iters:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True) # VRAM'i rahatlatmak için set_to_none=True çok önemli
            #--- Öğrenme Oranı Güncelleme ---
            scheduler.step()

# Eğitim bittikten sonra loss'u eski haline getirip yazdıralım ki doğru değeri görelim
gercek_loss = loss.item() * accumulation_steps
print(f"Eğitim tamamlandı. Son Loss: {loss.item():.4f}")

# --- KAYIT İŞLEMİ BURAYA YAZILACAK ---
print("Model ağırlıkları ve parametreleri kaydediliyor...")
torch.save({
    'model_state_dict': m.state_dict(),
    'vocab_size': vocab_size,
    'n_enmbd': n_enmbd,
    'n_head': n_head,
    'n_layer': n_layer
}, 'duman_gpt.pth')
print("Kayıt başarılı! Dosya: 'duman_gpt.pth'")

# --- KÜÇÜK BİR TEST ---
print("\n--- Modelin İlk Cümleleri (Temperature=0.7) ---")
# Rastgele 0'dan başlamak yerine, modelin formatına uygun bir başlangıç verelim
baslangic_metni = "Kullanıcı: Selam\n Asistan:"
context = torch.tensor([encode(baslangic_metni)], dtype=torch.long, device=device)

# Modeli test moduna alıyoruz
m.eval()
with torch.no_grad():
    generated = m.generate(context, max_new_tokens=100, temperature=0.7)[0].tolist()
    print(decode(generated))