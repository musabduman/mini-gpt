# 🤖 Mini-GPT (Duman GPT)

Andrej Karpathy'nin *"Let's build GPT: from scratch, in code, spelled out"* anlatımından ve mimari prensiplerinden esinlenerek, sıfırdan PyTorch ile geliştirilmiş ve **kişisel bilgisayarda yerel olarak eğitilmiş** hafif bir Türkçe Chat LLM (Decoder-Only Transformer) modelidir.

---

## 📌 Proje Özeti

Bu proje, Büyük Dil Modellerinin (LLM) arka planındaki temel Transformer mimarisini derinlemesine anlamak, sıfırdan uygulamak ve yerel donanım üzerinde uçtan uca eğitmek amacıyla geliştirilmiştir.

* **Mimari:** Decoder-only Transformer (Multi-Head Self-Attention, Feed-Forward, LayerNorm, Residual Connections, Learned Positional Embeddings).
* **Tokenizer:** `ytu-ce-cosmos/turkish-gpt2` (Türkçe BPE Tokenizer).
* **Eğitim Verisi:** `TFLai/Turkish-Alpaca` veri seti kullanılarak instruction/chat formatında (`Kullanıcı:` / `Asistan:`) düzenlenmiştir.
* **Eğitim Ortamı:** Kişisel bilgisayar (Yerel GPU üzerinde FP16 mixed precision ve gradient accumulation desteğiyle).

---

## 📂 Dosya Yapısı

* **`chat.py`**: Model mimarisini, eğitim döngüsünü (AdamW, Cosine Annealing, GradScaler) ve test üretimini içeren ana eğitim betiği.
* **`inference.py`**: Eğitilen model ağırlıkları (`duman_gpt.pth`) üzerinden terminalde interaktif sohbet arayüzü sunan betik.
* **`first_model.py`**: Modelin ilk temel eğitim ve mimari prototipi.
* **`data_set_olustur.py` & `second.py`**: Turkish-Alpaca veri setini indirip sohbet formatına (`veriseti.txt`) dönüştüren veri hazırlama betikleri.

---

## 🚀 Hızlı Başlangıç

### 1. Gereksinimler
```bash
pip install torch transformers datasets
```

### 2. Veri Setini Hazırlama
```bash
python data_set_olustur.py
```

### 3. Eğitimi Başlatma
```bash
python chat.py
```
> Eğitim tamamlandığında model ağırlıkları yerel olarak `duman_gpt.pth` adıyla kaydedilir.

### 4. Sohbet Arayüzü (Inference)
```bash
python inference.py
```

---

## 💡 Notlar
* Büyük model ağırlıkları (`duman_gpt.pth` ~324 MB) ve üretilen metin veri setleri dosya boyutu nedeniyle repoya dahil edilmemiştir; betikler çalıştırılarak yerel ortamda üretilebilir.
* Model Andrej Karpathy'nin Transformer eğitim yaklaşımı baz alınarak Türkçe diyalog yapısına göre uyarlanmıştır.
