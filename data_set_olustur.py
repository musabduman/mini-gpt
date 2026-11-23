from datasets import load_dataset
import os

print("Veri seti indiriliyor...")
dataset = load_dataset("TFLai/Turkish-Alpaca")

print(f"Sütunlar: {dataset['train'].column_names}")
train_data = dataset['train']

with open('veriseti.txt', 'w', encoding='utf-8') as f:
    atilan = 0
    yazilan = 0

    for item in train_data:
        soru    = (item.get("instruction") or "").strip()
        ek_bilgi = (item.get("input")       or "").strip()
        cevap   = (item.get("output")       or "").strip()

        # Boş kayıtları atla
        if not soru or not cevap:
            atilan += 1
            continue

        if ek_bilgi:
            soru = f"{soru}\n{ek_bilgi}"

        f.write(f"Kullanıcı: {soru}\nAsistan: {cevap}<|endoftext|>\n\n")
        yazilan += 1

boyut = os.path.getsize('veriseti.txt')
print(f"\nTamamlandı!")
print(f"  Yazılan kayıt : {yazilan}")
print(f"  Atlanan kayıt : {atilan}")
print(f"  Dosya boyutu  : {boyut / (1024*1024):.2f} MB")