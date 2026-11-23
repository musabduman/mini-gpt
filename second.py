from datasets import load_dataset

# Veriyi indir
dataset = load_dataset("TFLai/Turkish-Alpaca")

print(f"Veri setindeki sütunlar: {dataset['train'].column_names}")

train_data = dataset['train']

with open('veriseti.txt', 'w', encoding='utf-8') as f:

    for item in train_data:
        # Sütun sadece 'text' olduğu için direkt onu alıyoruz        
        soru = item.get("instruction", "") or ""
        ek_bilgi = item.get("input", "") or ""
        cevap = item.get("output", "") or ""
        
        ek_bilgi=ek_bilgi.strip()
        soru=soru.strip()
        cevap=cevap.strip()

        if ek_bilgi:
            # Zaten içinde 'Kullanıcı' ve 'Asistan' etiketleri olabilir, 
            # direkt dosyaya yazıp sonuna bir boşluk bırakıyoruz.
            soru = f"{soru} ({ek_bilgi})"
             
        f.write(f"Kullanıcı: {soru}\n")
        f.write(f"Asistan: {cevap}\n\n")
# Yazma bittikten sonra kontrol edelim
import os
boyut = os.path.getsize('veriseti.txt')
print(f"İşlem tamam! input.txt boyutu: {boyut / (1024*1024):.2f} MB")