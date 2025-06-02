import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Load dataset
df = pd.read_excel('rouge_kalimat')

# Siapkan evaluator
smoothie = SmoothingFunction().method4
rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Simpan hasil evaluasi
results = []

for idx, row in df.iterrows():
    ref = row['TEXT']
    gen = row['KALIMAT']

    # Tokenisasi untuk BLEU
    ref_tokens = [ref.split()]
    gen_tokens = gen.split()

    # Hitung BLEU
    bleu = sentence_bleu(ref_tokens, gen_tokens, smoothing_function=smoothie)

    # Hitung ROUGE
    rouge_scores = rouge.score(ref, gen)
    rouge1_f1 = rouge_scores['rouge1'].fmeasure
    rouge2_f1 = rouge_scores['rouge2'].fmeasure
    rougeL_f1 = rouge_scores['rougeL'].fmeasure

    # Simpan ke list
    results.append({
        'reference': ref,
        'generated': gen,
        'BLEU': bleu,
        'ROUGE-1': rouge1_f1,
        'ROUGE-2': rouge2_f1,
        'ROUGE-L': rougeL_f1
    })

# Konversi ke DataFrame hasil
eval_df = pd.DataFrame(results)

# Simpan hasil ke file CSV
eval_df.to_csv('evaluasi_bleu_rouge.csv', index=False)

print("Evaluasi selesai. Hasil disimpan di evaluasi_bleu_rouge.csv")
