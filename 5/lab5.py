import os
import re
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymorphy3
from razdel import tokenize
from gensim.models import Word2Vec
from openTSNE import TSNE
from sklearn.manifold import trustworthiness
from huggingface_hub import hf_hub_download

os.environ['CURL_CA_BUNDLE'] = ''

morph = pymorphy3.MorphAnalyzer()

STOPWORDS = {
    "и", "в", "не", "на", "я", "что", "он", "с", "как", "а", "то", "все",
    "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще",
    "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг",
    "ли", "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас",
    "это", "весь", "который", "мочь", "свой", "очень", "просто", "такой"
}

morph_cache = {}

def preprocess(text):
    if not isinstance(text, str):
        return []
    text = re.sub(r"<[^>]+>", " ", text.lower())
    text = re.sub(r"[^а-яё\s]", " ", text)
    tokens = [t.text for t in tokenize(text)]
    res = []
    for tok in tokens:
        if len(tok) < 3 or tok in STOPWORDS:
            continue
        if tok not in morph_cache:
            p = morph.parse(tok)[0]
            lemma = p.normal_form
            pos = p.tag.POS
            morph_cache[tok] = f"{lemma}_{pos}" if (pos and lemma not in STOPWORDS) else None
        tagged = morph_cache[tok]
        if tagged:
            res.append(tagged)
    return res

path = hf_hub_download(
    repo_id="blinoff/kinopoisk",
    filename="kinopoisk.jsonl",
    repo_type="dataset"
)
df_full = pd.read_json(path, lines=True)[["content", "grade3"]].dropna()


BASE_SAMPLE_SIZE = 3000
df = df_full.sample(n=BASE_SAMPLE_SIZE, random_state=42).reset_index(drop=True)

print(f"Лемматизация и POS-разметка ({len(df)} отзывов)...")
df["tokens"] = df["content"].apply(preprocess)

fractions = [0.1, 0.5, 1.0]
models = {}

for frac in fractions:
    sub_df = df.sample(frac=frac, random_state=42)
    sentences = sub_df["tokens"].tolist()
    model = Word2Vec(
        sentences=sentences,
        vector_size=150,
        window=10,
        min_count=3,
        sg=1,
        workers=4,
        epochs=20,
        seed=42
    )
    models[frac] = model
    print(f"Корпус {int(frac*100)}% ({len(sub_df)} текстов) -> Слов в словаре: {len(model.wv.index_to_key)}")

counter = Counter()
for toks in df["tokens"]:
    counter.update(toks)

common_vocab = set(models[0.1].wv.index_to_key) & set(models[0.5].wv.index_to_key) & set(models[1.0].wv.index_to_key)
top_100 = [w for w, _ in counter.most_common() if w in common_vocab][:100]

print("\nМетрика Trustworthiness (оценка сохранения локальной структуры при сжатии в 2D):")
fig, axes = plt.subplots(1, 3, figsize=(22, 7))

for idx, frac in enumerate(fractions):
    model = models[frac]
    high_dim_vectors = np.array([model.wv[w] for w in top_100])
    
    tsne = TSNE(
        n_components=2,
        perplexity=25,
        initialization="pca",
        n_jobs=4,
        random_state=42
    )
    low_dim_emb = np.asarray(tsne.fit(high_dim_vectors))
    
    tw_score = trustworthiness(high_dim_vectors, low_dim_emb, n_neighbors=10, metric="cosine")
    print(f"Объем {int(frac*100)}% -> Trustworthiness: {tw_score:.4f}")
    
    ax = axes[idx]
    ax.scatter(low_dim_emb[:, 0], low_dim_emb[:, 1], s=25, c="steelblue", alpha=0.8)
    for i, word in enumerate(top_100):
        ax.annotate(word, (low_dim_emb[i, 0], low_dim_emb[i, 1]), fontsize=7.5, alpha=0.85)
    
    ax.set_title(f"Корпус: {int(frac*100)}% | Trustworthiness: {tw_score:.4f}")
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()