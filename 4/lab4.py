import os
import math
import re
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import pymorphy3

os.environ['NLTK_ALLOW_PROXIED_URLOPEN'] = '1'

try:
    ru_stopwords = set(stopwords.words("russian"))
except LookupError:
    nltk.download('stopwords', quiet=True)
    ru_stopwords = set(stopwords.words("russian"))

class Preprocessor:
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()
        
    def process(self, text):
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        lemmas = []
        for t in tokens:
            if t not in ru_stopwords and len(t) >= 3:
                lemmas.append(self.morph.parse(t)[0].normal_form)
        return lemmas

class NGrams:
    @staticmethod
    def get(tokens, n=1):
        if n == 1:
            return tokens
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

class CustomTFIDF:
    def __init__(self, docs_data):
        self.docs = docs_data
        self.N = len(docs_data)
        self.vocab = sorted(list(set(w for d in docs_data for w in d)))
        
    def _tf(self, idx):
        words = self.docs[idx]
        if not words: 
            return {}
        counts = Counter(words)
        return {w: c / len(words) for w, c in counts.items()}
        
    def _idf(self, word):
        df = sum(1 for d in self.docs if word in d)
        return math.log(self.N / max(df, 1))
        
    def get_vector(self, idx):
        tf = self._tf(idx)
        tfidf = {w: tf_val * self._idf(w) for w, tf_val in tf.items()}
        return np.array([tfidf.get(w, 0.0) for w in self.vocab])

    def top_k(self, idx, k=5):
        tf = self._tf(idx)
        tfidf = {w: tf_val * self._idf(w) for w, tf_val in tf.items()}
        return sorted(tfidf.items(), key=lambda x: (-x[1], x[0]))[:k]

def cosine_sim(v1, v2):
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

def test_ngrams():
    t = ["машинный", "обучение", "быть", "будущее"]
    assert NGrams.get(t, 1) == ["машинный", "обучение", "быть", "будущее"]
    assert NGrams.get(t, 2) == ["машинный обучение", "обучение быть", "быть будущее"]
    assert NGrams.get(t, 3) == ["машинный обучение быть", "обучение быть будущее"]
    print("Tests passed.\n")

test_ngrams()

text1 = """В последние годы искусственный интеллект и машинное обучение стали неотъемлемой частью технологического прогресса. Эти современные технологии проникают во все сферы нашей жизни, начиная от умных помощников в смартфонах и заканчивая сложными системами медицинской диагностики. Машинное обучение позволяет компьютерам самостоятельно находить скрытые закономерности в огромных массивах данных, не требуя явного программирования каждого шага. Особенно впечатляющих результатов достигли глубокие нейронные сети, которые сегодня успешно распознают лица, переводят тексты на лету и генерируют реалистичные изображения. Внедрение искусственного интеллекта в бизнес-процессы позволяет автоматизировать рутинные задачи и повысить эффективность предприятий. Очевидно, что искусственный интеллект — это фундаментальный сдвиг в парадигме вычислений."""

text2 = """Анализ больших данных (Big Data) тесно переплетается с развитием алгоритмов машинного обучения. Ежедневно в мире генерируются эксабайты информации. Без применения современных технологий машинного обучения обработать такие объемы данных физически невозможно. Нейронные сети и классические алгоритмы помогают компаниям выявлять скрытые тренды и прогнозировать потребительский спрос. Важным направлением стала обработка естественного языка (NLP), позволяющая машинам понимать смысл текстов. Интеграция больших данных и искусственного интеллекта в повседневную жизнь является главным конкурентным преимуществом в современной цифровой экономике."""

text3 = """Проблема глобального изменения климата требует немедленного перехода к экологически чистым и возобновляемым источникам энергии. Использование ископаемого топлива приводит к масштабным выбросам парниковых газов, что ускоряет таяние ледников. Альтернативная энергетика, включающая солнечные панели и ветрогенераторы, предлагает устойчивый путь развития. В последние годы стоимость производства солнечной и ветровой энергии значительно снизилась благодаря новым инженерным разработкам. Переход на экологически чистые источники энергии не только поможет сохранить природу для будущих поколений, но и создаст миллионы новых рабочих мест в высокотехнологичных отраслях зеленой экономики."""

text4 = """Сбалансированное питание и регулярные физические нагрузки являются фундаментом для поддержания крепкого здоровья и долголетия. В условиях современного образа жизни крайне важно осознанно подходить к своему рациону. Отказ от фастфуда и продуктов с высоким содержанием сахара снижает риск развития сердечно-сосудистых заболеваний. Спортивные тренировки не только укрепляют мышечный корсет, но и стимулируют выработку эндорфинов, помогая организму эффективно справляться со стрессом. Здоровый образ жизни требует ежедневной дисциплины, однако результаты в виде высокой энергии и хорошего самочувствия полностью оправдывают затраченные усилия."""

docs = [text1, text2, text3, text4]
labels = ["Текст 1", "Текст 2", "Текст 3", "Текст 4"]

prep = Preprocessor()
docs_lemmas = [prep.process(t) for t in docs]

unigrams = [NGrams.get(l, 1) for l in docs_lemmas]
bigrams = [NGrams.get(l, 2) for l in docs_lemmas]
mixed = [u + b for u, b in zip(unigrams, bigrams)]

tfidf_uni = CustomTFIDF(unigrams)
tfidf_bi = CustomTFIDF(bigrams)
tfidf_mixed = CustomTFIDF(mixed)

print("Текст 1 - Топ-5 униграмм:")
for k, v in tfidf_uni.top_k(0, 5):
    print(f"{k}: {v:.4f}")

print("\nТекст 1 - Топ-5 биграмм:")
for k, v in tfidf_bi.top_k(0, 5):
    print(f"{k}: {v:.4f}")

n = len(docs)
sim_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        sim_matrix[i, j] = cosine_sim(tfidf_mixed.get_vector(i), tfidf_mixed.get_vector(j))

plt.figure(figsize=(7, 5))
sns.heatmap(sim_matrix, annot=True, cmap="YlOrRd", xticklabels=labels, yticklabels=labels, vmin=0, vmax=1, fmt=".2f")
plt.tight_layout()
plt.show()