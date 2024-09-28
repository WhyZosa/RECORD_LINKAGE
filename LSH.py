import random
from collections import defaultdict
import numpy as np
from regex import D
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

ch=pd.read_csv("main2_normalize.csv",chunksize=20000).get_chunk()

documents=ch.drop(columns=['uid']).values.tolist()
for i in range(len(documents)):
    documents[i]=documents[i][0]+" "+documents[i][1]+" "+str(documents[i][2])+" "+documents[i][3]

#documents=[doc for doc in documents if doc.strip()] 
    #+documents[index][1]+str(documents[index][2])+documents[index][3]

# Функция для создания случайного вектора
def generate_random_vector(dimension):
    return np.random.randn(dimension)

# Функция для расчета знака скалярного произведения
def hash_function(v, random_vector):
    return 1 if np.dot(v, random_vector) > 0 else 0

# Класс для реализации Locality Sensitive Hashing
class LSH:
    def __init__(self, num_hashes, dimension):
        self.num_hashes = num_hashes
        self.hash_tables = defaultdict(list)
        self.random_vectors = [generate_random_vector(dimension) for _ in range(num_hashes)]

    def generate_hash(self, vector):
        return tuple(hash_function(vector, rv) for rv in self.random_vectors)

    def add_to_hash_table(self, vector, label):
        hash_value = self.generate_hash(vector)
        self.hash_tables[hash_value].append(label)

    def query(self, vector):
        hash_value = self.generate_hash(vector)
        return self.hash_tables.get(hash_value, [])

# Пример данных


# Векторизация данных с помощью TF-IDF
vectorizer = TfidfVectorizer(min_df=1, max_df=1.0, max_features=None)
tfidf_matrix = vectorizer.fit_transform(documents).toarray()

# Параметры LSH
num_hashes = 5  # Количество хеш-функций
dimension = tfidf_matrix.shape[1]

# Инициализация LSH
lsh = LSH(num_hashes, dimension)

# Добавление документов в LSH
for idx, vector in enumerate(tfidf_matrix):
    lsh.add_to_hash_table(vector, idx)

# Поиск похожих документов для нового текста


# Поиск кандидатов в LSH


# Проверка схожести кандидатов с новым текстом

for user in documents:
    scores=[]
    new_vector = vectorizer.transform([user]).toarray()[0]
    candidates = lsh.query(new_vector)
    for candidate in candidates:
        candidate_text = documents[candidate]
        candidate_vector = tfidf_matrix[candidate]
        similarity = cosine_similarity([new_vector], [candidate_vector])[0][0]
        if 0.99>similarity>0.3:
            print(f"Документ: '{user}',Похожий документ: '{candidate_text}', Сходство: {similarity}")
    scores.sort(reverse=True,key=lambda x:x[0])
        

    



