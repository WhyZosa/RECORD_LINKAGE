import Levenshtein
from name_metric import jaccard_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer()

def get_score(s: str, t:str):
    m = vectorizer.fit_transform([s, t])
    return jaccard_similarity(s, t) * 0.4 + 0.3 * cosine_similarity(m[0:1], m[1:2])[0][0] + 0.3 * Levenshtein.distance(s, t)