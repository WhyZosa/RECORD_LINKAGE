from jellyfish import jaro_winkler_similarity, soundex
from fuzzywuzzy import fuzz
from Levenshtein import distance as levenshtein_distance

def jaccard_similarity(str1, str2):

    set1 = set(str1.split())
    set2 = set(str2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 1.0

def normalized_levenshtein(str1, str2):

    max_len = max(len(str1), len(str2))
    if max_len == 0:  # Проверка на пустые строки
        return 1.0
    return 1 - levenshtein_distance(str1, str2) / max_len

def space_penalty(str1, str2):
    words1 = str1.split()
    words2 = str2.split()
    
    # Сравниваем количество пробелов между словами
    gap_diff = abs(len(words1) - 1 - (len(words2) - 1))
    
    # Дополнительный штраф за каждое несовпадение в пробелах
    penalty = 0
    for w1, w2 in zip(words1, words2):
        if w1 != w2:
            penalty += 1
            
    # Нормализуем штраф
    max_len = max(len(words1), len(words2))
    return 1 - (gap_diff + penalty) / max_len

def get_score(name1: str, name2: str):
    # Нормализуем строки (приведение к нижнему регистру и удаление лишних пробелов)
    name1 = name1.lower().strip()
    name2 = name2.lower().strip()
    
    # Левенштейн (нормализованное расстояние)
    max_len = max(len(name1), len(name2))
    levenshtein = 1 - levenshtein_distance(name1, name2) / max_len if max_len != 0 else 1.0
    
    # Джаро-Винклер
    jaro_winkler = jaro_winkler_similarity(name1, name2)
    
    # Фонетический алгоритм Soundex
    soundex_match = soundex(name1) == soundex(name2)
    
    # Soft Token Sort Ratio из fuzzywuzzy
    fuzzy_ratio = fuzz.token_sort_ratio(name1, name2) / 100
    
    # Комбинированное значение
    combined_score = 0.3 * levenshtein + 0.3 * jaro_winkler + 0.1 * soundex_match + 0.3 * fuzzy_ratio
    
    return combined_score