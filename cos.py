from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from Levenshtein import distance as levenshtein_distance
from collections import Counter
from Levenshtein import distance as levenshtein_distance
from jellyfish import jaro_winkler_similarity, soundex
from fuzzywuzzy import fuzz
 
# # Функция для расчета нормализованного расстояния Левенштейна
# def normalized_levenshtein(str1, str2):
#     max_len = max(len(str1), len(str2))
#     if max_len == 0:  # Проверяем, если обе строки пустые
#         return 1.0
#     return 1 - levenshtein_distance(str1, str2) / max_len
 
# # Функция для расчета коэффициента Жаккара для множеств слов
# def jaccard_similarity(str1, str2):
#     set1 = set(str1.split())
#     set2 = set(str2.split())
#     intersection = len(set1.intersection(set2))
#     union = len(set1.union(set2))
#     return intersection / union if union != 0 else 1.0
 
# Функция для расчета косинусного сходства
def cosine_similarity_metric(str1, str2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([str1, str2])
    cos_sim = cosine_similarity(vectors[0], vectors[1])[0][0]
    return cos_sim
 
# # Главная функция для вычисления среднего арифметического трёх метрик
# def average_similarity(str1, str2):
#     # Считаем каждую метрику
#     lev = normalized_levenshtein(str1, str2)
#     jac = jaccard_similarity(str1, str2)
#     cos = cosine_similarity_metric(str1, str2)
    
#     # Выводим каждую метрику для проверки
#     print(f"Левенштейн: {lev}")
#     print(f"Жаккар: {jac}")
#     print(f"Косинусное: {cos}")
#     a=0.3
#     # Среднее арифметическое
#     average_score = lev
#     # (lev<a)*(cos/2+jac/2)+(lev>a)*lev
#     return average_score
 


import numpy as np
from Levenshtein import distance as levenshtein_distance

def jaccard_similarity(str1, str2):
    """
    Вычисляет коэффициент Жаккара для двух строк на уровне слов.
    
    :param str1: Первая строка.
    :param str2: Вторая строка.
    :return: Значение коэффициента Жаккара между 0 и 1.
    """
    set1 = set(str1.split())
    set2 = set(str2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 1.0

def normalized_levenshtein(str1, str2):
    """
    Вычисляет нормализованное расстояние Левенштейна для двух строк.
    
    :param str1: Первая строка.
    :param str2: Вторая строка.
    :return: Значение расстояния Левенштейна между 0 и 1.
    """
    max_len = max(len(str1), len(str2))
    if max_len == 0:  # Проверка на пустые строки
        return 1.0
    return 1 - levenshtein_distance(str1, str2) / max_len

def space_penalty(str1, str2):
    """
    Вычисляет штраф за различие в пробелах между словами.
    
    :param str1: Первая строка.
    :param str2: Вторая строка.
    :return: Значение штрафа между 0 и 1.
    """
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

def combined_string_similarity(name1, name2):
    """
    Сравнивает два ФИО по различным метрикам и возвращает комбинированное значение сходства.
    
    :param name1: Первая строка ФИО.
    :param name2: Вторая строка ФИО.
    :return: Словарь с результатами всех метрик.
    """
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
    
    # print( {
    #     'levenshtein': levenshtein,
    #     'jaro_winkler': jaro_winkler,
    #     'soundex_match': soundex_match,
    #     'fuzzy_ratio': fuzzy_ratio,
    #     'combined_score': combined_score
    # })
    return combined_score


# Примеры использования
str1 = "Иванов Пётр Сергеевич"
str2 = "Иванов Пёдр Сиргеевич"
print(f"Опечатки: {combined_string_similarity(str1, str2):.2f}")
 
str3 = "Иванов Пётр Сергеевич"
str4 = "Иванов Сергеевич"
print(f"Пропуски: {combined_string_similarity(str3, str4):.2f}")
 
str5 = "Иванов Пётр Сергеевич"
str6 = "Иванов Сергеевич Пётр"
print(f"Перестановки: {combined_string_similarity(str5, str6):.2f}")

str5 = "Иванов Петр Сиргеевич"
str6 = "Иваов Сергеевич Пётр"
print(f"Перестановки + опечатки: {combined_string_similarity(str5, str6):.2f}")

str5 = "Иванов Сергеевич оглы"
str6 = "Иванов Пётроглы Сергеевич"
print(f"Перестановки + пропуски: {combined_string_similarity(str5, str6):.2f}")

str5 = "Ианов Сиргеевич оглы"
str6 = "Иванов Сергеевич Петр Оглы "
print(f"Опечатки + пропуски: {combined_string_similarity(str5, str6):.2f}")

str5 = "Иванов Сергевич оглы"
str6 = "Ибанов Пётр углы Сергиевич"
print(f"Опечатки + пропуски + перестановки: {combined_string_similarity(str5, str6):.2f}")

str5 = "Иванав Сергевич оглы"
str6 = "Ибанович Пётруглы Сергиевич"
print(f"Опечатки + пропуски + перестановки: {combined_string_similarity(str5, str6):.2f}")


str5 = "Иванов Пётр Сергевич"
str6 = "СергеевИван Петрович"
print(f"Кейс 1: {combined_string_similarity(str5, str6):.2f}")

str5 = "Дмитриев Пётр Сергевич"
str6 = "Сергеева Людмила Хуй"
print(f"Кейс 2: {combined_string_similarity(str5, str6):.2f}")

str1 = "Иванов Пётр Сергеевич"
str2 = "Иванов ПёдрСергеевич"
print(f"Кейс: {combined_string_similarity(str1, str2):.2f}")