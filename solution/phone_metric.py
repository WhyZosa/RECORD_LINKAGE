from math import sqrt
import Levenshtein

def get_score(s: str, t: str):
    if len(s) < len(t):
        s, t = t, s
    if len(t) > 9:
        return 1 - sqrt(Levenshtein.distance(s, t) / max(len(s), len(t)))
    elif len(s) > 9:
        ost = s[-len(t):]
        return (1 - sqrt(Levenshtein.distance(ost, t) / len(s))) * 0.7
    else:
        return 0.95 * (1 - sqrt(Levenshtein.distance(s, t) / max(len(s), len(t))))