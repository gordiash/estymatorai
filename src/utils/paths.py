from __future__ import annotations
import re


def ToSafeName(name: str) -> str:
    """Zwraca bezpieczną nazwę do użycia w ścieżkach plików/katalogów.

    - zamienia polskie znaki na uproszczone odpowiedniki,
    - usuwa znaki niedozwolone w nazwach,
    - scala wielokrotne separatoy do jednego myślnika,
    - przycina długość do rozsądnego limitu.
    """
    if name is None:
        return "unknown"
    s = str(name)
    # Transliteration podstawowych PL znaków
    translit = {
        "ą": "a", "ć": "c", "ę": "e", "ł": "l", "ń": "n",
        "ó": "o", "ś": "s", "ż": "z", "ź": "z",
        "Ą": "A", "Ć": "C", "Ę": "E", "Ł": "L", "Ń": "N",
        "Ó": "O", "Ś": "S", "Ż": "Z", "Ź": "Z",
    }
    s = "".join(translit.get(ch, ch) for ch in s)
    # Zamiana separatorów na myślnik
    s = re.sub(r"[\s/\\]+", "-", s)
    # Usunięcie niedozwolonych znaków
    s = re.sub(r"[^A-Za-z0-9_.-]", "", s)
    # Redukcja wielokrotnych myślników/kropek/underscore
    s = re.sub(r"[-_.]{2,}", "-", s)
    # Obcięcie długości
    s = s.strip("-._")[:80]
    return s or "unknown"


