import random
from typing import List

ALPHABET = 'abcdefghijklmnopqrstuvwxyz'


def find_positions(word_len: int, k: int) -> List[int]:
    positions = random.sample(range(word_len), k)
    positions = sorted(positions, reverse=True)
    return positions


def replace_letters(word: str, positions: List[int]) -> str:
    for position in positions:
        n_let = random.choice(ALPHABET.replace(word[position].lower(), ''))
        word = word[:position] + n_let + word[position + 1:]
    return word


def delete_letters(word: str, positions: List[int]) -> str:
    for position in positions:
        word = word[:position] + word[position + 1:]
    return word


def add_letters(word: str, positions: List[int]) -> str:
    for position in positions:
        n_let = random.choice(ALPHABET)
        word = word[:position] + n_let + word[position:]
    return word


def randomly_change_word(word: str, times: int = 1) -> str:
    positions = find_positions(len(word), times)
    for position in positions:
        op = random.choice([replace_letters, delete_letters, add_letters])
        word = op(word, [position])
    return word
