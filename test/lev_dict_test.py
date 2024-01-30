import random
import time
import unittest
from contextlib import contextmanager
from typing import List

import nltk
import pytest

from leven_search import LevenSearch, GranularEditCostConfig, EditCost, EditOp
from helpers import randomly_change_word


@contextmanager
def measure_time(name, n_iter):
    k_n_iter = n_iter / 1000
    start = time.time()
    yield
    end = time.time()
    print(f"{name}: Time: {(end - start):.4f},: {(end - start) / k_n_iter:.4f} / 1000 iter")


def letter_between(word: str, letter: str, min_occ: int, max_occ: int) -> bool:
    c = word.count(letter)
    return min_occ <= c <= max_occ


class LevenSearchTestCase(unittest.TestCase):
    lev_search: LevenSearch = None
    brown: List[str] = None
    long_brown: List[str] = None

    @classmethod
    def setUpClass(cls):
        cls.brown = [w for w in set([w for w in nltk.corpus.brown.words() if len(w) > 2])]
        cls.long_brown = [w for w in cls.brown if len(w) > 5]
        cls.lev_search = LevenSearch()
        with measure_time("indexing", len(cls.brown)):
            for word in cls.brown:
                cls.lev_search.insert(word)

    def test_find(self):
        random_words = random.choices(LevenSearchTestCase.brown, k=100)
        with measure_time("just find", len(random_words)):
            for word in random_words:
                self.assertEqual(LevenSearchTestCase.lev_search.find(word), True)

    def test_cannot_find(self):
        result = LevenSearchTestCase.lev_search.find('engineeering')
        self.assertFalse(result)

    def test_find_dist_exact_dist_2(self):
        random_words = random.choices(LevenSearchTestCase.brown, k=20)
        for word in random_words:
            res = LevenSearchTestCase.lev_search.find_dist(word, max_distance=2)
            self.assertTrue(res.is_in(word))
            self.assertTrue(res.get_distance(word) == 0)

    def test_find_dist_dist_1(self):
        words_orig = (['the', 'halfway', 'hours', 'that', 'Government'] +
                      ['which', 'hopes', 'work', 'However', 'criss-crossing'] +
                      ['engineering', 'way', 'nature', 'orders', 'happen'])

        words_distorted = (['yhe', 'halvway', 'hocrs', 'thit', 'Governgent'] +  # updated
                           ['whih', 'opes', 'wor', 'Howeve', 'crisscrossing'] +  # deleted
                           ['engineeering', 'wafy', 'naturex', 'corders', 'haappen'])  # added

        words_too_far_orig = ['did', 'usually', 'diet', 'Communists', 'that']
        words_too_far = ['didid', 'xusualy', 'sdiets', 'Communisttss', 'thatat']
        time1 = time.time()
        for word_ok, words_distorted in zip(words_orig, words_distorted):
            res = LevenSearchTestCase.lev_search.find_dist(words_distorted, max_distance=1)
            self.assertTrue(res.is_in(word_ok))
            self.assertFalse(res.is_in(words_distorted))
            self.assertTrue(res.get_distance(word_ok) == 1)
        for word_ok, words_distorted in zip(words_too_far_orig, words_too_far):
            res = LevenSearchTestCase.lev_search.find_dist(words_distorted, max_distance=1)
            self.assertFalse(res.is_in(word_ok))
            self.assertFalse(res.is_in(words_distorted))
            self.assertTrue(res.get_distance(word_ok) is None)
        time2 = time.time()
        print(f"find time: {time2 - time1:.4f}")

    def test_find_large_dist_0(self):
        random_words = random.choices(LevenSearchTestCase.long_brown, k=2000)
        with measure_time("find_large_dist_0", len(random_words)):
            for word in random_words:
                res = LevenSearchTestCase.lev_search.find_dist(word, max_distance=0)
                self.assertTrue(res.is_in(word))
                self.assertTrue(res.get_distance(word) == 0)

    def test_find_large_dist_1(self):
        random_words = random.choices(LevenSearchTestCase.long_brown, k=2000)
        with measure_time("find_large_dist_1", len(random_words)):
            for word in random_words:
                alt_word = randomly_change_word(word, 1)
                res = LevenSearchTestCase.lev_search.find_dist(alt_word, max_distance=1)
                self.assertTrue(res.is_in(word))
                self.assertTrue(res.get_distance(word) > 0)

    def test_find_large_dist_2(self):
        random_words = random.choices(LevenSearchTestCase.long_brown, k=2000)
        with measure_time("find_large_dist_2", len(random_words)):
            for word in random_words:
                alt_word = randomly_change_word(word, 2)
                res = LevenSearchTestCase.lev_search.find_dist(alt_word, max_distance=2)
                self.assertTrue(res.is_in(word))

    def test_granular_edit_cost(self):
        random_words = random.choices(LevenSearchTestCase.long_brown, k=2000)
        words_with_a = list(filter(lambda w: letter_between(w, letter='a', min_occ=1, max_occ=5), random_words))
        with measure_time("granular_distance", len(words_with_a)):
            for word in words_with_a:
                alt_word = word.replace('a', 'x')
                edit_cost = GranularEditCostConfig(default_cost=10, edit_costs=[EditCost('x', 'a', 1)])
                res = LevenSearchTestCase.lev_search.find_dist(alt_word, max_distance=10, edit_cost_config=edit_cost)
                self.assertTrue(res.is_in(word))
        # Test word with distance > 1
        test_word = "pxnorxmxsy"
        edit_cost = GranularEditCostConfig(default_cost=10, edit_costs=[EditCost('x', 'a', 1)])
        res = LevenSearchTestCase.lev_search.find_dist(test_word, max_distance=20, edit_cost_config=edit_cost)
        self.assertTrue(res.is_in("panoramas"))
        self.assertEqual(13, res.get_distance("panoramas"))

    def test_edit_cost_only_default(self):
        test_word = "panoramasq"
        res = LevenSearchTestCase.lev_search.find_dist(test_word, max_distance=10, edit_cost_config=10)
        self.assertTrue(res.is_in("panoramas"))
        self.assertEqual(10, res.get_distance("panoramas"))

    def test_use_result_edit_as_input(self):
        test_word = "panoramasq"
        res = LevenSearchTestCase.lev_search.find_dist(test_word, max_distance=10, edit_cost_config=10)

        self.assertTrue(res.is_in("panoramas"))
        self.assertEqual(10, res.get_distance("panoramas"))

        result_panoramas = res.get_result('panoramas')
        self.assertIsNotNone(result_panoramas)
        edits = result_panoramas.updates
        self.assertEqual(1, len(edits))
        edit_with_cost = EditCost.from_edit(edits[0], 2)
        granular_cost = GranularEditCostConfig(default_cost=10, edit_costs=[edit_with_cost])
        res_2 = LevenSearchTestCase.lev_search.find_dist(test_word, max_distance=3, edit_cost_config=granular_cost)

        self.assertTrue(res_2.is_in("panoramas"))
        self.assertEqual(2, res_2.get_distance("panoramas"))

    def test_granular_edit_cost_object_input(self):
        test_word = "pxnorxmxsy"
        edit_cost = GranularEditCostConfig(default_cost=20, edit_costs=[EditCost('x', 'a', 1)])
        res = LevenSearchTestCase.lev_search.find_dist(test_word, max_distance=30, edit_cost_config=edit_cost)
        self.assertTrue(res.is_in("panoramas"))
        self.assertEqual(23, res.get_distance("panoramas"))

    def test_granular_float_edit_cost_object_input(self):
        test_word = "pxnorxmxsy"
        edit_cost = GranularEditCostConfig(default_cost=1.8, edit_costs=[EditCost('x', 'a', 0.1)])
        res = LevenSearchTestCase.lev_search.find_dist(test_word, max_distance=3.1, edit_cost_config=edit_cost)
        self.assertTrue(res.is_in("panoramas"))
        self.assertEqual(2.1, pytest.approx(res.get_distance("panoramas")))

    def test_granular_edit_cost_for_delete(self):
        test_word = "mathematiciaxn"
        edit_cost = GranularEditCostConfig(default_cost=5, edit_costs=[EditCost(EditOp.DELETE, 'x', 1)])
        res = LevenSearchTestCase.lev_search.find_dist(test_word, max_distance=4, edit_cost_config=edit_cost)
        self.assertTrue(res.is_in("mathematician"))
        self.assertEqual(1, res.get_distance("mathematician"))

    def test_granular_edit_cost_for_delete_and_add(self):
        test_word = "mathematciaxn"
        edit_cost = GranularEditCostConfig(default_cost=5, edit_costs=[EditCost(EditOp.DELETE, 'x', 1),
                                                                       EditCost(EditOp.ADD, 'i', 1)])
        res = LevenSearchTestCase.lev_search.find_dist(test_word, max_distance=4, edit_cost_config=edit_cost)
        self.assertTrue(res.is_in("mathematician"))
        self.assertEqual(2, res.get_distance("mathematician"))

    def test_granular_edit_cost_for_delete_and_add_input_as_list(self):
        test_word = "mathematciaxn"
        edit_cost = [EditCost(EditOp.DELETE, 'x', 1), EditCost(EditOp.ADD, 'i', 1)]
        res = LevenSearchTestCase.lev_search.find_dist(test_word, max_distance=2, edit_cost_config=edit_cost)
        self.assertTrue(res.is_in("mathematician"))
        self.assertEqual(2, res.get_distance("mathematician"))

    def test_granular_edit_cost_for_delete_and_add_input_as_list_with_default(self):
        test_word = "mathematciaxn"
        edit_cost = [EditCost(EditOp.DELETE, 'x', 1), EditCost(EditOp.ADD, 'i', 1)]
        res = LevenSearchTestCase.lev_search.find_dist(test_word,
                                                       max_distance=4,
                                                       edit_cost_config=edit_cost,
                                                       default_cost=2)
        self.assertTrue(res.is_in("mathematician"))
        self.assertEqual(2, res.get_distance("mathematician"))

    def test_granular_edit_unknown_type(self):
        with self.assertRaises(Exception) as context:
            LevenSearchTestCase.lev_search.find_dist("something", max_distance=2, edit_cost_config="whatever")

        self.assertTrue("edit_cost must be a list or EditCost object" in str(context.exception))

    def test_granular_edit_cost_same_letters(self):
        test_word = "mathematician"
        edit_cost = GranularEditCostConfig(default_cost=5, edit_costs=[EditCost('t', 't', 1_000)])
        res = LevenSearchTestCase.lev_search.find_dist(test_word, max_distance=5, edit_cost_config=edit_cost)
        self.assertTrue(res.is_in("mathematician"))
        self.assertEqual(0, res.get_distance("mathematician"))
        # Check string representation of the result
        expected_repr = "Result:\n\tmathematician: ResultItem(word='mathematician', dist=0, updates=[])"
        self.assertEqual(expected_repr, res.__repr__())

    def test_repr_of_granular_edit_cost(self):
        granular_edit_cost = GranularEditCostConfig(default_cost=25, edit_costs=[EditCost(EditOp.DELETE, 'x', 1),
                                                                                 EditCost(EditOp.ADD, 'i', 2),
                                                                                 EditCost('w', 't', 5), ])
        expected_repr = "\n".join(['GranularEditCost:',
                                   '\tdefault_cost: 25',
                                   '\tletter cost: ',
                                   '\t\t[-] x : 1',
                                   '\t\t[+] i : 2',
                                   '\t\tw -> t : 5'])

        self.assertEqual(expected_repr, granular_edit_cost.__repr__())

    def test_repr_of_edit_cost(self):
        edit_cost = EditCost(EditOp.ADD, 'i', 2)
        expected_repr = "[+] i : 2"
        self.assertEqual(expected_repr, edit_cost.__repr__())
