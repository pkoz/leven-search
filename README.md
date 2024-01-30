# Efficient and Flexible Searching Within Levenshtein Distance

![Coverage](https://github.com/pkoz/leven-search/blob/main/test/coverage.svg)

### Introduction

Welcome to Leven-Search, a library designed for efficient and fast searching
of words within a specified Levenshtein distance.

This library is designed with Kaggle developers and researchers in mind
as well as all others who deal with natural language processing, text analysis,
and similar domains where the closeness of strings is a pivotal aspect.

### What is Levenshtein Distance?

[Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance) measures the difference between two
sequences.
In the context of strings, it is the minimum number of single-character edits
(insertions, deletions, or substitutions) required to change one word into another.

For example, the Levenshtein distance between "table" and "marble" is 2:

1. `table` → `mable` (substitution of `t` for `m')
2. `mable` → `marble` (insertion of `r`)

### Design Goals

The library is designed with the following goals in mind:

- Efficient indexing of large datasets of words. Indexing about 40k words from the
  Brown corpus takes about 300ms on a modern laptop, and the index takes about 53MB of RAM.
- Flexibility in searching. The library allows searching for words within a specified
  Levenshtein distance also allows configuring specific edit costs for each operation.
  It allows to configure other distances, like a [keyboard distance](https://en.wiktionary.org/wiki/keyboard_distance).
- Extensibility. The library is designed to be easily extensible to support other edit distances,
  such as [Damerau-Levenshtein distance](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance)
  or [Jaro-Winkler distance](https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance).

### Performance

Example performance of the library on a Brown corpus (only words larger than 2 characters) and a modern laptop:

| Distance | Time per 1000 searches<br/>_(in seconds)_ |
|----------|-------------------------------------------|
| 0        | 0.0146                                    | 
| 1        | 0.3933                                    | 
| 1 (*)    | 0.4154                                    | 
| 2        | 7.9556                                    | 

(*) with the per-letter cost granularity

### Installation

To install the library, simply run:

```bash
pip install leven-search
```

### Usage

First, import the library:

```python
import leven_search as lev
```

Then, create a LevenSearch object:

```python
searcher = lev.LevenSearch()
```

Next, add words to the searcher:

```python
searcher.insert("hello")
searcher.insert("world")
```

Finally, search for words within a specified Levenshtein distance:

```python
searcher.find_dist("mello", 1)
```
```
Result:
	hello: ResultItem(word='hello', dist=1, updates=[m -> h])
```

### Example

The following example shows how to use the library to search for words within a Brown corpus:

```python
import nltk
import leven_search as lev

# Download the Brown corpus
nltk.download('brown')

# Create a LevenSearch object
searcher = lev.LevenSearch()

for w in nltk.corpus.brown.words():
  if len(w) > 2:
    searcher.insert(w)
```

```python
# Search for words within a Levenshtein distance
searcher.find_dist('komputer', 1)
```

```
Result:
	computer: ResultItem(word='computer', dist=1, updates=[k -> c])
```

# Search for words within a Levenshtein distance with custom costs
```python
cost = lev.GranularEditCostConfig(default_cost=2, edit_costs=[lev.EditCost('k', 'c', 0.1)])
searcher.find_dist('komputer', 2, cost)
```
```
Result:
      computer: ResultItem(word='computer', dist=0.1, updates=[k -> c])
```
----
```python
searcher.find_dist('yomputer', 2, cost)
```

```
Result:
      computer: ResultItem(word='computer', dist=2, updates=[y -> c])
```
----
```python
searcher.find_dist('yomputer', 1, cost)
```
```
Result:
      None
```
