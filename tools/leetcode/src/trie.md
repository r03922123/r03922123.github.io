---
title: Tries: prefix trees for word search and autocomplete
summary: A tree keyed by characters, where every path from the root spells a prefix. Insert and search in O(word length), then the two places tries pay off — searching a grid for many words at once, and top-k autocomplete.
group: Trees and range queries
order: 2
tags: trie · strings · DFS
---
A hash set answers "is this word present?" in \(O(L)\). A trie answers that and also "does **any** word start with this prefix?", which a hash set cannot do without storing every prefix. Each node is one character position; children are keyed by the next character; a flag marks where a word ends.

## The structure
[208. Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/):

```python
class Trie:
    def __init__(self):
        self.children, self.word = {}, None     # word: the full word ending here, if any

    def insert(self, word):
        node = self
        for ch in word:
            node = node.children.setdefault(ch, Trie())
        node.word = word

    def _walk(self, prefix):
        node = self
        for ch in prefix:
            node = node.children.get(ch)
            if node is None:
                return None
        return node

    def search(self, word):
        node = self._walk(word)
        return node is not None and node.word is not None

    def starts_with(self, prefix):
        return self._walk(prefix) is not None


t = Trie()
t.insert("apple")
assert t.search("apple") and not t.search("app") and t.starts_with("app")
t.insert("app")
assert t.search("app")
```

Every operation is \(O(L)\) for a word of length \(L\), independent of how many words are stored. Storing the whole word at its end node, instead of a boolean, saves rebuilding it later.

## Searching a grid for many words
[212. Word Search II](https://leetcode.com/problems/word-search-ii/): find which of many words can be traced through adjacent cells of a letter grid. Searching each word separately repeats the same grid walks. Instead, put all words in one trie and DFS the grid **once**, following the trie as you go: a path stops as soon as it is not a prefix of any word.

```python
def find_words(board, words):
    root = Trie()
    for w in words:
        root.insert(w)
    rows, cols, found = len(board), len(board[0]), []

    def dfs(r, c, parent):
        ch = board[r][c]
        node = parent.children.get(ch)
        if node is None:
            return
        if node.word:
            found.append(node.word)
            node.word = None                    # report each word once
        board[r][c] = "#"                       # mark visited
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != "#":
                dfs(nr, nc, node)
        board[r][c] = ch
        if not node.children and node.word is None:
            del parent.children[ch]             # prune exhausted branches

    for r in range(rows):
        for c in range(cols):
            dfs(r, c, root)
    return sorted(found)


board = [list("oaan"), list("etae"), list("ihkr"), list("iflv")]
assert find_words(board, ["oath", "pea", "eat", "rain"]) == ["eat", "oath"]
assert find_words([list("ab"), list("cd")], ["abcb"]) == []
```

Pruning branches whose words were all found keeps later searches from walking dead paths.

## Autocomplete
For [642. Design Search Autocomplete System](https://leetcode.com/problems/design-search-autocomplete-system/)-style problems, walk to the prefix node, then rank what lies below. With a small number of completions per prefix, a DFS that collects them is enough; for large dictionaries, cache the top-k at each node during insertion instead, trading memory for \(O(L)\) queries.

```python
def complete(trie, prefix, counts, k=3):
    node = trie._walk(prefix)
    if node is None:
        return []
    words, stack = [], [node]
    while stack:
        n = stack.pop()
        if n.word:
            words.append(n.word)
        stack.extend(n.children.values())
    return sorted(words, key=lambda w: (-counts[w], w))[:k]   # hottest first, ties alphabetical


counts = {"i love you": 5, "island": 3, "iroman": 2, "i love leetcode": 2}
t = Trie()
for sentence in counts:
    t.insert(sentence)
assert complete(t, "i", counts) == ["i love you", "island", "i love leetcode"]
assert complete(t, "i ", counts) == ["i love you", "i love leetcode"]
assert complete(t, "x", counts) == []
```

## Variants
- **Bitwise tries** store numbers bit by bit from the highest, and answer "which stored number has the maximum XOR with x?" by greedily taking the opposite bit at each level ([421](https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/)).
- **Folder trees** are tries over path components; serializing each subtree finds duplicates ([1948](https://leetcode.com/problems/delete-duplicate-folders-in-system/)).

## Practice
- [211. Design Add and Search Words Data Structure](https://leetcode.com/problems/design-add-and-search-words-data-structure/)
- [421. Maximum XOR of Two Numbers in an Array](https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/)
- [642. Design Search Autocomplete System](https://leetcode.com/problems/design-search-autocomplete-system/)
- [1948. Delete Duplicate Folders in System](https://leetcode.com/problems/delete-duplicate-folders-in-system/)
