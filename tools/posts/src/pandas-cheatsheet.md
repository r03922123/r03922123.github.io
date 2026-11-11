---
title: pandas on one page: select, group, join, reshape and regex
summary: A working reference for the pandas operations that come up in nearly every data task: building and inspecting a frame, boolean masks, counting and ranking, missing values, groupby with agg versus transform, merges and anti-joins, melt and pivot, vectorising row logic, and string methods with a regex table. Every snippet runs on a tiny DataFrame and proves its claim with an assert.
group: Coding and data
order: 4
tags: pandas · python · data wrangling · regex · reference
---
In pandas, `df.a == 1 & df.b == 2` is parsed as a chained comparison and `df[mask]["col"] = value` writes into a copy, and most bugs come from a handful of traps like these. pandas is the tool most ML work starts in: logs arrive as a CSV or a SQL result, and a few lines of filtering, grouping and joining turn them into features or an evaluation table. The API is large, but the same twenty or so calls do most of the work. This page collects the calls and the traps. Each section runs on a frame small enough to check by eye, and the asserts show what each call returns, so the page doubles as a test that the claims hold on pandas 2.3. The asserts check results, not speed: the claims that row-wise apply and `iterrows()` are slow are not timed here, and copy-on-write in pandas 3 is described but not run. The SQL versions of many of these operations are in the [SQL cheatsheet](../sql-cheatsheet/).

## A frame, its columns and its index
A `DataFrame` is a set of named columns that share one row index. Each column is a `Series`: a typed array plus that index. Build one from a dict of columns or from a list of rows; then `dtypes`, `info()` and `describe()` tell you what you actually loaded.

```python
import re
import sys
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("error", FutureWarning)        # fail on deprecated calls, so the code below stays current

orders = pd.DataFrame({
    "order_id": [1, 2, 3, 4, 5, 6],
    "user": ["ana", "bo", "ana", "cy", "bo", "ana"],
    "city": ["Taipei", "Tokyo", "Taipei", "Osaka", None, "Taipei"],
    "amount": [30.0, 12.5, 30.0, 8.0, 50.0, np.nan],
})
rows = pd.DataFrame([(7, "dee", "Paris", 5.0)], columns=["order_id", "user", "city", "amount"])

assert orders.shape == (6, 4)
assert orders["amount"].dtype == "float64" and orders["user"].dtype == object
assert list(rows.columns) == list(orders.columns)

stats = orders["amount"].describe()                  # count, mean, std, min, quartiles, max
assert stats["count"] == 5                           # the NaN is skipped, not counted as zero
assert stats["mean"] == orders["amount"].sum() / 5   # sum() also skips NaN
assert orders.isna().sum().to_dict() == {"order_id": 0, "user": 0, "city": 1, "amount": 1}
```

`convert_dtypes()` switches columns to the nullable types (`Int64`, `string`, `boolean`), which keep integers as integers when a value is missing instead of silently turning the column into floats.

```python
ids = pd.Series([1, None, 3])
assert ids.dtype == "float64"                         # one gap turns the integers into floats
assert ids.convert_dtypes().dtype == "Int64"          # nullable integer: 1, <NA>, 3
assert ids.convert_dtypes().isna().tolist() == [False, True, False]
```

## Selecting rows and columns
Use `.loc[rows, columns]` for labels and boolean masks, `.iloc` for positions. A mask is a boolean Series aligned with the rows, and masks combine with `&`, `|` and `~`.

The classic bug is precedence. In Python `&` binds tighter than `==`, so `df.a == 1 & df.b == 2` is parsed as `df.a == (1 & df.b) == 2`, a chained comparison that raises an error at best. Wrap every comparison in parentheses, or use `query`.

```python
mask = (orders["user"] == "ana") & (orders["amount"] > 20)
assert orders.loc[mask, "order_id"].tolist() == [1, 3]
assert orders.query("user == 'ana' and amount > 20")["order_id"].tolist() == [1, 3]
try:
    orders["user"] == "ana" & orders["amount"] > 20   # parsed as a chained comparison around "ana" & amount
    raise AssertionError("expected an error")
except (TypeError, ValueError):
    pass

asia = ["Taipei", "Tokyo", "Osaka"]
assert orders.loc[orders["city"].isin(asia), "order_id"].tolist() == [1, 2, 3, 4, 6]
assert orders.loc[~orders["city"].isin(asia), "order_id"].tolist() == [5]     # None is "not in" the list

assert orders.iloc[0, 1] == "ana"                     # position: first row, second column
assert orders.loc[orders.index[-1], "user"] == "ana"  # label: last row's label, column "user"

# set_index makes a column the row labels; labels keep their type, so 5 is not "5"
by_id = orders.set_index("order_id")
assert by_id.loc[5, "user"] == "bo"
assert "5" not in by_id.index
assert by_id.reset_index().equals(orders)             # reset_index turns it back into a column
```

Write through `.loc` too. Indexing twice, as in `df[mask]["col"] = value`, assigns into a temporary copy made by the first step, so the original frame does not change. pandas 2.3 warns about it; under copy-on-write, the default from pandas 3, it never works.

```python
fixed = orders.copy()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    fixed[fixed["user"] == "cy"]["amount"] = 99.0     # chained: writes into a copy
assert fixed.loc[3, "amount"] == 8.0                  # unchanged
assert caught                                         # and pandas 2.3 warned about it

fixed.loc[fixed["user"] == "cy", "amount"] = 99.0     # one .loc call: writes into fixed
assert fixed.loc[3, "amount"] == 99.0
```

## Counting, sorting and ranking
`unique()` returns the distinct values in order of first appearance; `nunique()` counts them and skips missing values. `value_counts()` returns a Series of counts sorted from most to least frequent, indexed by value, and a mask on it is still a Series: take `.index` to get the values.

```python
assert orders["user"].unique().tolist() == ["ana", "bo", "cy"]
assert orders["city"].nunique() == 3                       # None is not a city
assert orders["city"].nunique(dropna=False) == 4

counts = orders["user"].value_counts()
assert counts.to_dict() == {"ana": 3, "bo": 2, "cy": 1}
frequent = counts[counts >= 2]
assert isinstance(frequent, pd.Series) and frequent.index.tolist() == ["ana", "bo"]

# sort_values returns a new frame; with inplace=True it returns None, so never assign that result
by_amount = orders.sort_values("amount", ascending=False)
assert by_amount["order_id"].tolist() == [5, 1, 3, 2, 4, 6]   # NaN goes last by default
assert orders.copy().sort_values("amount", inplace=True) is None
```

`rank` handles ties in several ways. `method="dense"` gives tied values the same rank and leaves no gaps (1, 1, 2), which is what a leaderboard with ties usually wants; `"min"` leaves gaps like a sports table (1, 1, 3); `"first"` breaks ties by order of appearance.

```python
scores = pd.Series([90, 85, 90, 70])
assert scores.rank(method="dense", ascending=False).tolist() == [1, 2, 1, 3]
assert scores.rank(method="min", ascending=False).tolist() == [1, 3, 1, 4]
assert scores.rank(method="first", ascending=False).tolist() == [1, 3, 2, 4]
```

## Missing values and duplicates
`isna()` and `notna()` are the tests to use: `NaN == NaN` is false, so `df[df.x == np.nan]` is always empty. `dropna()` drops rows with any missing value (narrow it with `subset=`), `fillna()` replaces them. `drop_duplicates(subset=..., keep=...)` keeps the first or last copy of each key.

```python
assert orders[orders["amount"] == np.nan].empty                     # never matches anything
assert orders.loc[orders["amount"].isna(), "order_id"].tolist() == [6]

assert len(orders.dropna()) == 4                                    # orders 5 and 6 each miss a value
assert len(orders.dropna(subset=["amount"])) == 5
assert orders["city"].fillna("unknown").tolist()[4] == "unknown"

assert orders.drop_duplicates(subset=["user"])["order_id"].tolist() == [1, 2, 4]
assert orders.drop_duplicates(subset=["user"], keep="last")["order_id"].tolist() == [4, 5, 6]
assert len(orders.drop_duplicates(subset=["user", "amount"])) == 5  # order 3 repeats order 1
```

## groupby: agg, transform, size and count
`groupby` splits rows by key, applies a function to each group and combines the results. What comes back depends on the verb:

- `agg` reduces each group to one row. The result is indexed by the key. Named aggregation, `agg(new=("column", func))`, names the output columns as it goes.
- `transform` returns one value per original row, aligned with the original index, so it can go straight into a new column. Use it to compare each row with its group, such as "rows equal to their group's maximum".
- `size()` counts rows per group; `count()` counts non-missing values per column, so the two differ when there are gaps.

By default `groupby` also drops rows whose key is missing; pass `dropna=False` to keep them as their own group.

```python
g = orders.groupby("user")
assert g["amount"].sum().to_dict() == {"ana": 60.0, "bo": 62.5, "cy": 8.0}
assert g["amount"].agg(["sum", "max"]).loc["bo"].tolist() == [62.5, 50.0]

summary = g.agg(n_orders=("order_id", "size"), spent=("amount", "sum"),
                big=("amount", lambda s: (s > 20).sum()))
assert summary.loc["ana"].tolist() == [3, 60.0, 2]

assert g.size().to_dict() == {"ana": 3, "bo": 2, "cy": 1}
assert g["amount"].count().to_dict() == {"ana": 2, "bo": 2, "cy": 1}     # ana's NaN is not counted

table = g.size().reset_index(name="count")          # back to a flat frame with a named column
assert table.columns.tolist() == ["user", "count"]

best = g["amount"].transform("max")                 # same length and index as orders
assert best.tolist() == [30.0, 50.0, 30.0, 8.0, 50.0, 30.0]
assert orders.loc[orders["amount"] == best, "order_id"].tolist() == [1, 3, 4, 5]
orders["share"] = orders["amount"] / g["amount"].transform("sum")
assert np.isclose(orders.loc[orders["user"] == "bo", "share"].sum(), 1.0)
orders = orders.drop(columns="share")

assert orders.groupby("city")["order_id"].count().sum() == 5             # the order with no city vanished
assert orders.groupby("city", dropna=False)["order_id"].count().sum() == 6
```

## Joining frames
`pd.merge(left, right, on=key, how=...)` is SQL's join. `how` is `"inner"` (the default), `"left"`, `"right"`, `"outer"` or `"cross"` (every pair, which takes no key). When the key columns have different names, use `left_on=` and `right_on=`.

Two habits prevent most join bugs. First, a key that repeats on both sides multiplies rows, so pass `validate="many_to_one"` (or `"one_to_one"`) when you expect the right side to be unique and pandas raises instead of quietly duplicating. Second, `indicator=True` adds a `_merge` column that says where each row came from, which is the cleanest anti-join: "left rows with no match".

```python
users = pd.DataFrame({"name": ["ana", "bo", "eve"], "tier": ["gold", "free", "gold"]})

inner = pd.merge(orders, users, left_on="user", right_on="name")
assert len(inner) == 5 and "cy" not in set(inner["user"])

left = pd.merge(orders, users, left_on="user", right_on="name", how="left")
assert len(left) == 6 and left.loc[left["user"] == "cy", "tier"].isna().all()

outer = pd.merge(orders, users, left_on="user", right_on="name", how="outer")
assert len(outer) == 7                                  # eve has no orders but still appears

assert len(pd.merge(users, users, how="cross")) == 9

flagged = pd.merge(users, orders, left_on="name", right_on="user", how="left", indicator=True)
no_orders = flagged.loc[flagged["_merge"] == "left_only", "name"].tolist()
assert no_orders == ["eve"]

dupes = pd.concat([users, users.iloc[[0]]])             # "ana" now appears twice on the right
assert len(pd.merge(orders, dupes, left_on="user", right_on="name")) == 8   # 5 rows became 8
try:
    pd.merge(orders, dupes, left_on="user", right_on="name", validate="many_to_one")
    raise AssertionError("expected a MergeError")
except pd.errors.MergeError:
    pass
```

## Reshaping: melt and pivot
Wide data has one column per measurement; long data has one row per measurement. `melt` goes from wide to long: `id_vars` are the columns repeated on every row, `value_vars` are the columns that become rows, and `var_name` and `value_name` name the two new columns. `pivot` goes back, and `pivot_table` does the same while aggregating any rows that share an index and column pair, which plain `pivot` refuses.

```python
wide = pd.DataFrame({"day": ["Mon", "Tue"], "A": [1, 4], "B": [2, 5], "C": [3, 6]})
long = wide.melt(id_vars="day", value_vars=["A", "B", "C"], var_name="product", value_name="sales")
assert long.shape == (6, 3)
assert long.sort_values(["day", "product"])["sales"].tolist() == [1, 2, 3, 4, 5, 6]

back = long.pivot(index="day", columns="product", values="sales").reset_index()
back.columns.name = None
assert back.equals(wide)                               # melt and pivot are inverses

repeated = pd.concat([long, long.iloc[[0]]])           # Mon/A appears twice
try:
    repeated.pivot(index="day", columns="product", values="sales")
    raise AssertionError("expected a ValueError")
except ValueError:
    pass
summed = repeated.pivot_table(index="day", columns="product", values="sales", aggfunc="sum")
assert summed.loc["Mon", "A"] == 2
```

## Row logic without loops
`df.apply(f, axis=1)` calls a Python function once per row and wraps each row in a new Series first, which is slow. Most row logic is a column expression in disguise: `np.where` for one condition, `np.select` for several, arithmetic and comparisons for the rest. When you do need to loop, `itertuples()` yields lightweight named tuples and is much faster than `iterrows()`.

```python
flag_apply = orders.apply(lambda r: "big" if r["amount"] > 20 else "small", axis=1)
flag_vec = pd.Series(np.where(orders["amount"] > 20, "big", "small"), index=orders.index)
assert flag_apply.equals(flag_vec)                     # NaN > 20 is False in both

band = np.select([orders["amount"] >= 30, orders["amount"] >= 10], ["high", "mid"], default="low")
assert band.tolist() == ["high", "mid", "high", "low", "high", "low"]

total = sum(row.amount for row in orders.itertuples() if row.user == "bo")
assert total == orders.loc[orders["user"] == "bo", "amount"].sum() == 62.5
```

## Strings and regular expressions
The `.str` accessor applies a string operation to every element and passes missing values through as missing. It covers the Python string methods (`lower`, `capitalize`, `strip`, slicing with `.str[1:]`) and the regex methods below:

| Method | Returns |
|---|---|
| `str.contains(pat, case=False, na=False)` | boolean mask: does the pattern occur anywhere? |
| `str.fullmatch(pat)` | boolean mask: does the whole string match? |
| `str.extract(pat)` | one column per capture group, from the first match |
| `str.findall(pat)` | a list of every match per row |
| `str.replace(pat, repl, regex=True)` | the string with matches replaced |
| `str.count(pat)` | number of matches per row |

`extract` needs at least one capture group and raises without one. Named groups, `(?P<name>...)`, name the output columns.

```python
s = pd.Series(["$12.50", "$7", None])
price = s.str.replace(r"^\$", "", regex=True).astype(float)        # astype, not .float()
assert price.iloc[:2].tolist() == [12.5, 7.0] and np.isnan(price.iloc[2])

names = pd.Series(["aLICE", "bob", "CHARLIE"])
assert names.str.capitalize().tolist() == ["Alice", "Bob", "Charlie"]
assert names.str.capitalize().equals(names.str[0].str.upper() + names.str[1:].str.lower())

codes = pd.Series(["abc-123", "xy-7", "nope"])
parts = codes.str.extract(r"(?P<word>\w+)-(?P<num>\d+)")
assert parts.columns.tolist() == ["word", "num"] and parts.loc[0].tolist() == ["abc", "123"]
assert parts.loc[2].isna().all()                                   # no match gives NaN, not an error
assert codes.str.extract(r"(\w+)-(?:\d+)").shape == (3, 1)          # (?:...) groups without capturing
try:
    codes.str.extract(r"\w+-\d+")
    raise AssertionError("expected a ValueError")
except ValueError:
    pass

codes = pd.Series(["ref 12-34-5678", "ord A9-12-345-678"])
assert codes.str.replace(r"\D", "", regex=True).tolist() == ["12345678", "912345678"]
assert codes.str.count(r"\d").tolist() == [8, 9]
assert pd.Series(["a1b22c333"]).str.findall(r"\d+").iloc[0] == ["1", "22", "333"]
```

The pieces of a pattern, each checked against Python's `re` below:

| Pattern | Matches |
|---|---|
| `^`, `$` | start and end of the string (`$` also matches just before a final newline) |
| `.` | any character except a newline |
| `*`, `+`, `?` | zero or more, one or more, zero or one |
| `{k}`, `{lo,up}` | exactly \(k\), or between `lo` and `up`, repetitions |
| `[abc]`, `[^abc]` | one character from the set, or not from it |
| `\d`, `\w`, `\s` | a digit, a word character (letters, digits, underscore), whitespace |
| `\D`, `\W`, `\S` | the complements |
| `(...)`, `(?:...)` | capturing group, non-capturing group |
| `a(?=b)`, `a(?!b)` | an `a` followed, or not followed, by `b` |
| `(?<=b)a`, `(?<!b)a` | an `a` preceded, or not preceded, by `b` |

Alternation, `cat|dog`, matches either side; group it, as in `(?:cat|dog)s`, to limit how far it reaches. Lookarounds check their condition without consuming it, so the match is only the `a`.

```python
assert re.findall(r"sa*", "s sa saa") == ["s", "sa", "saa"]
assert re.findall(r"sa+", "s sa saa") == ["sa", "saa"]
assert re.findall(r"sa?", "s sa saa") == ["s", "sa", "sa"]
assert re.fullmatch(r"s{2}", "ss") and not re.fullmatch(r"s{2}", "sss")
assert re.fullmatch(r"a{1,3}", "aa")
assert re.findall(r"(?:cat|dog)s", "cats dogs cat") == ["cats", "dogs"]
if sys.version_info < (3, 13):                   # checked on Python 3.11 and 3.12
    assert re.fullmatch(r"a{1, 3}", "a{1, 3}")    # a space in the braces makes them literal text
    assert not re.fullmatch(r"a{1, 3}", "aa")

assert [m.start() for m in re.finditer(r"a(?=b)", "ab ac ad")] == [0]
assert [m.start() for m in re.finditer(r"a(?!b)", "ab ac ad")] == [3, 6]
assert [m.start() for m in re.finditer(r"(?<=b)a", "ba ca da")] == [1]
assert [m.start() for m in re.finditer(r"(?<!b)a", "ba ca da")] == [4, 7]
assert re.findall(r"\w+", "user_1@x") == ["user_1", "x"]          # underscore counts as a word character

assert re.fullmatch(r"\d", "٣")                   # \d also matches non-ASCII digits; use [0-9] to mean ASCII
assert not re.fullmatch(r"[0-9]", "٣")
```

One worked example combines most of the table: [1517. Find Users With Valid E-Mails](https://leetcode.com/problems/find-users-with-valid-e-mails/) asks for addresses whose prefix starts with a letter, contains only letters, digits, `_`, `.` and `-`, and ends in `@leetcode.com`. Escape the dot in the domain, and prefer `fullmatch` to `contains` with `^...$`: in Python, `$` also matches just before a trailing newline, so an address with a stray `\n` slips through.

```python
mails = pd.Series(["mei@leetcode.com", "no-at-sign", "kai-@leetcode.com",
                   "_lin@leetcode.com", "ren.x@leetcodeXcom", "yu#7@leetcode.com",
                   "tao9@leetcode.com\n"])
prefix = r"[a-zA-Z][a-zA-Z0-9_.-]*@leetcode\.com"

anchored = mails.str.contains("^" + prefix + "$", regex=True)
strict = mails.str.fullmatch(prefix)
assert strict.tolist() == [True, False, True, False, False, False, False]
assert anchored.tolist() == strict.tolist()[:-1] + [True]           # the trailing newline got through
assert re.fullmatch(r"x@leetcode.com", "x@leetcodeXcom")            # an unescaped dot matches any character
```

## Pitfalls
- **Unparenthesised masks.** `a == 1 & b == 2` is not "a is 1 and b is 2". Parenthesise, or use `query`.
- **Chained assignment.** `df[mask]["col"] = v` writes into a copy. Use one `.loc[mask, "col"] = v`.
- **Keeping an in-place result.** `inplace=True` returns `None`. Prefer the returned frame and drop `inplace`.
- **Comparing with missing values.** `NaN` equals nothing, itself included. Use `isna()`.
- **Silent row loss and row growth.** `groupby` drops missing keys, inner joins drop unmatched rows and repeated keys multiply rows. Check `len()` before and after, use `dropna=False`, `validate=` and `indicator=True`.
- **Size versus count.** `size()` counts rows, `count()` counts non-missing values.
- **Aggregate versus transform.** `agg` gives one row per group, `transform` one value per original row.
- **Regex surprises.** Escape literal dots and dollar signs, give `extract` a capture group, use `fullmatch` for whole-string validation, and write `{lo,up}` without a space.

Several of the worked patterns follow problems from LeetCode's pandas study plans; the explanations and code here are my own, checked against pandas 2.3 and the official pandas and Python `re` documentation.
