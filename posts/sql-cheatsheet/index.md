# SQL on one page: evaluation order, NULLs, joins, window functions and the classic patterns

SQL looks like English, which is exactly why it misleads: you write `SELECT` first, but the database runs it almost last. Most SQL bugs come from a handful of places where the written order, NULLs or a default setting do something you did not expect. This page collects them, together with the window functions and query shapes that come up again and again in analytics work.

Nothing here is taken on trust. Each query is shown as plain SQL, then run against an in-memory SQLite database from Python, with asserts on the result. The queries are written in the large subset that PostgreSQL and SQLite share. Where the two differ, the PostgreSQL form is given in a block whose first line says **PostgreSQL only**, and the phrasebook near the end lines up the date, string and cast functions side by side.

## The tables
Three small tables carry most of the examples. `employees` has a self-reference (`manager_id`) and one missing salary:

| id | name | dept | salary | manager_id |
|---|---|---|---|---|
| 1 | Ana | eng | 120 | NULL |
| 2 | Ben | eng | 100 | 1 |
| 3 | Cy | eng | 100 | 1 |
| 4 | Dee | sales | 90 | 1 |
| 5 | Eve | sales | 70 | 4 |
| 6 | Fay | ops | NULL | 4 |
| 7 | Gus | ops | 60 | 4 |

`orders` holds seven orders from two shops across three months, with two orders on the same day so ties show up. `logins` holds one row per login.

```python
import sqlite3

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

db = sqlite3.connect(":memory:")
db.executescript("""
CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, dept TEXT, salary INTEGER, manager_id INTEGER);
INSERT INTO employees VALUES
  (1, 'Ana', 'eng', 120, NULL), (2, 'Ben', 'eng', 100, 1), (3, 'Cy', 'eng', 100, 1),
  (4, 'Dee', 'sales', 90, 1), (5, 'Eve', 'sales', 70, 4), (6, 'Fay', 'ops', NULL, 4),
  (7, 'Gus', 'ops', 60, 4);

CREATE TABLE orders (id INTEGER PRIMARY KEY, shop TEXT, day TEXT, amount INTEGER);
INSERT INTO orders VALUES
  (1, 'A', '2026-01-05', 10), (2, 'A', '2026-01-20', 20), (3, 'A', '2026-01-20', 5),
  (4, 'A', '2026-02-03', 30), (5, 'A', '2026-03-11', 15),
  (6, 'B', '2026-01-07', 7), (7, 'B', '2026-02-14', 3);

CREATE TABLE logins (user_id TEXT, day TEXT);
INSERT INTO logins VALUES
  ('u1', '2026-01-01'), ('u1', '2026-01-02'), ('u1', '2026-01-02'), ('u1', '2026-01-03'),
  ('u1', '2026-01-05'), ('u1', '2026-01-06'), ('u2', '2026-01-02');
""")


def q(sql, params=()):
    return db.execute(sql, params).fetchall()


assert sqlite3.sqlite_version_info >= (3, 44)   # window functions, FILTER, NULLS LAST, IS NOT DISTINCT FROM, concat()
```

## The order a query is evaluated in
A `SELECT` statement is written in one order and evaluated, logically, in another:

```plaintext
1. FROM, JOIN          build the rows
2. WHERE               keep some rows
3. GROUP BY            form groups
4. aggregates, HAVING  compute COUNT, SUM, ... per group, then keep some groups
5. window functions    computed over the rows that survived steps 1 to 4
6. SELECT list         compute output columns and name them
7. DISTINCT
8. UNION, INTERSECT, EXCEPT
9. ORDER BY            the only clause that can always use SELECT aliases
10. LIMIT, OFFSET      (standard SQL spells it FETCH FIRST n ROWS ONLY)
```

The optimizer is free to do the work in any order that gives the same answer; this list says what each clause can see. Four rules follow from it:

- `WHERE` runs before grouping, so it cannot filter on an aggregate. That is what `HAVING` is for, and in the text `HAVING` must come after `GROUP BY`.
- Window functions run after `WHERE` and `HAVING`, so neither clause can filter on a window result. Compute the window in a subquery or CTE and filter outside it.
- Because windows run after aggregation, a window can wrap an aggregate: `SUM(COUNT(*)) OVER ()` is the total of the group counts.
- An alias defined in `SELECT` does not exist yet in `WHERE`. PostgreSQL rejects `WHERE alias > 0`. SQLite quietly accepts it, which is a classic way for a query to pass locally and fail in production.

The query below gives each department's share of the salaried employees. Read it in evaluation order: `WHERE` drops Fay, grouping gives eng 3, sales 2 and ops 1, `HAVING` drops ops, and only then does the window add up the remaining counts.

```sql
SELECT dept,
       COUNT(*) AS n,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_kept,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM employees WHERE salary IS NOT NULL), 1) AS pct_of_all
FROM employees
WHERE salary IS NOT NULL
GROUP BY dept
HAVING COUNT(*) >= 2
ORDER BY n DESC;
```

The two percentage columns differ for exactly that reason. The window total sees only the groups that survived `HAVING` (5 people), while the scalar subquery counts everyone with a salary (6 people). Decide which denominator you mean before writing either.

```python
share = """
SELECT dept,
       COUNT(*) AS n,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_kept,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM employees WHERE salary IS NOT NULL), 1) AS pct_of_all
FROM employees
WHERE salary IS NOT NULL
GROUP BY dept
HAVING COUNT(*) >= 2
ORDER BY n DESC;
"""
assert q(share) == [("eng", 3, 60.0, 50.0), ("sales", 2, 40.0, 33.3)]

# a window result cannot be filtered in WHERE: it does not exist yet
try:
    q("SELECT name FROM employees WHERE RANK() OVER (ORDER BY salary DESC) = 1")
    raise AssertionError("expected an error")
except sqlite3.OperationalError as e:
    assert "misuse of window function" in str(e)

# SQLite lets WHERE see a SELECT alias; PostgreSQL would say the column does not exist
assert q("SELECT name, salary * 2 AS doubled FROM employees WHERE doubled > 200") == [("Ana", 240)]
```

## NULLs: counting, comparing and NOT IN
NULL means "unknown", and almost every operator that touches an unknown returns unknown. Five consequences cover most surprises:

- `COUNT(*)` counts rows, whatever they contain. `COUNT(col)` counts the rows where `col` is not NULL, and `COUNT(DISTINCT col)` counts distinct non-NULL values.
- `AVG`, `SUM`, `MIN` and `MAX` skip NULLs. The average salary is 540 / 6 = 90, not 540 / 7. If a missing value should count as zero, say so with `COALESCE(salary, 0)`.
- `x = NULL` is never true, not even when `x` is NULL. Test with `IS NULL`, or compare two nullable columns with `IS NOT DISTINCT FROM`, which treats two NULLs as equal.
- `x NOT IN (subquery)` returns no rows at all if the subquery yields a single NULL, because `x <> NULL` is unknown. Use `NOT EXISTS` instead (next section).
- PostgreSQL sorts NULL as larger than every value, so it comes last in ascending order and **first** in `ORDER BY ... DESC`. SQLite does the opposite. Write `NULLS LAST` or `NULLS FIRST` whenever NULLs can appear in a sort key.

```sql
SELECT COUNT(*), COUNT(salary), COUNT(DISTINCT salary), AVG(salary), AVG(COALESCE(salary, 0))
FROM employees;

-- employees who manage nobody: returns nothing, because manager_id contains a NULL
SELECT name FROM employees
WHERE id NOT IN (SELECT manager_id FROM employees);
```

```python
n_rows, n_salary, n_distinct, avg, avg_zero = q("""
SELECT COUNT(*), COUNT(salary), COUNT(DISTINCT salary), AVG(salary), AVG(COALESCE(salary, 0))
FROM employees;
""")[0]
assert (n_rows, n_salary, n_distinct, avg) == (7, 6, 5, 90.0)
assert abs(avg_zero - 540 / 7) < 1e-12

assert q("""
SELECT name FROM employees
WHERE id NOT IN (SELECT manager_id FROM employees);
""") == []
assert q("SELECT COUNT(*) FROM employees WHERE manager_id = NULL") == [(0,)]
assert q("SELECT COUNT(*) FROM employees WHERE manager_id IS NULL") == [(1,)]
assert q("SELECT NULL IS NOT DISTINCT FROM NULL") == [(1,)]

# NULL placement in a descending sort, made explicit
assert q("SELECT name FROM employees ORDER BY salary DESC NULLS FIRST LIMIT 1") == [("Fay",)]
assert q("SELECT name FROM employees ORDER BY salary DESC NULLS LAST LIMIT 1") == [("Ana",)]
```

## Arithmetic, casts and CASE
An integer divided by an integer is an integer in both PostgreSQL and SQLite: `7 / 2` is 3. Make one side non-integer first, either with `1.0 * x / y` or `CAST(x AS REAL) / y`. Guard against division by zero with `NULLIF`: `x / NULLIF(y, 0)` returns NULL instead of an error.

PostgreSQL also has the shorthand cast `x::numeric`, and one rule that catches people: two-argument `ROUND(x, n)` exists only for `numeric`, so rounding a `double precision` result, such as anything from `SQRT`, needs a cast first.

```sql
-- PostgreSQL only
SELECT ROUND(SQRT(2)::numeric, 2);   -- 1.41
SELECT ROUND(SQRT(2), 2);            -- error: function round(double precision, integer) does not exist
```

`CASE` is SQL's if-else, and inside an aggregate it turns rows into columns, often called conditional aggregation or a pivot. The standard `FILTER` clause says the same thing more directly and works in both databases.

```sql
SELECT dept,
       SUM(CASE WHEN salary >= 90 THEN 1 ELSE 0 END) AS senior,
       COUNT(*) FILTER (WHERE salary < 90) AS junior,
       CASE WHEN MAX(salary) >= 100 THEN 'high' ELSE 'normal' END AS band
FROM employees
GROUP BY dept
ORDER BY dept;
```

```python
assert q("SELECT 7 / 2, 1.0 * 7 / 2, CAST(7 AS REAL) / 2, 7 / NULLIF(0, 0)") == [(3, 3.5, 3.5, None)]

assert q("""
SELECT dept,
       SUM(CASE WHEN salary >= 90 THEN 1 ELSE 0 END) AS senior,
       COUNT(*) FILTER (WHERE salary < 90) AS junior,
       CASE WHEN MAX(salary) >= 100 THEN 'high' ELSE 'normal' END AS band
FROM employees
GROUP BY dept
ORDER BY dept;
""") == [("eng", 3, 0, "high"), ("ops", 0, 1, "normal"), ("sales", 1, 1, "normal")]
```

Fay's NULL salary falls through every `CASE` branch to `ELSE 0`, and fails the `FILTER` test too, so ops counts one junior and no seniors.

## Joins, self joins and anti-joins
`INNER JOIN` keeps matching pairs, `LEFT JOIN` also keeps unmatched left rows with NULLs on the right, `FULL JOIN` keeps unmatched rows from both sides, and `CROSS JOIN` pairs every row with every row. A self join is an ordinary join of a table with itself under two aliases, here to put each employee next to their manager:

```sql
SELECT e.name, m.name AS manager
FROM employees AS e
LEFT JOIN employees AS m ON m.id = e.manager_id
ORDER BY e.id;
```

With a `LEFT JOIN`, where a condition goes matters. A condition on the right table in `ON` decides which rows match; the same condition in `WHERE` runs after the join, throws away the NULL-padded rows and quietly turns the query into an inner join. And `COUNT(*)` counts the padded row, so count a right-side column instead.

```sql
-- well-paid direct reports per employee
SELECT m.name, COUNT(r.id) AS reports
FROM employees AS m
LEFT JOIN employees AS r ON r.manager_id = m.id AND r.salary >= 80
GROUP BY m.id, m.name
ORDER BY m.id;
```

An **anti-join** keeps rows with no match. `NOT EXISTS` is the safe default: it is immune to the NULL trap above, and most planners run it as an efficient anti-join. A `LEFT JOIN ... WHERE right.key IS NULL` gives the same rows. For the positive case, `EXISTS` and `IN` usually get the same plan in a modern database; what differs is the NULL behaviour of `NOT IN`, not speed.

```sql
SELECT e.name
FROM employees AS e
WHERE NOT EXISTS (SELECT 1 FROM employees AS r WHERE r.manager_id = e.id)
ORDER BY e.id;
```

A cross join with a filter compares every pair of rows, for example to find the closest two points. Pair rows by `a.id < b.id`: it skips each row against itself and visits each pair once. Excluding self-pairs by comparing coordinates instead would also skip two different points that sit at the same place, and miss a true distance of zero.

```sql
SELECT ROUND(MIN(SQRT(POWER(a.x - b.x, 2) + POWER(a.y - b.y, 2))), 2) AS shortest
FROM points AS a
JOIN points AS b ON a.id < b.id;
-- PostgreSQL only: ROUND(MIN(SQRT(...))::numeric, 2)
```

```python
assert q("""
SELECT e.name, m.name AS manager
FROM employees AS e
LEFT JOIN employees AS m ON m.id = e.manager_id
ORDER BY e.id;
""") == [("Ana", None), ("Ben", "Ana"), ("Cy", "Ana"), ("Dee", "Ana"),
         ("Eve", "Dee"), ("Fay", "Dee"), ("Gus", "Dee")]

on_version = q("""
SELECT m.name, COUNT(r.id) AS reports
FROM employees AS m
LEFT JOIN employees AS r ON r.manager_id = m.id AND r.salary >= 80
GROUP BY m.id, m.name
ORDER BY m.id;
""")
assert on_version == [("Ana", 3), ("Ben", 0), ("Cy", 0), ("Dee", 0), ("Eve", 0), ("Fay", 0), ("Gus", 0)]

where_version = q("""SELECT m.name, COUNT(r.id) FROM employees AS m
LEFT JOIN employees AS r ON r.manager_id = m.id WHERE r.salary >= 80
GROUP BY m.id, m.name ORDER BY m.id""")
assert where_version == [("Ana", 3)]                  # everyone else vanished

count_star = q("""SELECT m.name, COUNT(*) FROM employees AS m
LEFT JOIN employees AS r ON r.manager_id = m.id AND r.salary >= 80
GROUP BY m.id, m.name ORDER BY m.id""")
assert dict(count_star)["Ben"] == 1                   # the NULL-padded row was counted

not_exists = q("""
SELECT e.name
FROM employees AS e
WHERE NOT EXISTS (SELECT 1 FROM employees AS r WHERE r.manager_id = e.id)
ORDER BY e.id;
""")
left_join_null = q("""SELECT e.name FROM employees AS e
LEFT JOIN employees AS r ON r.manager_id = e.id WHERE r.id IS NULL ORDER BY e.id""")
assert not_exists == left_join_null == [("Ben",), ("Cy",), ("Eve",), ("Fay",), ("Gus",)]

pts = [(1, 0, 0), (2, 3, 4), (3, 1, 1), (4, 5, 5), (5, 3, 4)]   # points 2 and 5 coincide
db.execute("CREATE TABLE points (id INTEGER PRIMARY KEY, x REAL, y REAL)")
db.executemany("INSERT INTO points VALUES (?, ?, ?)", pts)
shortest = q("""
SELECT ROUND(MIN(SQRT(POWER(a.x - b.x, 2) + POWER(a.y - b.y, 2))), 2) AS shortest
FROM points AS a
JOIN points AS b ON a.id < b.id;
""")[0][0]
assert shortest == round(pdist(np.array(pts)[:, 1:]).min(), 2) == 0.0
by_coords = q("""SELECT ROUND(MIN(SQRT(POWER(a.x - b.x, 2) + POWER(a.y - b.y, 2))), 2)
FROM points AS a, points AS b WHERE (a.x, a.y) <> (b.x, b.y)""")[0][0]
assert by_coords == 1.41                              # the coincident pair was skipped
```

## Window functions
A window function computes a value for each row from a set of related rows, without collapsing them the way `GROUP BY` does. The full shape, supported by PostgreSQL 11 and later and SQLite 3.30 and later (SQLite gained the `GROUPS` and `EXCLUDE` frame options in 3.28 and `NULLS FIRST | LAST` in 3.30):

```plaintext
function(args) OVER (
    [PARTITION BY expr, ...]                                -- separate windows, like GROUP BY
    [ORDER BY expr [ASC | DESC] [NULLS FIRST | LAST], ...]  -- order inside each window
    [ROWS | RANGE | GROUPS BETWEEN frame_start AND frame_end]
    [EXCLUDE NO OTHERS | CURRENT ROW | GROUP | TIES]
)
frame_start, frame_end:  UNBOUNDED PRECEDING | n PRECEDING | CURRENT ROW
                         | n FOLLOWING | UNBOUNDED FOLLOWING
```

### Ranking
`ROW_NUMBER` numbers rows 1, 2, 3 with no ties. `RANK` gives tied rows the same number and then skips (1, 2, 2, 4). `DENSE_RANK` does not skip (1, 2, 2, 3). `NTILE(k)` deals rows into `k` buckets as evenly as possible by row count, so it ignores ties: two equal salaries can land in different quartiles. Give `ROW_NUMBER` and `NTILE` a unique sort key, or the result can change between runs.

```sql
SELECT name, salary,
       ROW_NUMBER() OVER (ORDER BY salary DESC, id) AS row_num,
       RANK()       OVER (ORDER BY salary DESC)     AS rnk,
       DENSE_RANK() OVER (ORDER BY salary DESC)     AS dense,
       NTILE(4)     OVER (ORDER BY salary DESC, id) AS quartile
FROM employees
WHERE salary IS NOT NULL
ORDER BY salary DESC, id;
```

```python
assert q("""
SELECT name, salary,
       ROW_NUMBER() OVER (ORDER BY salary DESC, id) AS row_num,
       RANK()       OVER (ORDER BY salary DESC)     AS rnk,
       DENSE_RANK() OVER (ORDER BY salary DESC)     AS dense,
       NTILE(4)     OVER (ORDER BY salary DESC, id) AS quartile
FROM employees
WHERE salary IS NOT NULL
ORDER BY salary DESC, id;
""") == [("Ana", 120, 1, 1, 1, 1), ("Ben", 100, 2, 2, 2, 1), ("Cy", 100, 3, 2, 2, 2),
         ("Dee", 90, 4, 4, 3, 2), ("Eve", 70, 5, 5, 4, 3), ("Gus", 60, 6, 6, 5, 4)]
```

Six rows into four buckets gives sizes 2, 2, 1, 1: the remainder goes to the first buckets.

### Frames: ROWS, RANGE and the default
An aggregate used as a window, such as `SUM(...) OVER (...)`, adds up the rows in the current row's **frame**. With no `ORDER BY` the frame is the whole partition. With an `ORDER BY` and no frame clause, the default is `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, and `RANGE` includes the current row's **peers**, the rows that tie with it on the sort key. A running total over a key with ties therefore jumps by the whole tied group at once. `ROWS` counts physical rows instead, and `GROUPS` counts peer groups.

```sql
SELECT id, day, amount,
       SUM(amount) OVER (ORDER BY day) AS range_total,
       SUM(amount) OVER (ORDER BY day, id
                         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS rows_total,
       AVG(amount) OVER (ORDER BY day, id
                         ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving_avg_3
FROM orders
WHERE shop = 'A'
ORDER BY day, id;
```

The default frame is also why `LAST_VALUE` looks broken: the frame ends at the current row, or at its last peer when the sort key has ties, so the "last value" is the current row's own value (or a tied row's). Widen the frame to the whole partition, or use `FIRST_VALUE` with the order reversed.

```sql
SELECT shop, id,
       LAST_VALUE(amount) OVER (PARTITION BY shop ORDER BY day, id) AS default_frame,
       LAST_VALUE(amount) OVER (PARTITION BY shop ORDER BY day, id
                                ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS last_in_shop
FROM orders
ORDER BY shop, day, id;
```

```python
frames = q("""
SELECT id, day, amount,
       SUM(amount) OVER (ORDER BY day) AS range_total,
       SUM(amount) OVER (ORDER BY day, id
                         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS rows_total,
       AVG(amount) OVER (ORDER BY day, id
                         ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving_avg_3
FROM orders
WHERE shop = 'A'
ORDER BY day, id;
""")
ids, days, amounts, range_total, rows_total, moving = map(list, zip(*frames))
assert range_total == [10, 35, 35, 65, 80]            # the two 2026-01-20 orders arrive together
assert rows_total == [10, 30, 35, 65, 80]

# the same three columns from pandas
a = pd.DataFrame({"day": days, "amount": amounts})
assert rows_total == a.amount.cumsum().tolist()
assert range_total == a.groupby("day").amount.sum().cumsum().reindex(a.day).tolist()
assert np.allclose(moving, a.amount.rolling(3, min_periods=1).mean())

assert q("""
SELECT shop, id,
       LAST_VALUE(amount) OVER (PARTITION BY shop ORDER BY day, id) AS default_frame,
       LAST_VALUE(amount) OVER (PARTITION BY shop ORDER BY day, id
                                ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS last_in_shop
FROM orders
ORDER BY shop, day, id;
""") == [("A", 1, 10, 15), ("A", 2, 20, 15), ("A", 3, 5, 15), ("A", 4, 30, 15), ("A", 5, 15, 15),
         ("B", 6, 7, 3), ("B", 7, 3, 3)]

# GROUPS counts peer groups: the previous date that has orders plus the current date, whatever the row count
# with ties on the sort key the default frame ends at the last peer, so both 2026-01-20 orders
# get the same LAST_VALUE, and at least one of them is not its own amount
tied = dict(q("""SELECT id, LAST_VALUE(amount) OVER (ORDER BY day) FROM orders
WHERE shop = 'A' AND day = '2026-01-20'"""))
assert tied[2] == tied[3] and tied[2] in (20, 5)

assert [r[0] for r in q("""SELECT SUM(amount) OVER (ORDER BY day GROUPS BETWEEN 1 PRECEDING AND CURRENT ROW)
FROM orders WHERE shop = 'A' ORDER BY day, id""")] == [10, 35, 35, 55, 45]
```

### Looking back with LAG
`LAG(x, n, default)` reads `x` from `n` rows earlier in the partition and `LEAD` reads ahead. They replace the old self join on `id = previous.id + 1`, which breaks the moment an id is missing. Month-over-month change is the standard use. The `WINDOW` clause names a window once so it can be reused:

```sql
WITH monthly AS (
    SELECT strftime('%Y-%m', day) AS month, SUM(amount) AS revenue
    FROM orders
    GROUP BY month
)
SELECT month, revenue,
       revenue - LAG(revenue) OVER w AS change,
       ROUND(100.0 * (revenue - LAG(revenue) OVER w) / LAG(revenue) OVER w, 1) AS pct_change
FROM monthly
WINDOW w AS (ORDER BY month)
ORDER BY month;
```

In PostgreSQL write `TO_CHAR(day, 'YYYY-MM')` or `DATE_TRUNC('month', day)` for the month. `LAG` means the previous **row**, not the previous month: if a month has no orders it has no row, and the comparison silently spans two months. Join to a calendar of months first when gaps are possible.

```python
assert q("""
WITH monthly AS (
    SELECT strftime('%Y-%m', day) AS month, SUM(amount) AS revenue
    FROM orders
    GROUP BY month
)
SELECT month, revenue,
       revenue - LAG(revenue) OVER w AS change,
       ROUND(100.0 * (revenue - LAG(revenue) OVER w) / LAG(revenue) OVER w, 1) AS pct_change
FROM monthly
WINDOW w AS (ORDER BY month)
ORDER BY month;
""") == [("2026-01", 42, None, None), ("2026-02", 33, -9, -21.4), ("2026-03", 15, -18, -54.5)]
```

## Patterns worth knowing by heart
### Nth highest value
"Second highest salary" (LeetCode [176](https://leetcode.com/problems/second-highest-salary/)) has two traps: ties, and the case where there is no second value, which should give NULL rather than no row. `DENSE_RANK` handles ties, and an outer `MAX` turns "no rows" into a single NULL, because an aggregate over an empty set still returns one row. Remember the alias on the subquery: PostgreSQL before version 16 requires one on every subquery in `FROM`.

```sql
SELECT MAX(salary) AS nth_highest
FROM (
    SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS r
    FROM employees
    WHERE salary IS NOT NULL
) AS ranked
WHERE r = :n;
```

### Top N per group
The same ranking with `PARTITION BY` gives the top earners per department. Choose the function by what a tie should do: `ROW_NUMBER` returns at most N rows per group, never more even with ties, `DENSE_RANK` returns everyone in the top N salary levels.

```sql
SELECT dept, name, salary
FROM (
    SELECT dept, name, salary,
           DENSE_RANK() OVER (PARTITION BY dept ORDER BY salary DESC) AS r
    FROM employees
    WHERE salary IS NOT NULL
) AS ranked
WHERE r <= 2
ORDER BY dept, salary DESC, name;
```

```python
nth_sql = """
SELECT MAX(salary) AS nth_highest
FROM (
    SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS r
    FROM employees
    WHERE salary IS NOT NULL
) AS ranked
WHERE r = :n;
"""
distinct_desc = sorted({s for (s,) in q("SELECT salary FROM employees WHERE salary IS NOT NULL")}, reverse=True)
for n in range(1, 8):                                  # brute force, including n past the end
    expected = distinct_desc[n - 1] if n <= len(distinct_desc) else None
    assert q(nth_sql, {"n": n}) == [(expected,)]
assert q(nth_sql, {"n": 2}) == [(100,)]

assert q("""
SELECT dept, name, salary
FROM (
    SELECT dept, name, salary,
           DENSE_RANK() OVER (PARTITION BY dept ORDER BY salary DESC) AS r
    FROM employees
    WHERE salary IS NOT NULL
) AS ranked
WHERE r <= 2
ORDER BY dept, salary DESC, name;
""") == [("eng", "Ana", 120), ("eng", "Ben", 100), ("eng", "Cy", 100),
         ("ops", "Gus", 60), ("sales", "Dee", 90), ("sales", "Eve", 70)]
```

### Gaps and islands: consecutive streaks
To find runs of consecutive days, number each user's days in order and subtract the number from the date. Within a run both go up by one per row, so the difference stays constant, and it changes at every gap. Grouping by that difference gives one row per streak. Remove duplicate days first, or a repeated day breaks the arithmetic.

```sql
WITH days AS (
    SELECT DISTINCT user_id, day FROM logins
),
numbered AS (
    SELECT user_id, day,
           julianday(day) - ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY day) AS island
    FROM days
)
SELECT user_id, MIN(day) AS start, COUNT(*) AS length
FROM numbered
GROUP BY user_id, island
ORDER BY user_id, start;
```

In PostgreSQL, with `day` a `date` column, write `day - (ROW_NUMBER() OVER (...))::int`, which is again a date.

```python
islands = q("""
WITH days AS (
    SELECT DISTINCT user_id, day FROM logins
),
numbered AS (
    SELECT user_id, day,
           julianday(day) - ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY day) AS island
    FROM days
)
SELECT user_id, MIN(day) AS start, COUNT(*) AS length
FROM numbered
GROUP BY user_id, island
ORDER BY user_id, start;
""")
assert islands == [("u1", "2026-01-01", 3), ("u1", "2026-01-05", 2), ("u2", "2026-01-02", 1)]

# brute force: walk each user's sorted days and cut at every gap
from datetime import date

brute = []
for user in ["u1", "u2"]:
    ds = sorted({date.fromisoformat(d) for (d,) in q("SELECT day FROM logins WHERE user_id = ?", (user,))})
    start = prev = ds[0]
    for d in ds[1:] + [None]:
        if d is None or (d - prev).days != 1:
            brute.append((user, start.isoformat(), (prev - start).days + 1))
            start = d
        prev = d
assert brute == islands
```

### Finding and removing duplicates
`GROUP BY` the identifying columns with `HAVING COUNT(*) > 1` lists the duplicates. To delete all but the first copy, delete every row that has an earlier twin. Normalize before comparing: here emails differ only in case.

```sql
SELECT LOWER(email) AS email, COUNT(*) AS copies
FROM contacts
GROUP BY LOWER(email)
HAVING COUNT(*) > 1;

DELETE FROM contacts
WHERE EXISTS (
    SELECT 1 FROM contacts AS keep
    WHERE LOWER(keep.email) = LOWER(contacts.email)
      AND keep.id < contacts.id
);
```

```sql
-- PostgreSQL only: DELETE ... USING joins the table to itself
DELETE FROM contacts AS dup
USING contacts AS keep
WHERE LOWER(dup.email) = LOWER(keep.email)
  AND dup.id > keep.id;
```

```python
db.execute("CREATE TABLE contacts (id INTEGER PRIMARY KEY, email TEXT)")
db.executemany("INSERT INTO contacts VALUES (?, ?)",
               [(1, "a@x.org"), (2, "b@x.org"), (3, "A@x.org"), (4, "a@x.org"), (5, "b@x.org")])
assert q("""
SELECT LOWER(email) AS email, COUNT(*) AS copies
FROM contacts
GROUP BY LOWER(email)
HAVING COUNT(*) > 1;
""") == [("a@x.org", 3), ("b@x.org", 2)]
db.execute("""
DELETE FROM contacts
WHERE EXISTS (
    SELECT 1 FROM contacts AS keep
    WHERE LOWER(keep.email) = LOWER(contacts.email)
      AND keep.id < contacts.id
);
""")
assert q("SELECT id, email FROM contacts ORDER BY id") == [(1, "a@x.org"), (2, "b@x.org")]
```

Run the `SELECT` first and the `DELETE` inside a transaction, so a wrong condition can be rolled back.

## Phrasebook: PostgreSQL and SQLite side by side
Dates, strings and casts are where dialects drift furthest. The PostgreSQL column is not run here; the SQLite column is checked below.

| Task | PostgreSQL | SQLite |
|---|---|---|
| Year-month text | `TO_CHAR(d, 'YYYY-MM')` | `strftime('%Y-%m', d)` |
| Truncate to month | `DATE_TRUNC('month', d)` | `date(d, 'start of month')` |
| Year as a number | `EXTRACT(YEAR FROM d)` | `CAST(strftime('%Y', d) AS INTEGER)` |
| Minute of a timestamp | `EXTRACT(MINUTE FROM ts)` | `CAST(strftime('%M', ts) AS INTEGER)` |
| Days between dates | `d2 - d1` on two `date` values | `julianday(d2) - julianday(d1)` |
| Add a month | `d + INTERVAL '1 month'` | `date(d, '+1 month')` |
| Day of week | `TO_CHAR(d, 'FMDay')` gives `Monday` | `strftime('%w', d)` gives 0 to 6, Sunday is 0 |
| Current time | `NOW()`, in the session time zone | `datetime('now')`, in UTC |
| Cast | `x::numeric` or `CAST(x AS numeric)` | `CAST(x AS REAL)` |
| Largest of several columns | `GREATEST(a, b, c)` | `MAX(a, b, c)` |
| Substring position | `POSITION('at' IN 'data')` | `instr('data', 'at')` |
| Substring | `SUBSTRING(s FROM 2 FOR 3)` | `substr(s, 2, 3)` |
| Replace text | `REPLACE('data', 'a', 'x')` gives `dxtx` | the same |

Three rows hide real differences, not just spelling. `GREATEST` skips NULLs in PostgreSQL, while SQLite's multi-argument `MAX` returns NULL if any argument is NULL. Adding a month to January 31 gives February 28 in PostgreSQL but overflows to March 3 in SQLite. And the `||` operator returns NULL if either side is NULL, while `CONCAT` skips NULLs, in both databases. PostgreSQL's `AGE(a, b)`, which returns a calendar interval such as `10 mons 9 days`, has no SQLite counterpart.

```python
d = "2025-11-10 15:47:00"
assert q("SELECT strftime('%Y-%m', ?), date(?, 'start of month')", (d, d)) == [("2025-11", "2025-11-01")]
assert q("SELECT CAST(strftime('%Y', ?) AS INTEGER), CAST(strftime('%M', ?) AS INTEGER)", (d, d)) == [(2025, 47)]
assert q("SELECT julianday('2025-11-10') - julianday('2025-01-01')") == [(313.0,)]
assert (date(2025, 11, 10) - date(2025, 1, 1)).days == 313
assert q("SELECT strftime('%w', '2025-11-10')") == [("1",)] and date(2025, 11, 10).strftime("%A") == "Monday"
assert q("SELECT MAX(2, 8, 3), MIN(2, 8, 3), MAX(1, NULL)") == [(8, 2, None)]
assert q("SELECT instr('data', 'at'), substr('abcdef', 2, 3), replace('data', 'a', 'x')") == [(2, "bcd", "dxtx")]
assert q("SELECT date('2026-01-31', '+1 month')") == [("2026-03-03",)]
assert q("SELECT 'a' || NULL, concat('a', NULL)") == [(None, "a")]
assert q("SELECT CAST('123' AS INTEGER) + 1") == [(124,)]
```

## Pitfalls checklist
- Filtering on an aggregate in `WHERE`, or on a window result in `WHERE` or `HAVING`. Use `HAVING` for the first and a CTE for the second.
- Integer division: `COUNT(*) / total` is 0 for any share under 100%. Multiply by `100.0` first.
- `NOT IN` against a column that can hold NULL. Use `NOT EXISTS`.
- A right-table condition in the `WHERE` of a `LEFT JOIN`, and `COUNT(*)` over a `LEFT JOIN`. Move the condition into `ON` and count a right-side column.
- Forgetting the default frame. With `ORDER BY` it is `RANGE ... CURRENT ROW`, so ties move together and `LAST_VALUE` returns the current row's value (or its last peer's).
- `ROW_NUMBER` or `NTILE` without a unique sort key: the output can change between runs.
- NULL sort order: last in ascending PostgreSQL sorts, first in descending ones, the reverse in SQLite. Say `NULLS LAST`.
- `LAG` steps back one row, not one period. Fill in missing periods first.
- A query that only works because SQLite is lenient, such as a `SELECT` alias used in `WHERE`. Test on the database you will deploy to.

The same operations in pandas are in the [pandas cheatsheet](../pandas-cheatsheet/). The free interactive book [Select Star SQL](https://selectstarsql.com/) and the PostgreSQL and SQLite documentation on window functions informed this page; the examples, data and explanations here are written fresh.
