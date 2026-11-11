// Tests for search-core.mjs (SPEC-site-search): the ranking rules, then the golden queries on the real index.
// Run: node --test scripts/test_search.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { rank } from "../search-core.mjs";

const entry = (u, fields) => ({ u, t: "", s: "", g: "", c: "", h: [], ...fields });
const urls = (entries, query) => rank(entries, query).map((e) => e.u);

test("an exact match outranks a prefix match", () => {
  const entries = [entry("/a/", { t: "Knapsacks in practice" }), entry("/b/", { t: "The knapsack" })];
  assert.deepEqual(urls(entries, "knapsack"), ["/b/", "/a/"]);
});

test("a title match outranks a summary match", () => {
  const entries = [entry("/a/", { s: "About dropout" }), entry("/b/", { t: "Dropout" })];
  assert.deepEqual(urls(entries, "dropout"), ["/b/", "/a/"]);
});

test("every query word is required", () => {
  const entries = [entry("/a/", { t: "Two pointers" }), entry("/b/", { t: "Two tower retrieval" })];
  assert.deepEqual(urls(entries, "two tower"), ["/b/"]);
  assert.deepEqual(urls(entries, "two tower missing"), []);
});

test("matching ignores case", () => {
  const entries = [entry("/a/", { t: "CUPED variance reduction" }), entry("/b/", { t: "Something else" })];
  assert.deepEqual(urls(entries, "cuped"), ["/a/"]);
  assert.deepEqual(urls(entries, "VARIANCE"), ["/a/"]);
});

test("an empty or punctuation-only query returns nothing", () => {
  const entries = [entry("/a/", { t: "Anything" })];
  for (const query of ["", "   ", "?!", "-- ..", "()"]) assert.deepEqual(rank(entries, query), [], JSON.stringify(query));
});

test("ties sort the same way on every run", () => {
  const entries = ["/c/", "/a/", "/d/", "/b/"].map((u) => entry(u, { t: "Same title" }));
  const first = urls(entries, "same");
  assert.deepEqual(first, ["/a/", "/b/", "/c/", "/d/"]);
  assert.deepEqual(urls([...entries].reverse(), "same"), first);
});

// The golden queries: the expected post, by its slug, must be in the top 3 on the committed index.
const GOLDEN = [
  ["dropout", "dropout"], ["cuped", "ab-testing"], ["position bias", "position-bias"], ["knapsack", "knapsack"],
  ["union find", "union-find"], ["peeking", "ab-testing"], ["cold start", "cold-start"],
  ["two tower", "two-tower-retrieval"], ["vector database", "vector-databases"],
  ["bayes", "conditional-probability-bayes"], ["poisson", "poisson-distribution"],
  ["flashattention", "flashattention"], ["sql window", "sql-cheatsheet"],
];
const index = JSON.parse(readFileSync(new URL("../search-index.json", import.meta.url), "utf8"));

for (const [query, slug] of GOLDEN) {
  test(`golden: "${query}" finds ${slug} in the top 3`, () => {
    const top = rank(index, query).slice(0, 3).map((e) => e.u.split("/").at(-2));
    assert.ok(top.includes(slug), `top 3: ${top.join(", ")}`);
  });
}
