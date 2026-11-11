// The search box on the Posts and LeetCode lists (SPEC-site-search). This script inserts the box, so a reader
// without JavaScript sees the lists as before and no dead box. The index is fetched only when the box is first
// used, and no query leaves the page. Index and query text reach the page through textContent only, never markup.
import { rank } from "/search-core.mjs";

const main = document.querySelector("main");
const topics = document.querySelector(".topics");
const box = document.createElement("div");
box.className = "search";
const form = document.createElement("form");
form.setAttribute("role", "search");
const label = Object.assign(document.createElement("label"), { htmlFor: "q", textContent: "Search all posts" });
const input = Object.assign(document.createElement("input"), { type: "search", id: "q", autocomplete: "off" });
const status = document.createElement("p");
status.setAttribute("role", "status");                 // a polite live region: the result count is announced
const results = Object.assign(document.createElement("ul"), { className: "series" });
form.append(label, input);
box.append(form, status, results);
topics.before(box);

let index;
const load = () => (index ??= fetch("/search-index.json").then((r) => {
  if (!r.ok) throw new Error(r.status);
  return r.json();
}));

function fail() {
  form.remove();                                        // no dead box: the lists below still work
  main.classList.remove("searching");
  status.textContent = "Search could not load. The lists below still work.";
}

function row(e) {
  const li = document.createElement("li");
  const a = Object.assign(document.createElement("a"), { href: e.u, textContent: e.t });
  const summary = Object.assign(document.createElement("span"), { textContent: e.s });
  const section = Object.assign(document.createElement("span"), { className: "meta", textContent: e.c });
  li.append(a, summary, section);
  return li;
}

async function search() {
  const q = input.value;
  // replaceState, not pushState: each keystroke would otherwise be a step for the back button.
  const url = new URL(location.href);
  if (q.trim()) url.searchParams.set("q", q); else url.searchParams.delete("q");
  history.replaceState(history.state, "", url);       // the #topic filter stays, so clearing restores it
  let entries;
  try {
    entries = await load();
  } catch {
    return fail();
  }
  if (q !== input.value) return;                        // a newer keystroke has its own call
  const found = q.trim() ? rank(entries, q) : null;
  main.classList.toggle("searching", found !== null);   // results replace the cards and sections
  results.replaceChildren(...(found || []).map(row));
  status.textContent = found === null ? "" : found.length ? `${found.length} result${found.length === 1 ? "" : "s"}`
    : "No results";
}

form.addEventListener("submit", (event) => event.preventDefault());
input.addEventListener("input", search);
input.addEventListener("focus", () => load().catch(fail), { once: true });
const q = new URLSearchParams(location.search).get("q");
if (q) {
  input.value = q;
  search();
}
