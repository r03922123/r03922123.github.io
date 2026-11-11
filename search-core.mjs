// Ranking for the site search (SPEC-site-search). Pure, with no DOM, so node --test can check it.
// An entry is one row of search-index.json: u url, t title, s summary, g tags, c section, h headings.

// Title 5, tags 3, headings 2, summary and section 1: the weights the prototype reached 13 of 13 golden queries with.
const FIELDS = [["t", 5], ["g", 3], ["h", 2], ["s", 1], ["c", 1]];

// Lowercase runs of letters and digits; punctuation only separates words, so "two-tower" is "two tower".
const tokens = (text) => String(text).toLowerCase().match(/[\p{L}\p{N}]+/gu) || [];

// Per word and field, a whole-word match counts twice a prefix match, so an exact hit outranks a longer word.
function score(entry, words) {
  const fields = FIELDS.map(([key, weight]) => [tokens([entry[key]].flat().join(" ")), weight]);
  let total = 0;
  for (const word of words) {
    let points = 0;
    for (const [list, weight] of fields) {
      if (list.includes(word)) points += 2 * weight;
      else if (list.some((t) => t.startsWith(word))) points += weight;
    }
    if (!points) return 0;                 // every word must match, so adding a word narrows the results
    total += points;
  }
  return total;
}

// The top 20 entries for a query, best first; ties go by URL, so the order is the same on every run.
export function rank(entries, query) {
  const words = tokens(query);
  if (!words.length) return [];
  return entries.map((e) => [score(e, words), e]).filter(([s]) => s > 0)
    .sort((a, b) => b[0] - a[0] || (a[1].u < b[1].u ? -1 : a[1].u > b[1].u ? 1 : 0)).slice(0, 20).map(([, e]) => e);
}
