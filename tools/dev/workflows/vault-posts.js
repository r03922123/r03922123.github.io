export const meta = {
  name: 'vault-posts',
  description: 'Turn 55 Obsidian vault notes into rewritten, code-verified public posts, each followed by an adversarial review',
  phases: [
    { title: 'Write', detail: 'one writer per note: original public post, runnable asserted code' },
    { title: 'Review', detail: 'one skeptic per post: claims, citations, code, privacy; fix and rebuild' },
  ],
}

const SITE = '/Users/linjiehong/Desktop/projects/r03922123.github.io'
const P = `${SITE}/tools/posts`
const V = '/Users/linjiehong/Documents/Obsidian Vault'
const F = 'Machine learning fundamentals', PR = 'Probability', ST = 'Statistics and experimentation', L = 'LLMs and generative AI', R = 'Recommender systems', SD = 'ML systems and design', C = 'Coding and data'

const POSTS = [
  ['ml-fundamentals-quick-reference', F, 1, '02-ml-breadth/ML Fundamentals.md', ''],
  ['bias-variance-tradeoff', F, 2, '02-ml-breadth/Bias-Variance Tradeoff.md', ''],
  ['regression-as-maximum-likelihood', F, 3, '02-ml-breadth/Regression high level intro.md', ''],
  ['cross-entropy-vs-mse', F, 4, '02-ml-breadth/Why cross entropy is better than mse in binary classification.md', ''],
  ['stable-softmax-cross-entropy', F, 5, '02-ml-breadth/Numerically Stable Softmax and Cross Entropy.md', 'The note is adapted from Jay Mody\'s blog post "Numerically Stable Softmax and Cross Entropy": write a fully original derivation and code, and credit his post in a closing line.'],
  ['entropy-and-information', F, 6, '02-ml-breadth/Entropy and Information Theory.md', ''],
  ['regularization-l1-l2', F, 7, '02-ml-breadth/Regularization.md', ''],
  ['dropout', F, 8, '02-ml-breadth/Dropout.md', ''],
  ['overfitting-playbook', F, 9, '02-ml-breadth/How to mitigate Overfitting.md', ''],
  ['batchnorm-vs-layernorm', F, 10, '02-ml-breadth/BatchNorm vs LayerNorm.md', ''],
  ['attention-scaling', F, 11, '02-ml-breadth/Why scale by sqrt(d_head) for attention scores.md', ''],
  ['contrastive-loss', F, 12, '02-ml-breadth/Contrastive Loss.md', ''],
  ['conditional-probability-bayes', PR, 1, '02-ml-breadth/probability/Conditional Probability.md', ''],
  ['bernoulli-binomial', PR, 2, '02-ml-breadth/probability/Bernoulli and Binomial.md', 'The note is very short (96 words): expand it into a complete post.'],
  ['poisson-distribution', PR, 3, '02-ml-breadth/probability/Poisson Distribution.md', ''],
  ['exponential-distribution', PR, 4, '02-ml-breadth/probability/Exponential Distribution PMF.md', 'The note title says PMF; the exponential is continuous and has a PDF. Fix this.'],
  ['normal-distribution', PR, 5, '02-ml-breadth/probability/Normal Distribution.md', ''],
  ['beta-uncertainty', PR, 6, '02-ml-breadth/probability/Uncertainty Theory.md', ''],
  ['is-this-coin-fair', PR, 7, '02-ml-breadth/probability/Probability Questions - Fair Coin.md', ''],
  ['coin-flips-normal-approximation', PR, 8, '02-ml-breadth/probability/Probability Questions - Flipping 576 Times - InterviewQuery.md', 'The problem comes from InterviewQuery: do NOT reproduce their problem statement; pose the question in your own words and numbers, solve it, and credit InterviewQuery as the inspiration.'],
  ['probability-cheatsheet', PR, 9, '02-ml-breadth/probability/Probability One Pager Cheatsheet.md', ''],
  ['statistical-estimation-mle-map', ST, 1, '02-ml-breadth/statistics/Statistical Estimation.md', ''],
  ['p-values', ST, 2, '02-ml-breadth/statistics/P-Value.md', ''],
  ['ab-testing', ST, 3, '02-ml-breadth/statistics/AB Testing.md', 'Link the existing long-form post /posts/evaluating-ml-systems/ where relevant (it has an online-testing section); do not duplicate it.'],
  ['causal-inference', ST, 4, '02-ml-breadth/statistics/Causal Inference.md', ''],
  ['simulation-based-evaluation', ST, 5, '02-ml-breadth/statistics/Simulation-Based Evaluation.md', ''],
  ['roc-auc-vs-pr-auc', ST, 6, '02-ml-breadth/statistics/ROC AUC and PR AUC.md', ''],
  ['naive-bayes', ST, 7, '02-ml-breadth/statistics/Naive Bayes.md', ''],
  ['decision-trees-to-boosting', ST, 8, '02-ml-breadth/statistics/Decision Tree series.md', ''],
  ['statistics-cheatsheet', ST, 9, '02-ml-breadth/statistics/Interview Stats Cheatsheet.md', 'The note is framed for interviews: make it a general statistics reference.'],
  ['foundation-models', L, 1, '03-ml-depth/llm-genai/LLMs Foundation Models.md', ''],
  ['transformers-and-llms-key-ideas', L, 2, '03-ml-depth/llm-genai/Stanford CME295 Transformers & LLMs.md', 'These are personal course notes from Stanford CME295: write an original synthesis of the key ideas (KV cache, MQA/GQA, speculative decoding, mixed precision, etc.) citing the original papers, and credit the course as where you studied it. Do not reproduce slides or lecture text.'],
  ['flashattention', L, 3, '03-ml-depth/llm-genai/FlashAttention.md', ''],
  ['llm-inference', L, 4, '03-ml-depth/llm-genai/LLM Inference.md', ''],
  ['lora-explained', L, 5, '03-ml-depth/llm-genai/LoRA.md', 'A long-form post /posts/lora-from-scratch/ already implements LoRA step by step: make this post the conceptual one (why low rank, initialization, alpha/r, where to apply, variants) and link that post instead of repeating its code.'],
  ['parameter-efficient-fine-tuning', L, 6, '03-ml-depth/llm-genai/LLM Finetuning LoRA PEFT.md', 'Make this the comparison post across PEFT methods (LoRA, QLoRA, adapters, prefix/prompt tuning, full fine-tuning); link /posts/lora-from-scratch/ and ../lora-explained/ rather than repeating them.'],
  ['rlhf-alignment', L, 7, '03-ml-depth/llm-genai/RLHF Alignment.md', 'Link /posts/text-to-music-2026/ for the DPO derivation rather than repeating it in full.'],
  ['rag-architecture', L, 8, '03-ml-depth/llm-genai/RAG Architecture.md', ''],
  ['vector-databases', L, 9, '03-ml-depth/Vector Databases.md', ''],
  ['llm-agents-langchain-langgraph', L, 10, '03-ml-depth/llm-genai/LangChain LangGraph Agents.md', 'langchain/langgraph are NOT installed: show library usage in ```plaintext blocks (clearly marked as not executed), and put the runnable ideas (an agent loop, tool dispatch, state graph) in stdlib-only python blocks.'],
  ['two-tower-retrieval', R, 1, '03-ml-depth/recsys/Two-Tower Retrieval.md', ''],
  ['recommendation-ranking', R, 2, '03-ml-depth/recsys/Recommendation Ranking.md', 'Link /posts/position-bias/ for debiasing clicks.'],
  ['recsys-features', R, 3, '03-ml-depth/recsys/Feature Ideas for Recommendation Systems.md', ''],
  ['sequential-user-modeling', R, 4, '03-ml-depth/recsys/Sequential User Modeling.md', ''],
  ['transformers4rec', R, 5, '03-ml-depth/recsys/Transformer4Rec.md', 'The note is scrappy NVIDIA Merlin/NVTabular snippets that need a GPU (not installed): make it an explanatory post on session-based recommendation with Transformers4Rec; library code in ```plaintext blocks marked as not executed, and a small runnable stdlib/numpy/torch illustration of the core idea.'],
  ['cold-start', R, 6, '03-ml-depth/recsys/Cold-Start problem.md', ''],
  ['ml-system-design-framework', SD, 1, '03-ml-depth/system-design/ML system design - Introduction.md', 'Remove interview framing: present it as a framework for designing ML systems.'],
  ['ml-system-design-process', SD, 2, '03-ml-depth/system-design/ML System Design - Delivery.md', 'The note is an interview playbook with time budgets: turn it into a general design-review process for ML systems (no interview framing, no time budgets for an interview). Link /posts/evaluating-ml-systems/ for evaluation.'],
  ['ml-infrastructure', SD, 3, '03-ml-depth/system-design/ML Infra Design.md', ''],
  ['docker-for-ml', SD, 4, '03-ml-depth/system-design/Docker.md', 'Shell and Dockerfile snippets go in ```plaintext blocks (they are not executed).'],
  ['crowd-heatmap-feasibility', SD, 5, '03-ml-depth/system-design/都市人潮熱圖開發可行性.md', 'The note is in Chinese and short: translate and expand it into an English post about the feasibility of a city crowd-heatmap app (data sources, privacy, forecasting approach, system sketch). Keep it as a design exploration, not a claim that it was built.'],
  ['solving-problems-under-time-limits', C, 1, '01-coding/Coding Flow.md', 'The note is a coding-interview process: make it a general method for solving algorithm problems under a time limit (no interview framing, no grading criteria). Link /leetcode/ for patterns.'],
  ['ml-coding-from-scratch', C, 2, '01-coding/ML Coding.md', 'Implement the building blocks it lists from scratch (numpy/torch) with asserts against library results.'],
  ['sql-cheatsheet', C, 3, '01-coding/SQL Cheatsheet.md', 'Verify the SQL: put each query in a python block that runs it on an in-memory sqlite3 database with a tiny table and asserts the result (sqlite 3.45 supports window functions). Show the bare SQL too. Mark Postgres-only syntax clearly.'],
  ['pandas-cheatsheet', C, 4, '01-coding/Pandas Cheatsheet.md', 'pandas 2.3 is installed: every snippet runs on a tiny DataFrame with asserts.'],
].map(([slug, group, order, note, extra]) => ({ slug, group, order, note, extra }))

const ALL_SLUGS = POSTS.map(p => p.slug).join(', ')
const RULES = `
You are writing one post for a public personal site (an educational ML blog in the style of Lil'Log). Owner: Jie-Hong Lin, ML engineer.
Source format: ${P}/src/<slug>.md, front matter between '---' lines with exactly these keys: title, summary, group, order, tags (tags joined with ' · '). Body: ## and ### headings, paragraphs, '- ' and '1. ' lists, pipe tables, \`\`\`python fences (executed), \`\`\`plaintext / \`\`\`sql / \`\`\`bash fences (shown, not executed), inline \`code\`, **bold**, [text](url), inline math \\( ... \\), display math on its own lines starting with \\[ and ending with \\]. NEVER use $ for math. No images, no Obsidian syntax (no [[wikilinks]], no > [!callouts], no ==highlights==).
Match the voice, depth and structure of these published posts first (read them): ${SITE}/tools/leetcode/src/heap.md and ${SITE}/posts/position-bias/index.html (a long-form post). Skim ${SITE}/tools/leetcode/build.py to see the renderer.
Rules:
- ORIGINAL prose and code. The vault note is raw study material: never paste it. Fix its errors instead of repeating them. Where material comes from a book, course, blog or site, rewrite fully and credit the source in a closing line. Cite papers/authors only when you are confident the citation is accurate.
- Every \`\`\`python block runs: all of a post's python blocks are concatenated in order and executed as one script (numpy, pandas, scipy, scikit-learn, torch, sqlite3 are installed; no network; keep runtime under ~20 s). Use asserts to prove what the prose claims; cross-check against a brute force or a library where feasible.
- Structure: a short plain-words intro, sections that build the idea, worked examples, a "When to use it"/"Pitfalls" style section where it fits. Roughly 700-1600 words of prose plus code. Group: the one given. Order: the one given.
- Links: sibling posts as [text](../<slug>/) only for these slugs: ${ALL_SLUGS}. Long-form posts as /posts/position-bias/, /posts/evaluating-ml-systems/, /posts/lora-from-scratch/, /posts/text-to-music-2026/. LeetCode as /leetcode/ or /leetcode/<slug>/ (see ${SITE}/tools/leetcode/src for slugs).
- PRIVACY, the site is public: no interview framing ("in an interview", "interviewer", "interview question"), no company names used as hiring targets, no "role-relevance", "company-demand" or "target-company" material, nothing personal, never mention the vault or Obsidian.
- Do NOT edit any other file in the repo (another engineer is redesigning the site at the same time), and do NOT run git commands.
Build and test your post with:  cd ${SITE} && python3 -I tools/posts/build.py . <slug>   (it runs the python blocks and renders posts/<slug>/index.html; it also rewrites posts/index.html, which is expected). Iterate until it prints "<slug>: N min, K python blocks pass" (or "no python").`

const WRITER = { type: 'object', properties: { slug: { type: 'string' }, title: { type: 'string' }, build_line: { type: 'string' },
  note_issues: { type: 'array', items: { type: 'string' }, description: 'errors, borrowed material or private/interview content found in the note and how the post handles it' } },
  required: ['slug', 'title', 'build_line', 'note_issues'] }
const REVIEW = { type: 'object', properties: { slug: { type: 'string' },
  issues: { type: 'array', items: { type: 'object', properties: { issue: { type: 'string' }, fixed: { type: 'boolean' } }, required: ['issue', 'fixed'] } },
  build_line: { type: 'string' } }, required: ['slug', 'issues', 'build_line'] }

const results = await pipeline(
  POSTS,
  (p) => agent(`${RULES}
Your post: slug "${p.slug}", group "${p.group}", order ${p.order}. Source note (read it fully): ${V}/${p.note}
${p.extra ? 'Specific instructions: ' + p.extra : ''}
Write ${P}/src/${p.slug}.md, build it until it passes, and return the structured summary.`,
    { label: `write:${p.slug}`, phase: 'Write', schema: WRITER }),
  (w, p) => agent(`${RULES}
You are the adversarial reviewer of ${P}/src/${p.slug}.md (rendered at ${SITE}/posts/${p.slug}/index.html), written from ${V}/${p.note}. ${p.extra ? 'Instructions the writer had: ' + p.extra : ''}
Writer's report: ${JSON.stringify(w)}
Assume the post is wrong somewhere and try to prove it: (1) every technical claim, formula, derivation, complexity and number; (2) every citation and attribution (remove anything you cannot confirm); (3) code: do the asserts test what the prose claims, are brute forces genuinely independent, are edge cases covered; (4) originality: no passages pasted from the note or another source; (5) house rules: no $ math, no Obsidian syntax, allowed links only, and PRIVACY (no interview framing, no hiring-target companies, nothing personal, no vault mention).
Fix every real issue in the .md, rebuild until it passes, and return the structured review.`,
    { label: `review:${p.slug}`, phase: 'Review', schema: REVIEW }),
)
const done = results.filter(Boolean)
log(`${done.length}/${POSTS.length} posts written and reviewed`)
return done.map(r => ({ slug: r.slug, build: r.build_line, fixed: r.issues.filter(i => i.fixed).length, open: r.issues.filter(i => !i.fixed).map(i => i.issue) }))