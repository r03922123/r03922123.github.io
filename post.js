// Post pages: render \( \) and \[ \] math with KaTeX, then highlight code blocks.
document.addEventListener("DOMContentLoaded", () => {
  renderMathInElement(document.body, {
    delimiters: [{ left: "\\[", right: "\\]", display: true }, { left: "\\(", right: "\\)", display: false }],
    ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
  });
  hljs.highlightAll();
});
