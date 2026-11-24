/* Page behaviours: scroll reveal and BibTeX copy (no dependencies). */
(function () {
  "use strict";
  const els = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
    }, { threshold: 0.08, rootMargin: "0px 0px -40px 0px" });
    els.forEach((el) => io.observe(el));
  } else {
    els.forEach((el) => el.classList.add("in"));
  }
  const copyBtn = document.getElementById("copy-bib");
  if (copyBtn) {
    copyBtn.addEventListener("click", async () => {
      const txt = document.getElementById("bib-text").textContent;
      try { await navigator.clipboard.writeText(txt); }
      catch (_) {
        const ta = document.createElement("textarea");
        ta.value = txt; document.body.appendChild(ta); ta.select(); document.execCommand("copy"); ta.remove();
      }
      const span = copyBtn.querySelector("span");
      span.textContent = "Copied!";
      setTimeout(() => (span.textContent = "Copy"), 1600);
    });
  }
})();
