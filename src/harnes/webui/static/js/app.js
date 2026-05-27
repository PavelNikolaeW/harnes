/* Минимальный helper: copy-to-clipboard на любом [data-copy="..."] элементе. */
document.addEventListener('click', (ev) => {
  const t = ev.target.closest('[data-copy]');
  if (!t) return;
  navigator.clipboard.writeText(t.dataset.copy).then(() => {
    const orig = t.textContent;
    t.textContent = 'copied';
    setTimeout(() => { t.textContent = orig; }, 900);
  }).catch(() => {});
});
