(function () {
  if (window.__modalA11yInstalled) return;
  window.__modalA11yInstalled = true;

  const FOCUSABLE = [
    'a[href]',
    'button:not([disabled])',
    'input:not([disabled]):not([type="hidden"])',
    'select:not([disabled])',
    'textarea:not([disabled])',
    '[tabindex]:not([tabindex="-1"])',
  ].join(',');

  const MODAL_SELECTORS = [
    '.modal-overlay',
    '.source-modal-overlay',
    '.attach-modal-overlay',
    '.mention-popup-overlay',
  ].join(',');

  const state = new WeakMap();

  function isVisible(el) {
    if (!el || !el.isConnected) return false;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') return false;
    if (el.hasAttribute('hidden')) return false;
    return true;
  }

  function focusables(root) {
    return Array.from(root.querySelectorAll(FOCUSABLE))
      .filter((n) => !n.disabled && n.offsetParent !== null);
  }

  function open(el) {
    const st = state.get(el) || {};
    if (st.open) return;
    const prevTrigger = document.activeElement;
    const handler = (e) => {
      if (e.key !== 'Tab') return;
      const fs = focusables(el);
      if (fs.length === 0) { e.preventDefault(); return; }
      const first = fs[0], last = fs[fs.length - 1];
      const active = document.activeElement;
      if (e.shiftKey && (active === first || !el.contains(active))) {
        last.focus(); e.preventDefault();
      } else if (!e.shiftKey && active === last) {
        first.focus(); e.preventDefault();
      }
    };
    el.addEventListener('keydown', handler);
    state.set(el, { open: true, prevTrigger, handler });
    // Defer to allow Alpine transitions to settle.
    setTimeout(() => {
      const fs = focusables(el);
      if (fs.length) {
        // Skip the close ('×') button if a more meaningful target exists.
        const target = fs.find((n) => !n.classList.contains('modal-close')
          && !n.classList.contains('btn-close')
          && !n.classList.contains('mention-popup-close')) || fs[0];
        try { target.focus(); } catch (e) { /* ignore */ }
      }
    }, 30);
  }

  function close(el) {
    const st = state.get(el);
    if (!st || !st.open) return;
    el.removeEventListener('keydown', st.handler);
    state.set(el, { open: false });
    if (st.prevTrigger && typeof st.prevTrigger.focus === 'function'
        && document.contains(st.prevTrigger)) {
      try { st.prevTrigger.focus(); } catch (e) { /* ignore */ }
    }
  }

  function attach(el) {
    if (el.__a11yAttached) return;
    el.__a11yAttached = true;
    if (!el.hasAttribute('role')) el.setAttribute('role', 'dialog');
    el.setAttribute('aria-modal', 'true');
    const obs = new MutationObserver(() => {
      if (isVisible(el)) open(el); else close(el);
    });
    obs.observe(el, { attributes: true, attributeFilter: ['style', 'class', 'hidden'] });
    if (isVisible(el)) open(el);
  }

  function scan(root) {
    (root || document).querySelectorAll(MODAL_SELECTORS).forEach(attach);
  }

  function init() {
    scan(document);
    const obs = new MutationObserver((muts) => {
      for (const m of muts) {
        m.addedNodes && m.addedNodes.forEach((n) => {
          if (n.nodeType === 1) {
            if (n.matches && n.matches(MODAL_SELECTORS)) attach(n);
            scan(n);
          }
        });
      }
    });
    obs.observe(document.body, { childList: true, subtree: true });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
