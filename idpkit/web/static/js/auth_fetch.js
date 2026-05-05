(function () {
  if (window.__authFetchInstalled) return;
  window.__authFetchInstalled = true;

  function ensureToast() {
    let el = document.getElementById('__auth_session_toast');
    if (el) return el;
    el = document.createElement('div');
    el.id = '__auth_session_toast';
    el.setAttribute('role', 'status');
    el.setAttribute('aria-live', 'polite');
    el.style.cssText = [
      'position:fixed', 'top:20px', 'right:20px',
      'background:#1f2937', 'color:#fff',
      'padding:12px 18px', 'border-radius:8px',
      'font-size:14px', 'font-weight:500',
      'z-index:99999',
      'box-shadow:0 6px 20px rgba(0,0,0,0.25)',
      'opacity:0', 'transition:opacity 200ms',
      'pointer-events:none', 'max-width:340px',
    ].join(';');
    document.body.appendChild(el);
    return el;
  }

  function showToast(msg) {
    const el = ensureToast();
    el.textContent = msg;
    requestAnimationFrame(() => { el.style.opacity = '1'; });
  }

  const originalFetch = window.fetch.bind(window);
  let redirecting = false;

  function urlOf(input) {
    if (typeof input === 'string') return input;
    if (input && typeof input.url === 'string') return input.url;
    try { return String(input); } catch (e) { return ''; }
  }

  window.fetch = async function patchedFetch(input, init) {
    const res = await originalFetch(input, init);
    if (res.status === 401 && !redirecting) {
      const url = urlOf(input);
      const path = window.location.pathname;
      const isAuthEndpoint = /\/api\/auth\/(login|register|logout)/.test(url);
      const onAuthPage = path === '/login' || path === '/register';
      if (!isAuthEndpoint && !onAuthPage) {
        redirecting = true;
        showToast('Your session expired — redirecting to sign in…');
        const next = encodeURIComponent(path + window.location.search);
        setTimeout(() => { window.location.href = '/login?next=' + next; }, 900);
      }
    }
    return res;
  };

  // Backwards-compatible alias for templates that prefer an explicit name.
  window.authFetch = window.fetch;
})();
