(function () {
  if (window.__authFetchInstalled) return;
  window.__authFetchInstalled = true;

  const CSRF_COOKIE = 'csrftoken';
  const CSRF_HEADER = 'X-CSRF-Token';
  const STATE_CHANGING = new Set(['POST', 'PUT', 'PATCH', 'DELETE']);

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

  function methodOf(input, init) {
    if (init && init.method) return String(init.method).toUpperCase();
    if (input && typeof input === 'object' && input.method) {
      return String(input.method).toUpperCase();
    }
    return 'GET';
  }

  function isSameOriginApi(url) {
    try {
      const u = new URL(url, window.location.origin);
      return u.origin === window.location.origin && u.pathname.startsWith('/api/');
    } catch (e) {
      return typeof url === 'string' && url.startsWith('/api/');
    }
  }

  function readCookie(name) {
    const parts = document.cookie ? document.cookie.split(';') : [];
    const prefix = name + '=';
    for (let i = 0; i < parts.length; i++) {
      const c = parts[i].trim();
      if (c.indexOf(prefix) === 0) {
        try { return decodeURIComponent(c.substring(prefix.length)); }
        catch (e) { return c.substring(prefix.length); }
      }
    }
    return '';
  }

  // Returns a (possibly-new) init object with the CSRF header injected when
  // appropriate. Never mutates the caller's init or Request object.
  function withCsrf(input, init) {
    const url = urlOf(input);
    if (!isSameOriginApi(url)) return init;
    const method = methodOf(input, init);
    if (!STATE_CHANGING.has(method)) return init;
    const token = readCookie(CSRF_COOKIE);
    if (!token) return init;

    // Merge into a Headers instance so we don't trample caller-set headers,
    // but don't override an explicitly-set CSRF header.
    const baseHeaders = (init && init.headers)
      || (input && typeof input === 'object' && input.headers)
      || {};
    const headers = new Headers(baseHeaders);
    if (headers.has(CSRF_HEADER)) return init;
    headers.set(CSRF_HEADER, token);

    return Object.assign({}, init || {}, { headers });
  }

  // Prime the csrftoken cookie via a safe request. Safe-method responses on an
  // authenticated session will mint the cookie if missing.
  let primingPromise = null;
  function primeCsrf() {
    if (primingPromise) return primingPromise;
    primingPromise = originalFetch('/api/auth/me', {
      method: 'GET',
      credentials: 'same-origin',
      headers: { 'Accept': 'application/json' },
    }).catch(() => null).finally(() => { primingPromise = null; });
    return primingPromise;
  }

  // Decide whether the original (input, init) pair can be safely re-fetched.
  // A Request whose body has already been read, or any stream-backed body,
  // is not replayable — retrying would throw "body stream already read".
  function isReplayable(input, init) {
    if (typeof Request !== 'undefined' && input instanceof Request) {
      if (input.bodyUsed) return false;
      // Streams from a Request can only be consumed once even if not yet read.
      if (input.body && typeof ReadableStream !== 'undefined' && input.body instanceof ReadableStream) {
        return false;
      }
    }
    const body = init && init.body;
    if (body && typeof ReadableStream !== 'undefined' && body instanceof ReadableStream) {
      return false;
    }
    return true;
  }

  async function isCsrfFailure(res) {
    if (res.status !== 403) return false;
    try {
      const body = await res.clone().json();
      return !!(body && (
        body.code === 'csrf_invalid'
        || (typeof body.detail === 'string' && /CSRF/i.test(body.detail))
      ));
    } catch (e) {
      return false;
    }
  }

  window.fetch = async function patchedFetch(input, init) {
    let res = await originalFetch(input, withCsrf(input, init));

    // One-shot retry for CSRF failures: this handles the bootstrap race where
    // the page loaded before a csrftoken cookie was issued, and any case where
    // the cookie was dropped/expired while the session token is still valid.
    if (
      isSameOriginApi(urlOf(input))
      && STATE_CHANGING.has(methodOf(input, init))
      && isReplayable(input, init)
      && await isCsrfFailure(res)
    ) {
      await primeCsrf();
      // Note: we re-derive the init from the original arguments so the retry
      // picks up the freshly-set cookie value.
      res = await originalFetch(input, withCsrf(input, init));
    }

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
