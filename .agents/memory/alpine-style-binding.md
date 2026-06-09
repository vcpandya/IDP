---
name: Alpine :style string overwrites static style
description: Alpine v3 string-form :style wipes a coexisting static style attribute; use object form to merge.
---

# Alpine `:style` string vs object form

In Alpine v3 (confirmed in 3.14.8 CDN source), the two `:style` forms behave differently when the element also has a static `style="..."` attribute:

- **String form** `:style="'width:' + pct + '%'"` → internally `el.setAttribute("style", value)`, which **OVERWRITES the entire style attribute**, wiping any coexisting static inline styles.
- **Object form** `:style="{ width: pct + '%' }"` → internally `el.style.setProperty(...)`, which **merges** with existing inline styles.

**Why:** This caused a progress-bar bug where `<div style="height:100%;background:#7c3aed" :style="'width:'+pct+'%'">` rendered with the right width but lost its height/background, so the bar never visually filled. Fast-finishing bars (uploads) hid the same bug; a slow reprocess job made it obvious.

**How to apply:** Whenever an element has BOTH a static `style="..."` attribute AND an Alpine `:style` binding, use the **object form** so the static styles survive. This also affects tiny `:style="'background:'+color"` color dots that carry static `width/height` — they'd render invisible with string form.
