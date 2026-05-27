# Changelog

## v1.3.0 — Pure-CSS Mesh Gradient Themes, Zero External Images (2026-05-27)

### Summary
Removed all external image dependencies (Unsplash, loremflickr). Every dataset theme is now a **pure-CSS mesh gradient** — layered `radial-gradient` colour spots over a `linear-gradient` base — crafted specifically for each domain. Zero HTTP requests for the header background; works offline; loads instantly.

---

### Why

| Before | Problem |
|---|---|
| `source.unsplash.com` (v1.1.x) | Shut down — returns HTTP 503 |
| `loremflickr.com` (v1.2.0) | Random CC photos: irrelevant, low-quality, external dependency |

Pure CSS is faster, more reliable, visually on-brand, and never shows an irrelevant stock photo.

### What changed

#### `header_pattern` replaces `img_keywords` + `img_overlay`

Each domain theme now carries a single `header_pattern` key — a CSS `background` property value built from:

```
radial-gradient(ellipse at X% Y%, rgba(R,G,B,α) 0%, transparent N%),   ← colour spot 1
radial-gradient(ellipse at X% Y%, rgba(R,G,B,α) 0%, transparent N%),   ← colour spot 2
radial-gradient(ellipse at X% Y%, rgba(R,G,B,α) 0%, transparent N%),   ← colour spot 3
linear-gradient(Adeg, #colour1 0%, #colour2 50%, #colour3 100%)         ← base tone
```

The radial spots are positioned and coloured to evoke each domain's real-world context:

| Domain | Visual personality |
|---|---|
| 🌸 Botanical | Purple bloom at bottom-left, green canopy top-right |
| 🩺 Health | Dual teal spotlights, clinical depth |
| ✈️ Aviation | White cloud wisps on sky-blue diagonal |
| 💰 Finance | Warm gold corners over deep navy |
| 🚢 Shipping | Ocean depth rising from bottom |
| 🏠 Real Estate | Warm red ember top-right, earth tones |
| 👤 HR | Dual purple spotlights, professional |
| 🍷 Quality | Crimson/wine diagonal spotlight |
| ⚓ Maritime | Navy depth with blue horizon glow |
| 🛒 Retail | Warm orange dual spots |
| ⚡ Energy | Solar yellow top, ember orange base |
| 🚗 Automotive | Cool steel side-lights |
| 🌾 Agriculture | Earth-green rising from horizon |
| 🔒 Cybersecurity | Dark matrix with green pulse glow |
| 🎓 Education | Academic blue diagonal |
| 🏆 Sports | Turf-green stadium lighting |
| 🛡️ Insurance | Navy with electric-blue shield glow |
| 📦 Supply Chain | Amber warmth on dark earth |
| 💬 NLP | Electric purple dual diagonal spots |

#### Generic fallback — unique colour per dataset

Unknown datasets get a `hsla(hue, ...)` radial gradient where `hue` is derived from `hash(column_names)` — so every novel dataset gets a unique, **deterministic** colour that stays the same every time it's run.

#### `_generate_frontend` simplified

Before (14 lines):
```python
try:
    import urllib.parse as _urlparse
    _img_kw = theme.get("img_keywords", "")
    ...
    _img_url = f"https://loremflickr.com/1400/560/{_kw_csv}"
    header_bg = f"{_img_overlay}, url('{_img_url}') center / cover no-repeat"
except Exception:
    header_bg = gradient
```

After (1 line):
```python
header_bg = theme.get("header_pattern", gradient)
```

---

### Files changed

| File | Change |
|---|---|
| `auto_pipeline.py` | `_detect_domain` rewritten with `header_pattern`; `_generate_frontend` simplified |
| `bootstrap.py` | Identical changes inside embedded `FILES["auto_pipeline.py"]` string |
| `docs/changelog.md` | This entry |

---

## v1.2.0 — Real Background Photos + Cold-Start Fix + 19 Domains (2026-05-27)

### Summary
Three improvements: (1) background images now use **loremflickr.com** (replaces the defunct `source.unsplash.com`), so every dataset gets a real, keyword-matched photo; (2) Predict button is locked until the server confirms it is fully ready — fixes the Render cold-start 404 race; (3) four new domains added (Insurance 🛡️, Supply Chain 📦, NLP 💬, plus the prior Sports 🏆).

---

### What changed

#### Background images — switched from Unsplash (defunct) to loremflickr

`source.unsplash.com/featured/` was shut down and returns HTTP 503. Replaced with:

```
https://loremflickr.com/1400/560/{keyword1,keyword2,keyword3,keyword4}
```

loremflickr is free, requires no API key, and returns real Creative-Commons photos matched to the keywords. The first four words from `img_keywords` are used, joined with commas.

#### Cold-start race condition fixed (Render 404)

**Root cause:** Render's free tier spins down after 15 min of inactivity. When it wakes, `/health` responds within seconds (before the ML model finishes loading). Previously, `checkServer()` called `/health` once and immediately enabled the Predict button — so if the user clicked quickly, FastAPI was still loading and returned 404 from Render's load balancer.

**Fix:**
- `setServerState(state, msg)` helper controls dot color + label + button disabled/opacity atomically
- `_healthTimer` polls `/health` every 5 seconds until `{"status":"ok"}` is confirmed
- Predict button stays **greyed out / disabled** (`opacity: 0.5`, `disabled`) until the server is truly ready
- Timer clears itself once online (stops polling)
- If the user submits before ready: friendly message "Server is not ready yet. Please wait..."
- `.finally` block re-checks `_serverReady` instead of unconditionally re-enabling

#### New domains (total: 19)

| Icon | Domain | Trigger keywords (sample) |
|---|---|---|
| 🛡️ Insurance | `insurance`, `premium`, `claim`, `policy`, `coverage`, `deductible` |
| 📦 Supply Chain | `supply`, `inventory`, `demand`, `procurement`, `vendor`, `sku` |
| 💬 NLP | `text`, `sentiment`, `review`, `nlp`, `tweet`, `corpus`, `token` |

---

### Files changed

| File | Change |
|---|---|
| `auto_pipeline.py` | loremflickr URL; 3 new domains; cold-start retry JS |
| `bootstrap.py` | Same changes inside embedded `FILES["auto_pipeline.py"]` string |
| `docs/changelog.md` | This entry |

---

## v1.1.1 — Server Status Indicator & Better Error Handling (2026-05-27)

### Summary
The prediction UI now shows a live **server status bar** (green dot = online, red = offline) and replaces the cryptic `"Failed to fetch"` browser error with a clear, actionable message when the FastAPI server is not running.

---

### Problem fixed
When the uvicorn/FastAPI server was not running, clicking Predict caused the `.catch` handler to display the raw browser error `"Failed to fetch"` (or `"API Error: Unable to connect to API (ConnectionRefused)"` on some browsers) — unhelpful to end users.

### What changed

#### `auto_pipeline.py` and `bootstrap.py` (embedded `FILES["auto_pipeline.py"]`)

Both files updated identically.

##### New `.srv-bar` CSS (inserted before `/* Footer */`)

```css
.srv-bar  { display: flex; align-items: center; gap: 8px; font-size: 0.78rem;
            font-weight: 600; color: #666; margin-bottom: 20px;
            padding: 9px 14px; background: #f8f9fa;
            border-radius: 8px; border: 1px solid #e8eaed; }
.srv-dot  { width: 9px; height: 9px; border-radius: 50%;
            background: #d1d5db; flex-shrink: 0; transition: background 0.3s; }
.srv-dot.online   { background: #22c55e; }
.srv-dot.offline  { background: #ef4444; }
.srv-dot.checking { animation: pulse 1s infinite; }
@keyframes pulse  { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
```

##### New HTML server status bar (inserted at top of `.card`, before the form title)

```html
<div class="srv-bar">
  <span class="srv-dot checking" id="srvDot"></span>
  <span id="srvTxt">Checking server…</span>
</div>
```

##### New `checkServer()` JavaScript (runs on page load)

Pings `GET /health`. Updates the dot and label:
- ✅ HTTP 200 → green dot, "Server online"
- ⚠️ Non-2xx → red dot, "Server error (HTTP NNN)"
- ❌ Network error → red dot, "Server offline — run: uvicorn app:app --reload"

##### Improved `.catch` handler

Before: `errMsg.textContent = err.message;` → shows raw "Failed to fetch"

After: detects `TypeError` / "Failed to fetch" / "NetworkError" / "ERR_CONNECTION_REFUSED" and displays:
```
⚠️ Cannot reach the API server.
Make sure uvicorn is running:
    uvicorn app:app --reload
```
Also sets the server dot to red so both the status bar and the error panel indicate the problem.

---

### Files changed

| File | Change |
|---|---|
| `auto_pipeline.py` | `.srv-bar` CSS + HTML div + `checkServer()` + improved `.catch` |
| `bootstrap.py` | Same changes inside embedded `FILES["auto_pipeline.py"]` string |
| `docs/changelog.md` | This entry |

---

## v1.1.0 — Dynamic Image-Themed Frontend (2026-05-27)

### Summary
The generated `index.html` prediction UI now automatically picks a **background image and color palette** that match the dataset — no hardcoded images or fixed themes.

---

### What changed

#### `auto_pipeline.py` and `bootstrap.py` (embedded `FILES["auto_pipeline.py"]`)

Both files share the same `_detect_domain` and `_generate_frontend` functions. Both were updated identically.

##### `_detect_domain` — expanded from 10 to 16 domains + smart generic fallback

| Domain | Keywords (sample) | Icon | New? |
|---|---|---|---|
| 🌸 Botanical | iris, sepal, petal, setosa, versicolor | 🌸 | ✅ New |
| 🩺 Health | glucose, diabetes, cancer, patient | 🩺 | Updated |
| ✈️ Aviation | flight, airline, airport, departure | ✈️ | Updated |
| 💰 Finance | loan, credit, fraud, stock, revenue | 💰 | Updated |
| 🚢 Shipping | cargo, freight, container, vessel | 🚢 | Updated |
| 🏠 Real Estate | house, sqft, bedroom, property | 🏠 | Updated |
| 👤 HR Analytics | employee, attrition, performance | 👤 | Updated |
| 🍷 Quality | wine, acidity, sulphates, density | 🍷 | Updated |
| ⚓ Maritime | titanic, survived, pclass, embarked | ⚓ | Updated |
| 🛒 Retail | customer, purchase, sales, cart | 🛒 | Updated |
| ⚡ Energy | solar, wind, co2, renewable | ⚡ | Updated |
| 🚗 Automotive | car, vehicle, horsepower, mpg | 🚗 | ✅ New |
| 🌾 Agriculture | crop, soil, rainfall, nitrogen | 🌾 | ✅ New |
| 🔒 Cybersecurity | attack, malware, intrusion, network | 🔒 | ✅ New |
| 🎓 Education | student, grade, exam, gpa | 🎓 | ✅ New |
| 🏆 Sports | game, team, win, league, score | 🏆 | ✅ New |

Each domain now has two new fields:
- **`img_keywords`** — search terms sent to Unsplash (e.g. `"iris flowers purple nature garden botanical"`)
- **`img_overlay`** — semi-transparent RGBA gradient layered on top of the photo so text is always readable

##### Generic fallback — now derives keywords from actual column names

Before: always showed the same generic blue/robot theme.

After: extracts meaningful words from the dataset filename, project name, and column names (stripping stop-words and short tokens). For example:
- `cars_data.csv` + columns `["make", "model", "mileage"]` → query `"make model mileage"` → relevant car photos
- `dataset.csv` + columns `["x1", "x2", "y"]` → query `"dataset experiment"` → abstract/data imagery

##### `_generate_frontend` — live Unsplash background image

New code block added after theme extraction:

```python
import urllib.parse as _urlparse
_img_kw      = theme.get("img_keywords", "")
_img_overlay = theme.get("img_overlay", gradient)
if _img_kw:
    _encoded  = _urlparse.quote(_img_kw)
    _img_url  = f"https://source.unsplash.com/featured/1400x560?{_encoded}"
    header_bg = f"{_img_overlay}, url('{_img_url}') center / cover no-repeat"
else:
    header_bg = gradient
```

The Unsplash Source endpoint (`source.unsplash.com/featured/...`) requires no API key and returns contextually relevant photos for any keyword query.

##### CSS changes in generated `index.html`

| Element | Before | After |
|---|---|---|
| `.hdr background` | `TMPL_GRADIENT` (solid gradient) | `TMPL_HEADER_BG` (overlay + photo URL) |
| `.hdr` | No `background-size`, no `position` | `background-size: cover; background-position: center; position: relative` |
| `.hdr h1` | No text shadow | `text-shadow: 0 2px 8px rgba(0,0,0,0.35)` |
| `.hdr p` | `opacity: 0.82` | `opacity: 0.90; text-shadow: 0 1px 4px rgba(0,0,0,0.30)` |
| Placeholder | `TMPL_GRADIENT` | `TMPL_HEADER_BG` |

---

### Examples

| Dataset | Theme | Background image query |
|---|---|---|
| `Iris.csv` | 🌸 Botanical (`#4a1942` purple / `#f3e5f5` lavender) | `iris flowers purple nature garden botanical` |
| `titanic.csv` | ⚓ Maritime (`#0d47a1` navy / `#e3f2fd` sky blue) | `ocean ship sea storm waves dramatic` |
| `diabetes.csv` | 🩺 Health (`#0d6e6e` teal / `#f0faf9` mint) | `hospital medical doctor healthcare stethoscope` |
| `cars_data.csv` | 🚗 Automotive (`#37474f` slate / `#eceff1` light grey) | `car automobile vehicle road speed` |
| `housing.csv` | 🏠 Real Estate (`#5d4037` brown / `#fbe9e7` warm cream) | `house suburb real estate property architecture` |
| `mystery.csv` (no match) | 🤖 ML (derived) | keywords extracted from column names |

---

### Files changed

| File | Change |
|---|---|
| `auto_pipeline.py` | `_detect_domain` expanded; `_generate_frontend` updated |
| `bootstrap.py` | Same changes inside embedded `FILES["auto_pipeline.py"]` string |
| `docs/changelog.md` | This file (new) |
