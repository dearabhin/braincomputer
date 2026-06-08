# 🚀 Deploying BrainComputer.in — beginner step-by-step guide

This walks you through putting the whole project online, from zero. Follow it **in order**.
Every command is copy-paste; wherever you must change something it's written in `THIS_STYLE`.

## The big picture (read this once)

There are **three** pieces, hosted in three different places:

| Piece | Lives on | Cost |
|------|----------|------|
| **Frontend** (the website) | Cloudflare Pages | Free |
| **Backend API** (Flask) | A small VPS (DigitalOcean droplet) | ~$6/mo |
| **GPU model** (TRIBE v2) | Modal's cloud (serverless GPU) | Pay-per-use (cents/run) |

> ⚠️ **The GPU does NOT run on your VPS.** The $6 droplet is too weak to run TRIBE v2. The droplet
> only *forwards* uploads to Modal, which runs the model on a real GPU and sends scores back.

**The journey of one upload:**
```
You → braincomputer.in (Cloudflare Pages) → api.braincomputer.in (your droplet, Flask)
        → Modal (GPU runs TRIBE v2) → scores come back → shown on the website
```

### What you'll need (create these free accounts first)
- [ ] A **Hugging Face** account → https://huggingface.co
- [ ] A **Modal** account → https://modal.com
- [ ] A **DigitalOcean** account → https://digitalocean.com (or Azure — steps are similar)
- [ ] A **GitHub** account → https://github.com (we'll push the code here)
- [ ] Your **Cloudflare** account with `braincomputer.in` already added ✅ (you've done this)
- [ ] Your project code (this folder) on your laptop

You'll also install three small tools on your **laptop** along the way: `git`, the `modal` CLI, and
optionally `wrangler`. Let's go.

---

## Part 0 — Put your code on GitHub

Both Cloudflare Pages and your droplet pull the code from GitHub, so do this first.

On your **laptop**, in the project folder:

```bash
# install git if you don't have it: https://git-scm.com/downloads
git init
git add .
git commit -m "Initial commit: BrainComputer.in"
```

Create a new **empty** repo on GitHub (no README), call it `braincomputer`, then:

```bash
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/braincomputer.git
git branch -M main
git push -u origin main
```

✅ Your code is now on GitHub. (The `.gitignore` already keeps secrets/`.env` out — good.)

---

## Part 1 — Hugging Face: get access to Llama 3.2 + a token

TRIBE v2 uses Llama 3.2 *inside* it as a frozen text encoder. It's **free**, but "gated" — you click
to accept the license once.

1. Log in to Hugging Face. Go to **https://huggingface.co/meta-llama/Llama-3.2-3B**.
2. Click **"Agree and access repository"**. Fill the short form. Approval is usually instant.
3. Create an access token: profile → **Settings → Access Tokens → Create new token**.
   - Type: **Read**. Name it `braincomputer`.
   - **Copy the token** (starts with `hf_...`). You'll paste it into Modal next. Keep it secret.

✅ You have an `hf_...` token and Llama 3.2 access.

---

## Part 2 — Modal: deploy the GPU worker

Modal runs TRIBE v2 on a GPU only when needed (scales to zero, so it's cheap).

### 2.1 Install the Modal CLI (on your laptop)
```bash
pip install modal
modal token new      # opens a browser to log in / create your token
```
✅ `modal token new` finishes with "Token written to ~/.modal.toml".

### 2.2 Give Modal your Hugging Face token (as a secret)
```bash
modal secret create huggingface HF_TOKEN=hf_PASTE_YOUR_TOKEN_HERE
```
✅ Run `modal secret list` — you should see `huggingface`.

### 2.3 Deploy the worker

> Important: run this **from inside the `inference/` folder** so Modal can find the
> `networks.py` / `scoring.py` / `insights.py` / `preprocess.py` files it bundles.

```bash
cd inference
modal deploy modal_app.py
```
The first deploy builds the image (installs TRIBE v2 + deps) — this takes several minutes. ✅ When it
finishes you'll see a URL/app name `braincomputer-tribe` and "App deployed".

### 2.4 Smoke-test it with a real clip (optional but recommended)
Put any short `.mp4` in the `inference/` folder, then:
```bash
modal run modal_app.py --input your_clip.mp4
```
The **first run is slow** (downloads multi-GB model weights into a cache, once). ✅ It prints a JSON
result with `engagement_index`, `networks`, etc. Future runs are much faster.

```bash
cd ..   # back to the project root
```

### 2.5 Make a Modal token for your server
Your droplet needs to call Modal too. In the Modal dashboard:
**Settings → API Tokens → New Token**. Copy the **Token ID** (`ak-...`) and **Token Secret** (`as-...`).
Save them for Part 3.

---

## Part 3 — Backend: the DigitalOcean droplet

### 3.1 Create the droplet
1. DigitalOcean → **Create → Droplets**.
2. Region: closest to you. Image: **Ubuntu 24.04 (LTS)**.
3. Size: **Basic → Regular → $6/mo** (1 GB RAM is plenty — no GPU needed).
4. Authentication: **SSH key** (recommended) or Password. If unsure, choose Password and set a strong one.
5. Hostname: `braincomputer-api`. Click **Create Droplet**.
6. Copy the droplet's **public IPv4 address** (e.g. `203.0.113.10`). You'll need it twice.

### 3.2 Connect to it
On your laptop:
```bash
ssh root@YOUR_DROPLET_IP
```
(Type `yes` if asked about authenticity. Use the password if you chose that.)
✅ You're now on the server (prompt looks like `root@braincomputer-api:~#`).

### 3.3 Install Docker (run these ON the droplet)
```bash
apt update && apt upgrade -y
curl -fsSL https://get.docker.com | sh
```
✅ Check it: `docker --version` and `docker compose version` both print versions.

### 3.4 Open the firewall
```bash
ufw allow OpenSSH
ufw allow 80
ufw allow 443
ufw --force enable
```
✅ `ufw status` shows 80, 443, and OpenSSH allowed.

### 3.5 Get the code onto the droplet
```bash
apt install -y git
git clone https://github.com/YOUR_GITHUB_USERNAME/braincomputer.git
cd braincomputer
```

### 3.6 Create the backend `.env` file
This holds the secrets the API needs. Create it with `nano`:
```bash
nano .env
```
Paste this, replacing the Modal token values from step 2.5:
```ini
MOCK_INFERENCE=0
MODAL_TOKEN_ID=ak-PASTE_YOURS
MODAL_TOKEN_SECRET=as-PASTE_YOURS
ALLOWED_ORIGINS=https://braincomputer.in,https://www.braincomputer.in
```
Save and exit nano: **Ctrl+O**, **Enter**, then **Ctrl+X**.

> Storage note: with no R2 settings, uploads/results are stored on the droplet's disk (fine to start).
> To use Cloudflare R2 later, add the `R2_*` lines from `backend/.env.example` to this file.

### 3.7 Point the Caddyfile at your domain
The included `Caddyfile` already uses `api.braincomputer.in`. If your domain differs, edit it:
```bash
nano Caddyfile     # change the hostname if needed, then Ctrl+O / Enter / Ctrl+X
```

### 3.8 Start everything
```bash
docker compose up -d --build
```
First build takes a few minutes. ✅ Check status:
```bash
docker compose ps          # both "api" and "caddy" should be "running"
docker compose logs -f     # watch logs; Ctrl+C to stop watching
```
HTTPS won't work yet because DNS isn't pointed here — that's Part 4.

---

## Part 4 — Cloudflare DNS: point `api` at the droplet

In the **Cloudflare dashboard → braincomputer.in → DNS → Records → Add record**:

| Type | Name | Content (value) | Proxy status |
|------|------|-----------------|--------------|
| **A** | `api` | `YOUR_DROPLET_IP` | **DNS only** (grey cloud ☁️) |

> Why grey cloud? Caddy gets its free HTTPS certificate from Let's Encrypt, which needs to reach your
> server directly. The orange cloud (proxy) would block that. Grey cloud = "DNS only". Click the orange
> cloud icon to turn it grey before saving.

Save it. DNS takes a few minutes to propagate. Test from your laptop:
```bash
ping api.braincomputer.in        # should show YOUR_DROPLET_IP
```

Now Caddy can get its certificate. On the droplet, watch the logs:
```bash
docker compose logs -f caddy     # look for "certificate obtained successfully"
```
Then test the API over HTTPS (from your laptop):
```bash
curl https://api.braincomputer.in/healthz
```
✅ You should see `{"mock":false,"ok":true}`. Your backend is live and secure!

---

## Part 5 — Frontend: Cloudflare Pages

### 5.1 Create the Pages project
1. Cloudflare dashboard → **Workers & Pages → Create → Pages → Connect to Git**.
2. Authorize GitHub and pick your **`braincomputer`** repo.
3. Build settings:
   - **Framework preset:** None
   - **Build command:** *(leave empty)*
   - **Build output directory:** `frontend`
4. Click **Save and Deploy**. ✅ After ~1 min you get a URL like `braincomputer.pages.dev`.

### 5.2 Tell the frontend where your API is
The frontend defaults to `https://api.braincomputer.in` in production, which matches Part 4 — so if you
kept that hostname, **you're already done**. (If you used a different API hostname, edit the
`<meta name="bc-api-base" ...>` tag near the top of `frontend/index.html` and `frontend/results.html`,
uncomment it, set your URL, then `git commit` + `git push` — Pages auto-redeploys.)

### 5.3 Connect your real domain to Pages
In your Pages project → **Custom domains → Set up a custom domain**:
1. Add `braincomputer.in` → Cloudflare auto-creates the DNS record (leave it proxied/orange). Activate.
2. Add `www.braincomputer.in` the same way.

Wait a couple minutes. ✅ Visit **https://braincomputer.in** — the BrainComputer site loads.

---

## Part 6 — Test the whole thing end-to-end

1. Open **https://braincomputer.in**.
2. Drag in a short `.mp4` reel and click **Analyze engagement →**.
3. You're taken to the results page with a spinner. The **first real analysis is slow** (Modal cold
   start + weight load — up to a minute or two). Later ones are quick.
4. ✅ You should see the engagement index ring, the six brain-network bars, the timeline, and insight
   cards.

If it works — **congratulations, BrainComputer.in is live!** 🎉

---

## Updating your site later

Whenever you change code on your laptop:
```bash
git add . && git commit -m "describe your change" && git push
```
- **Frontend** redeploys **automatically** (Cloudflare Pages watches GitHub).
- **Backend** — pull + rebuild on the droplet:
  ```bash
  ssh root@YOUR_DROPLET_IP
  cd braincomputer && git pull && docker compose up -d --build
  ```
- **Inference** — redeploy from your laptop:
  ```bash
  cd inference && modal deploy modal_app.py && cd ..
  ```

---

## Troubleshooting

**`curl .../healthz` fails or times out**
- `docker compose ps` — are both containers "running"? `docker compose logs api` for errors.
- DNS not ready: `ping api.braincomputer.in` must return your droplet IP. Wait a few minutes.
- Firewall: `ufw status` must allow 80 and 443.

**Caddy won't get a certificate** (`logs caddy` shows challenge errors)
- The `api` DNS record **must be grey cloud (DNS only)**, not orange. Fix it in Cloudflare and run
  `docker compose restart caddy`.

**Website loads but uploads fail / CORS error in browser console**
- `ALLOWED_ORIGINS` in your droplet `.env` must include `https://braincomputer.in`. After editing:
  `docker compose up -d` (recreates the container with new env).
- Confirm the frontend is calling the right API: it should hit `https://api.braincomputer.in`.

**Analysis errors out (`status: error`)**
- Usually Modal: check `modal app list` shows `braincomputer-tribe` deployed, and the droplet `.env`
  has correct `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`. See logs: `docker compose logs api`.
- Hugging Face/Llama access: re-confirm Part 1 and that `modal secret list` shows `huggingface`.

**`modal run` fails with a dependency/import error** (e.g. `AttributeError: module 'exca.steps.base' has
no attribute 'NoValue'`, or `Could not import module 'AutoProcessor'`)
- TRIBE leaves a couple of transitive dependencies unpinned, so pip grabs versions newer than TRIBE was
  built against. The two known ones are already pinned in `inference/modal_app.py`:
  `exca==0.5.20` and `transformers==5.3.0` (the versions current at TRIBE's 2026-03-25 release).
- If a *different* package throws a version/import error after a rebuild, pin it the same way: find the
  version released around 2026-03-25, add `"package==X.Y.Z"` to the `.pip_install(...)` list, and re-run.

**Want to test the site without spending GPU money?**
- Set `MOCK_INFERENCE=1` in the droplet `.env` and `docker compose up -d`. Uploads return fake (but
  realistic-looking) results instantly. Switch back to `0` for real analysis.

---

## Costs recap

| Item | Cost |
|------|------|
| Cloudflare Pages + DNS | $0 |
| DigitalOcean droplet (1 GB) | ~$6 / month |
| Modal GPU | A few **cents per analysis**; $0 when idle. Free monthly credits cover light use. |
| Cloudflare R2 (optional storage) | Free tier (generous) |

Your ~$100 budget comfortably covers a portfolio-scale launch for many months.

---

## ⚖️ Reminder
TRIBE v2 is **CC-BY-NC-4.0 (non-commercial)**. Keep BrainComputer.in free and the "predicted brain
activity, not a guarantee of reach" disclaimer visible. Don't add paid tiers without written permission
from Meta.
