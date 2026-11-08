# PromptForge

Paste a prompt, get a shorter one that says the same thing, and see the token counts before and after. PromptForge also builds prompts from a project description and writes a handoff prompt when you have to move a project to another AI.

Tokens are counted with OpenAI's tokenizer (tiktoken, o200k_base). Gemini does the rewriting.

## The three tabs

**Prompt Shortener.** Two compression levels:
- Normal removes filler, greetings and repeated points and keeps every idea and detail.
- Aggressive goes further, but only by cutting repetition, merging instructions that overlap and using shorter wording. It never drops an idea, requirement, number, name or piece of code, and it keeps full readable sentences.

Three modes: English only, Mixed languages (short German, French, Portuguese and Spanish words and phrases replace English sentences when that costs fewer tokens, with nothing lost), or Compare both. After a run you can test it live: the page asks Gemini to answer both the original and the shortened prompt and compares the answers.

**Prompt Maker.** Describe the idea, frontend, backend and security in four boxes and get one short context prompt plus a prompt for each part.

**Handoff.** Add your project files and say what you were doing. A parser reads the files (languages, modules, routes, fields, mismatches between frontend and backend) and Gemini writes the human part. You get one prompt for the next AI. Nothing is generated as files, it only writes the prompt. Files stay in the page until you build.

Also: history with search, favourites and tags, a stats page, a language token test, a colour picker, and a liquid morph animation in the bottom right while Gemini is working.

## Run it

Needs Python 3.10 or newer and an internet connection.

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # Windows: copy .env.example .env
```

Open `.env` and paste your Gemini key after `GEMINI_API_KEY=` (free key: https://aistudio.google.com/apikey).

```
python app.py
```

Then open http://127.0.0.1:8000. Do not double-click `static/index.html`, the page needs the server.

The first start downloads the tokenizer file once, so it takes a few extra seconds.

## Folder

```
app.py               the server: endpoints, Gemini calls, token counting, database
features.py          levels, mixed-language prompt, live test, stats, handoff route
handoff.py           the code reader behind the Handoff tab
static/index.html    the whole frontend
static/bg/           the three tab background pictures
psts.db              created on first run, stores your history
.env                 your key (never commit this)
```

`features.py` is attached to `app.py` with two lines near the bottom, after the routes are defined:

```
import features
features.register(globals())
```

## Endpoints

| Method | Path          | What it does |
|--------|---------------|--------------|
| GET    | /             | the website |
| GET    | /health       | is the key set, which model, which token counter |
| POST   | /shorten      | body `{prompt, mode, model, level, save}`. mode is english, mixed or both. level is medium (Normal) or aggressive |
| POST   | /count        | body `{text}`, exact OpenAI token count for the live counter |
| POST   | /build        | body `{idea, frontend, backend, security, checklist, model}` |
| POST   | /handoff      | body `{files, goal, notes, model}`, returns the handoff prompt |
| POST   | /testlive     | body `{original, shortened, model}`, answers both and compares |
| GET    | /history      | the last 100 runs |
| GET    | /history/meta | favourites and tags |
| PATCH  | /history/<id> | set favourite or tags |
| DELETE | /history      | clears the history |
| GET    | /stats        | totals and tokens saved over time |
| POST   | /experiment   | body `{sentences, fresh}`, counts tokens in 5 languages |

## How the output is kept honest

Gemini is told to answer in a fixed JSON shape, so it cannot add chatter around the rewrite. Then the server checks the answer itself:

- A rewrite only counts if it uses fewer tokens than your original. If it does not, the server asks once more, and if that fails too, your original is shown unchanged.
- Numbers, links, emails, quoted text and code from your prompt must still be in the rewrite. If one is missing the server asks again and flags it on the page if it is still missing.
- Mixed languages: Gemini writes 3 candidates, the server counts each one with the OpenAI tokenizer and shows the cheapest that passes. A second Gemini call checks that each candidate kept every idea, and candidates that lost meaning are dropped or retried. If none beats English, Gemini is shown the real counts and tries again (up to 3 rounds). The mixed section always shows a mixed-language result, never the English one. If every candidate loses a number, link or piece of code, the one that lost the least is shown and the missing items are flagged.

The server cannot prove the meaning is identical. Read the shortened prompt once before you use it.

Each run uses 1 to 7 Gemini requests. The app only uses gemini-3.8-flash, and switches to gemini-3.6-flash when 3.8 is busy. The English rewrite is also checked for lost meaning and retried once if something went missing.

## Backup AIs

Gemini 3.8 runs first and Gemini 3.6 is next. If both fail, PromptForge tries the backup AIs that have a key in `.env`, in this order: Groq, Cerebras, OpenRouter, Mistral. You only need one. Free keys:

- Groq: https://console.groq.com/keys
- Cerebras: https://cloud.cerebras.ai
- OpenRouter: https://openrouter.ai/keys
- Mistral: https://console.mistral.ai

Add the key to `.env` (for example `GROQ_API_KEY=...`) and restart. To check that a backup works, open http://127.0.0.1:8000/backup-check. It tests each backup and shows the exact error if one fails. If a backup's default model has been retired, PromptForge asks the provider which models are live and picks one automatically. The result says which model answered. Each backup has a default model (see `BACKUPS` in `features.py`), and you can change it with `GROQ_MODEL`, `CEREBRAS_MODEL`, `OPENROUTER_MODEL` or `MISTRAL_MODEL` in `.env`. Free limits and model names change, so check the provider's page if one stops working. Backups are not in the model dropdown, they only step in when Gemini fails.

## Look and feel

- Each tab has its own background picture, cross-faded as you switch.
- Font is Quicksand (loaded from Google Fonts), with the system font as a fallback.
- The main buttons, tab switch and small inputs use liquid glass. In Chrome, Edge and Brave it is the real refraction effect from https://kube.io/blog/liquid-glass-css-svg/ (an SVG displacement filter). Safari and Firefox get a frosted blur instead.
- Page titles and the logo are drawn as dots.

## Privacy

Your prompts go to Gemini to be rewritten. History is stored only in `psts.db` on your machine, and you can turn saving off or clear it from the sidebar.
