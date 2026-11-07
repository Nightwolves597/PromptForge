# Prompt forge

Paste a prompt, get a shorter one with the same meaning, and see the token counts before and after.
Tokens are counted with OpenAI's tokenizer (tiktoken, o200k_base). Gemini does the shortening.

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

Then open http://127.0.0.1:8000 in your browser. Do not open `static/index.html` by double-clicking it, because the page needs the server.

The first start downloads the tokenizer file once, so it takes a few extra seconds.

## Folder

```
app.py            the server: endpoints, Gemini calls, token counting, database
static/index.html the whole frontend
psts.db           created on first run, stores your history
.env              your key (never commit this)
```

## Endpoints

| Method | Path          | What it does                                              |
|--------|---------------|-----------------------------------------------------------|
| GET    | /             | the website                                               |
| GET    | /health       | is the key set, which model, which token counter          |
| POST   | /shorten      | body `{prompt, mode, model}`, mode is english, mixed or both |
| POST   | /count        | body `{text}`, exact OpenAI token count (the live counter) |
| POST   | /build        | body `{idea, frontend, backend, security, checklist, model}`, returns a context prompt and task prompts |
| GET    | /history      | the last 100 runs                                         |
| DELETE | /history      | clears the history                                        |
| POST   | /experiment   | body `{sentences, fresh}`, counts tokens in 5 languages   |

## How the output is kept honest

Gemini is told to answer in a fixed JSON shape, so it cannot add chatter around the rewrite. Then the server checks the answer itself:

- A rewrite only counts if it uses fewer tokens than your original. If it does not, the server asks once more, and if that fails too, your original is shown unchanged.
- Numbers, links, emails, quoted text and code from your prompt must still be in the rewrite. If one is missing the server asks again, and flags the item on the page if it is still missing.
- Mixed languages: Gemini writes 3 mixed-language candidates. The server counts each with the OpenAI tokenizer and shows the cheapest. If none beats English, Gemini is shown the real counts and tries again (up to 3 rounds). The result is always shown with its token difference against English. If every candidate loses a number, link or code, the English version is shown instead.

The server cannot prove the meaning is identical. Read the shortened prompt once before you use it.

Each run uses 1 to 5 Gemini requests: the English rewrite (plus one retry if needed) and up to 3 rounds of mixed candidates.

## Builder tab

Fill in up to four boxes: idea and plan, frontend, backend, security. Gemini writes one short context prompt (paste it once at the start of a Claude chat) and one lean task prompt for each box you filled in. The security prompt asks Claude to review your app and, if the checklist box is ticked, to cover the common web risks (injection, XSS, broken login, CSRF, secrets, rate limits, headers).

The server counts every prompt with the OpenAI tokenizer and checks that the libraries, numbers, names and links from your notes are still in them. If something is missing it asks Gemini once more, and flags it on the page if it is still missing. Your four boxes are remembered in your browser. One run uses 1 or 2 Gemini requests.

Box E and its checkbox write a global prompt: standing instructions you save once in Claude's custom instructions or a Project's instructions, so every later chat follows them. It is sent with every message, so it is kept short and its token cost is shown. It also works on its own, with the other boxes empty.

## Using the page

- **Prompt Shortener** (top centre, left tab): paste a prompt, pick English only, Mixed languages or Compare both, and press Shorten. Token counts come from the OpenAI tokenizer.
- **Prompt Maker** (top centre, right tab): fill in the idea, frontend, backend and security boxes and get one short prompt per part to paste into Claude.
- **History** is the panel on the left. Click a run to load its prompt again. On a phone it opens from the History button.
- **Language test**, **Terms** and **Privacy** are linked at the bottom of the History panel.

## New in this version

- **Compression level** slider on the Shortener: light, medium or aggressive. Medium uses your prompts exactly as they are in `app.py`.
- **Test it live**: runs the original and the shortened prompt through Gemini and compares the two answers (three requests).
- **History** has search, favourites and tags (under Edit on each run). **Hide** removes the panel, and **Save new runs** really stops saving. The Colour button in the top bar picks the accent colour and switches between Night sky (the default, with stars and a shooting star) and Paper (the light look).
- **Stats** page: totals, a 14 day chart and what you use most.
- **Handoff** tab: ran out of usage in one AI? Add your project files (or a folder, or pasted code), say what you were doing, and get one compact prompt for any other AI. Plain code in `handoff.py` reads your files and pulls out the endpoints (method, path, fields sent, fields read back), backend routes, forms and inputs, pages, storage keys, database tables, env variable names, stack and ports, and flags mismatches between frontend and backend. Gemini only writes the short summary from your notes and an outline of file and function names. Source code is never sent to it, and anything that looks like a key, token or password is masked first. `.env` files are read for variable names only.

All of the new backend code is in `features.py` and `handoff.py` (keep both next to `app.py`). If you edit `app.py` yourself, keep these two lines above `if __name__ == "__main__":`

    import features
    features.register(globals())

## Busy models

If the chosen model is overloaded or times out, the server tries the next model in the list instead of showing an error, and skips the busy one for a minute. You only get an error if every model fails. The page says which model answered.

## Notes

- Free Gemini usage may be used by Google to improve its products, so do not paste passwords or private data.
- If the free tier limit is hit, wait a minute. The experiment result is saved, so repeat runs cost nothing.
- Token counts come from OpenAI's tokenizer. Other models (Claude, Gemini) count a little differently.
