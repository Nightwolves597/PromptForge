"""
Handoff: reads code files and pulls out the facts another AI needs to continue.

Nothing in here calls an AI. It is plain code, so what it finds is read from your
files and not guessed. It finds:

  - every API call the frontend makes: method, path, the fields it sends, and the
    fields it reads back from the answer (best effort, followed through helper
    functions and variables)
  - every backend route (Flask, FastAPI, Express): the fields it reads and returns
  - forms and inputs, hash pages, browser storage keys
  - database tables, environment variable names, libraries, ports
  - mismatches when both sides are present
  - secrets (API keys, tokens, passwords), which are masked and never reported

analyze(files, count_tokens) returns a dict of facts. render_prompt(...) turns
those facts and a short narrative into the final handoff prompt.
"""

import json
import os
import re

MAX_FILES = 40
MAX_FILE_CHARS = 300_000
MAX_TOTAL_CHARS = 1_500_000

SKIP_PARTS = {"node_modules", ".git", "dist", "build", "__pycache__", ".venv", "venv", ".next", "coverage", ".idea", ".vscode", ".cache"}
SKIP_NAMES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", ".ds_store", "composer.lock", "pipfile.lock"}
TEXT_EXT = {
    ".html", ".htm", ".css", ".scss", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".vue", ".svelte",
    ".py", ".json", ".sql", ".md", ".txt", ".yml", ".yaml", ".toml", ".ini", ".cfg", ".sh", ".env",
    ".php", ".rb", ".go", ".java", ".cs", ".rs",
}
NAMED_OK = {"dockerfile", "makefile", "procfile", "requirements.txt"}

STDLIB = {
    "os", "sys", "re", "json", "time", "math", "hashlib", "sqlite3", "datetime", "typing", "collections", "contextlib",
    "concurrent", "functools", "itertools", "pathlib", "random", "string", "uuid", "logging", "threading", "subprocess",
    "urllib", "http", "base64", "copy", "csv", "io", "shutil", "tempfile", "traceback", "unittest", "dataclasses", "enum",
    "abc", "argparse", "asyncio", "decimal", "secrets", "statistics", "textwrap", "warnings", "zoneinfo", "__future__",
    "socket", "ssl", "email", "html", "xml", "platform", "signal", "queue", "heapq", "bisect", "operator", "pprint",
}

ARRAY_METHODS = {"map", "forEach", "filter", "find", "some", "every", "flatMap", "reduce", "slice", "join", "length", "sort", "includes", "push", "concat", "indexOf"}
STRING_METHODS = {"trim", "toLowerCase", "toUpperCase", "split", "toFixed", "toString", "replace", "startsWith", "endsWith", "substring", "padStart"}
DROP_END = ARRAY_METHODS | STRING_METHODS | {"entries", "keys", "values", "reverse", "then", "catch"}
ITER_METHODS = ("map", "forEach", "filter", "find", "some", "every", "flatMap", "reduce")
JS_KEYWORDS = {"if", "for", "while", "switch", "catch", "return", "function", "typeof", "await", "async", "new", "super", "import", "export"}

# ----------------------------------------------------------------------
# Secrets
# ----------------------------------------------------------------------
SECRET_PATTERNS = [
    ("Google API key", re.compile(r"AIza[0-9A-Za-z_\-]{35}")),
    ("Anthropic or OpenAI key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}")),
    ("Private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
    ("Bearer token", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{20,}")),
]
SECRET_ASSIGN = re.compile(
    r"""(?ix)\b([A-Z0-9_]*(?:SECRET|TOKEN|PASSWORD|PASSWD|API[_-]?KEY|APIKEY|PRIVATE[_-]?KEY)[A-Z0-9_]*)\s*([:=])\s*(["'])([^"'\n]{8,})\3"""
)


def scrub(text):
    """Masks secrets. Returns (clean_text, [type, ...]). Values are never kept."""
    found = []
    for label, pattern in SECRET_PATTERNS:
        text, n = pattern.subn("[MASKED]", text)
        found += [label] * n

    def mask(m):
        value = m.group(4)
        if re.search(r"[()\[\]{}]|^\$|^%|^<", value) or re.match(r"(?i)^(os|process|import|env|config|settings|your|xxx|change|example|todo|none|null)", value):
            return m.group(0)
        found.append("Hardcoded secret")
        return f"{m.group(1)}{m.group(2)}{m.group(3)}[MASKED]{m.group(3)}"

    text = SECRET_ASSIGN.sub(mask, text)
    return text, found


# ----------------------------------------------------------------------
# Text helpers: comments, brackets, arguments
# ----------------------------------------------------------------------
def strip_js(src):
    """Blanks comments and the insides of regex literals, keeping every index."""
    out = list(src)
    n, i, prev = len(src), 0, ""
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if c in "'\"":
            j = i + 1
            while j < n and src[j] != c and src[j] != "\n":
                j += 2 if src[j] == "\\" else 1
            i, prev = j + 1, c
            continue
        if c == "`":
            j = i + 1
            while j < n and src[j] != "`":
                j += 2 if src[j] == "\\" else 1
            i, prev = j + 1, c
            continue
        if c == "/" and nxt == "/":
            j = src.find("\n", i)
            j = n if j == -1 else j
            for k in range(i, j):
                out[k] = " "
            i = j
            continue
        if c == "/" and nxt == "*":
            j = src.find("*/", i + 2)
            j = n if j == -1 else j + 2
            for k in range(i, j):
                if out[k] != "\n":
                    out[k] = " "
            i = j
            continue
        if c == "/" and (prev == "" or prev in "(,=:[!&|?{};" or re.search(r"(?:return|typeof|case|in|of)\s*$", src[max(0, i - 8):i])):
            j, in_class = i + 1, False
            while j < n and src[j] != "\n":
                ch = src[j]
                if ch == "\\":
                    j += 2
                    continue
                if ch == "[":
                    in_class = True
                elif ch == "]":
                    in_class = False
                elif ch == "/" and not in_class:
                    break
                j += 1
            for k in range(i + 1, min(j, n)):
                out[k] = "x"
            i, prev = j + 1, "/"
            continue
        if not c.isspace():
            prev = c
        i += 1
    return "".join(out)


def strip_py(src):
    """Blanks # comments, keeping every index."""
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if src.startswith(('"""', "'''"), i):
            q = src[i:i + 3]
            j = src.find(q, i + 3)
            j = n if j == -1 else j + 3
            out.append(src[i:j])
            i = j
            continue
        if c in "\"'":
            j = i + 1
            while j < n and src[j] != c and src[j] != "\n":
                j += 2 if src[j] == "\\" else 1
            out.append(src[i:j + 1])
            i = j + 1
            continue
        if c == "#":
            j = src.find("\n", i)
            j = n if j == -1 else j
            out.append(" " * (j - i))
            i = j
            continue
        out.append(c)
        i += 1
    return "".join(out)


def match_close(s, i, open_ch, close_ch):
    """Index of the bracket that closes the one at s[i]. -1 if there is none."""
    depth, n, j = 0, len(s), i
    while j < n:
        c = s[j]
        if c in "'\"`":
            q = c
            j += 1
            while j < n and s[j] != q:
                if s[j] == "\\":
                    j += 1
                j += 1
            j += 1
            continue
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return j
        j += 1
    return -1


def split_top(s, sep=","):
    """Splits on sep at bracket depth 0, ignoring strings."""
    parts, depth, cur, i, n = [], 0, [], 0, len(s)
    while i < n:
        c = s[i]
        if c in "'\"`":
            j = i + 1
            while j < n and s[j] != c:
                if s[j] == "\\":
                    j += 1
                j += 1
            cur.append(s[i:j + 1])
            i = j + 1
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if c == sep and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
        i += 1
    last = "".join(cur).strip()
    if last:
        parts.append(last)
    return parts


def expr_end(s, i):
    """End of an expression that starts at i (up to ; or a newline at depth 0)."""
    depth, n, j = 0, len(s), i
    while j < n:
        c = s[j]
        if c in "'\"`":
            q = c
            j += 1
            while j < n and s[j] != q:
                if s[j] == "\\":
                    j += 1
                j += 1
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            if depth == 0:
                return j
            depth -= 1
        elif depth == 0 and c in ";\n":
            return j
        j += 1
    return n


# ----------------------------------------------------------------------
# Shapes: nested dicts. {"a": {}, "list": {"[]": {"id": {}}}}
# ----------------------------------------------------------------------
def shape_add(tree, path):
    node = tree
    for seg in path:
        node = node.setdefault(seg, {})


def shape_render(node, depth=0, limit=40):
    if not node or depth > 4:
        return ""
    parts = []
    for key, child in list(node.items())[:limit]:
        if key == "[]":
            continue
        if key == "*":
            parts.append("...")
            continue
        label, sub_node = key, child
        if "[]" in child:
            label += "[]"
            sub_node = child["[]"]
        sub = shape_render(sub_node, depth + 1, limit)
        parts.append(label + ("{" + sub + "}" if sub else ""))
    return ",".join(parts)


def shape_paths(node, prefix=()):
    """All key paths as tuples, without the [] markers."""
    out = set()
    for key, child in node.items():
        if key == "[]":
            out |= shape_paths(child, prefix)
            continue
        out.add(prefix + (key,))
        out |= shape_paths(child, prefix + (key,))
    return out


def shape_merge(a, b):
    for key, child in b.items():
        shape_merge(a.setdefault(key, {}), child)
    return a


def clip(text, size=420):
    return text if len(text) <= size else text[:size - 3] + "..."


# ----------------------------------------------------------------------
# JavaScript: definitions, constants, URLs
# ----------------------------------------------------------------------
IDENT = r"[A-Za-z_$][\w$]*"


def param_names(raw):
    names = []
    for p in split_top(raw):
        m = re.match(r"\s*(" + IDENT + ")", p)
        names.append(m.group(1) if m and not p.strip().startswith(("{", "[")) else "")
    return names


def js_functions(src):
    """name -> {params, start, end} for functions and arrow functions."""
    defs = {}
    for m in re.finditer(r"\bfunction\s*\*?\s*(" + IDENT + r")\s*\(([^)]*)\)\s*\{", src):
        b = m.end() - 1
        e = match_close(src, b, "{", "}")
        if e > 0:
            defs.setdefault(m.group(1), {"params": param_names(m.group(2)), "start": b, "end": e})
    for m in re.finditer(r"\b(?:const|let|var)\s+(" + IDENT + r")\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>\s*", src):
        s = m.end()
        e = match_close(src, s, "{", "}") if s < len(src) and src[s] == "{" else expr_end(src, s)
        if e > 0:
            defs.setdefault(m.group(1), {"params": param_names(m.group(2)), "start": s, "end": e})
    for m in re.finditer(r"\b(?:const|let|var)\s+(" + IDENT + r")\s*=\s*(?:async\s+)?(" + IDENT + r")\s*=>\s*", src):
        s = m.end()
        e = match_close(src, s, "{", "}") if s < len(src) and src[s] == "{" else expr_end(src, s)
        if e > 0:
            defs.setdefault(m.group(1), {"params": [m.group(2)], "start": s, "end": e})
    return defs


def object_entries(obj_src):
    """[(key, value_source)] for the top level of an object literal."""
    s = obj_src.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return []
    out = []
    for part in split_top(s[1:-1]):
        part = part.strip()
        if not part or part.startswith("..."):
            continue
        m = re.match(r"""^(?:['"]([^'"]+)['"]|(""" + IDENT + r"""))\s*(?::\s*(.*))?$""", part, re.S)
        if m:
            out.append((m.group(1) or m.group(2), m.group(3) if m.group(3) is not None else (m.group(2) or "")))
    return out


def object_shape(obj_src, depth=0):
    """Nested shape of an object literal, and whether it has spreads or computed keys."""
    s = obj_src.strip()
    shape, dynamic = {}, False
    if not (s.startswith("{") and s.endswith("}")):
        return shape, dynamic
    for part in split_top(s[1:-1]):
        part = part.strip()
        if part.startswith("...") or part.startswith("["):
            dynamic = True
    for key, value in object_entries(s):
        node = shape.setdefault(key, {})
        v = value.strip()
        if depth < 2 and v.startswith("{"):
            sub, dyn = object_shape(v, depth + 1)
            shape_merge(node, sub)
            dynamic = dynamic or dyn
    return shape, dynamic


def js_constants(src):
    """String constants and {key: 'string'} maps."""
    consts, maps = {}, {}
    for m in re.finditer(r"\b(?:const|let|var)\s+(" + IDENT + r")\s*=\s*(['\"])([^'\"\n]*)\2\s*(?:;|\n|,)", src):
        consts[m.group(1)] = m.group(3)
    for m in re.finditer(r"\b(?:const|let|var)\s+(" + IDENT + r")\s*=\s*\{", src):
        e = match_close(src, m.end() - 1, "{", "}")
        if e < 0:
            continue
        entries = {}
        for key, value in object_entries(src[m.end() - 1:e + 1]):
            vm = re.match(r"""^(['"])([^'"\n]*)\1$""", value.strip())
            if vm:
                entries[key] = vm.group(2)
        if entries:
            maps[m.group(1)] = entries
    return consts, maps


def resolve_part(p, consts, maps):
    p = p.strip()
    m = re.match(r"""^(['"])(.*)\1$""", p, re.S)
    if m:
        return m.group(2), True
    if p.startswith("`") and p.endswith("`"):
        def sub(mm):
            inner = mm.group(1).strip()
            text, literal = resolve_part(inner, consts, maps)
            return text if literal else "{" + (re.findall(IDENT, inner) or ["x"])[-1] + "}"
        return re.sub(r"\$\{([^}]*)\}", sub, p[1:-1]), True
    m = re.match(r"^(" + IDENT + r")$", p)
    if m and m.group(1) in consts:
        return consts[m.group(1)], True
    m = re.match(r"""^(""" + IDENT + r""")(?:\.(""" + IDENT + r""")|\[\s*['"]([^'"]+)['"]\s*\])$""", p)
    if m and m.group(1) in maps:
        key = m.group(2) or m.group(3)
        if key in maps[m.group(1)]:
            return maps[m.group(1)][key], True
    words = re.findall(IDENT, p)
    return "{" + (words[-1] if words else "x") + "}", False


def resolve_url(expr, consts, maps):
    """The path an expression builds, or None when nothing in it is known."""
    expr = expr.strip()
    parts = [p for p in split_top(expr, "+") if p]
    out, any_literal = "", False
    for p in parts:
        text, literal = resolve_part(p, consts, maps)
        out += text
        any_literal = any_literal or literal
    if not any_literal:
        return None
    return out


def js_method(opts, helpers, src):
    if not opts:
        return "GET"
    m = re.search(r"""\bmethod\s*:\s*['"](\w+)['"]""", opts)
    if m:
        return m.group(1).upper()
    m = re.match(r"^(" + IDENT + r")\s*\(", opts)
    if m and m.group(1) in helpers:
        return helpers[m.group(1)]
    m = re.match(r"^(" + IDENT + r")$", opts)
    if m:
        mm = re.search(r"\b(?:const|let|var)\s+" + re.escape(m.group(1)) + r"\s*=\s*\{", src)
        if mm:
            e = match_close(src, mm.end() - 1, "{", "}")
            mth = re.search(r"""\bmethod\s*:\s*['"](\w+)['"]""", src[mm.end() - 1:e + 1])
            if mth:
                return mth.group(1).upper()
        return "?"
    return "GET"


def object_from(expr, src):
    """Shape and dynamic flag of an object literal, or of the variable that holds one."""
    expr = expr.strip()
    if expr.startswith("{"):
        return object_shape(expr)
    m = re.match(r"^(" + IDENT + r")$", expr)
    if not m:
        return {}, False
    name = m.group(1)
    mm = re.search(r"\b(?:const|let|var)\s+" + re.escape(name) + r"\s*=\s*\{", src)
    shape, dynamic = {}, False
    if mm:
        e = match_close(src, mm.end() - 1, "{", "}")
        if e > 0:
            shape, dynamic = object_shape(src[mm.end() - 1:e + 1])
    for am in re.finditer(r"\b" + re.escape(name) + r"""\s*(?:\.\s*(""" + IDENT + r""")|\[\s*(['"])([^'"]+)\2\s*\]|\[\s*([^\]'"]+)\])\s*=(?!=)""", src):
        if am.group(1) or am.group(3):
            shape.setdefault(am.group(1) or am.group(3), {})
        else:
            dynamic = True
    return shape, dynamic


def js_body(opts, helpers, src):
    """(sends_shape, dynamic) from a fetch options expression."""
    if not opts:
        return {}, False
    m = re.search(r"\bbody\s*:\s*JSON\.stringify\(", opts)
    if m:
        e = match_close(opts, m.end() - 1, "(", ")")
        return object_from(opts[m.end():e], src)
    m = re.match(r"^(" + IDENT + r")\s*\(", opts)
    if m and m.group(1) in helpers:
        e = match_close(opts, opts.index("("), "(", ")")
        args = split_top(opts[opts.index("(") + 1:e])
        if args:
            return object_from(args[0], src)
    if re.search(r"\bbody\s*:\s*(?:new\s+)?(?:FormData|formData|form\b)", opts):
        fields = {k: {} for k in re.findall(r"""\.append\(\s*['"]([^'"]+)['"]""", src)[:30]}
        return fields, True
    return {}, False


# ----------------------------------------------------------------------
# JavaScript: what the frontend reads from an answer
# ----------------------------------------------------------------------
def parse_access(chain):
    """'.a?.b["c"]' -> ('a', 'b', 'c')"""
    segs = re.findall(r"""\??\.\s*(""" + IDENT + r""")|\[\s*['"]([^'"\]]+)['"]\s*\]""", chain)
    return tuple(a or b for a, b in segs)


def collect_paths(text, var, funcs, depth=0, seen=None, prefix=()):
    """Key paths that `text` reads from the variable `var`."""
    seen = seen if seen is not None else set()
    found = set()
    v = re.escape(var)

    for m in re.finditer(r"(?<![\w$.])" + v + r"""((?:\s*\??\.\s*""" + IDENT + r"""|\[\s*['"][^'"\]]+['"]\s*\])+)""", text):
        found.add(prefix + parse_access(m.group(1)))
    for m in re.finditer(r"\b(?:const|let|var)\s*\{([^}]*)\}\s*=\s*(?:await\s+)?" + v + r"\b(?!\s*[.(\[])", text):
        for part in split_top(m.group(1)):
            k = re.match(r"\s*(" + IDENT + ")", part)
            if k:
                found.add(prefix + (k.group(1),))
    for m in re.finditer(r"\b(?:const|let|var)\s+(" + IDENT + r")\s*=\s*" + v + r"((?:\.[\w$]+)+)\s*(?:\|\||\?\?|;|\n|,|\)|$)", text):
        base = prefix + parse_access(m.group(2))
        found.add(base)
        if m.group(1) != var and (m.group(1), base) not in seen:
            seen.add((m.group(1), base))
            found |= collect_paths(text, m.group(1), funcs, depth, seen, base)
    for m in re.finditer(r"(?<![\w$.])" + v + r"((?:\.[\w$]+)+)\.(" + "|".join(ITER_METHODS) + r")\s*\(", text):
        base = prefix + parse_access(m.group(1))
        found.add(base + ("[]",))
        op = m.end() - 1
        cl = match_close(text, op, "(", ")")
        if cl < 0:
            continue
        cb = text[op:cl + 1]
        pm = re.match(r"\(\s*(?:async\s*)?\(?\s*(" + IDENT + r")", cb)
        if pm and (pm.group(1), base) not in seen:
            seen.add((pm.group(1), base))
            found |= collect_paths(cb, pm.group(1), funcs, depth, seen, base + ("[]",))
        dm = re.match(r"\(\s*(?:async\s*)?\(?\s*\{([^}]*)\}", cb)
        if dm:
            for part in split_top(dm.group(1)):
                k = re.match(r"\s*(" + IDENT + ")", part)
                if k:
                    found.add(base + ("[]", k.group(1)))
    for m in re.finditer(r"for\s*\(\s*(?:const|let|var)\s+(" + IDENT + r")\s+of\s+" + v + r"((?:\.[\w$]+)*)\s*\)\s*\{", text):
        base = prefix + parse_access(m.group(2))
        e = match_close(text, m.end() - 1, "{", "}")
        if e > 0 and (m.group(1), base) not in seen:
            seen.add((m.group(1), base))
            found |= collect_paths(text[m.end() - 1:e + 1], m.group(1), funcs, depth, seen, base + ("[]",))
    if depth < 2:
        for m in re.finditer(r"(?<![\w$.])(" + IDENT + r")\s*\(", text):
            name = m.group(1)
            if name in JS_KEYWORDS or name not in funcs:
                continue
            cl = match_close(text, m.end() - 1, "(", ")")
            if cl < 0:
                continue
            args = split_top(text[m.end():cl])
            for idx, a in enumerate(args):
                if re.sub(r"^await\s+", "", a.strip()) == var and idx < len(funcs[name]["params"]) and funcs[name]["params"][idx]:
                    key = (name, idx, prefix)
                    if key in seen:
                        continue
                    seen.add(key)
                    fn = funcs[name]
                    found |= collect_paths(funcs[name]["src"][fn["start"]:fn["end"] + 1], funcs[name]["params"][idx], funcs, depth + 1, seen, prefix)
    return found


def to_shape(paths):
    tree = {}
    for p in paths:
        path = list(p)
        while path and path[-1] in DROP_END:
            last = path.pop()
            if last in ARRAY_METHODS and path:
                path.append("[]")
                break
        if path:
            shape_add(tree, path)
    return tree


def enclosing(funcs, pos):
    best = None
    for name, d in funcs.items():
        if d["start"] <= pos <= d["end"] and (best is None or d["end"] - d["start"] < best["end"] - best["start"]):
            best = d
    return best


def trace_response(src, call_start, call_end, funcs, callee_is_raw_fetch):
    before = src[max(0, call_start - 160):call_start]
    after = src[call_end:call_end + 240]
    fn = enclosing(funcs, call_start)
    scope_end = fn["end"] if fn else min(len(src), call_end + 6000)
    scope = src[call_end:scope_end]
    paths = set()
    var = None

    m = re.search(r"(?:const|let|var)\s+(" + IDENT + r")\s*=\s*(?:await\s+)?$", before)
    if m:
        var = m.group(1)
    else:
        m = re.search(r"(?:const|let|var)\s*\{([^}]*)\}\s*=\s*(?:await\s+)?$", before)
        if m:
            for part in split_top(m.group(1)):
                k = re.match(r"\s*(" + IDENT + ")", part)
                if k:
                    paths.add((k.group(1),))
        else:
            m = re.search(r"(" + IDENT + r")\s*\(\s*(?:await\s+)?$", before)
            if m and m.group(1) in funcs and m.group(1) not in JS_KEYWORDS:
                f = funcs[m.group(1)]
                if f["params"] and f["params"][0]:
                    paths |= collect_paths(f["src"][f["start"]:f["end"] + 1], f["params"][0], funcs, 1)
    m = re.match(r"^\s*\)?\s*\.then\(\s*\(?\s*(" + IDENT + r")\s*\)?\s*=>\s*\1\.json\(\)\s*\)\s*\.then\(\s*\(?\s*(" + IDENT + r")\s*\)?\s*=>", after)
    if m:
        body_start = call_end + m.end()
        end = match_close(src, call_end + after.index(".then(", after.index(".json()")) + 5, "(", ")") if ".then(" in after else -1
        body = src[body_start:end if end > 0 else body_start + 1500]
        paths |= collect_paths(body, m.group(2), funcs, 0)
    if var:
        if callee_is_raw_fetch:
            jm = re.search(r"\b(?:const|let|var)\s+(" + IDENT + r")\s*=\s*await\s+" + re.escape(var) + r"\.json\(\)", scope)
            if jm:
                paths |= collect_paths(scope, jm.group(1), funcs, 0)
            for jm in re.finditer(re.escape(var) + r"\.json\(\)\s*\)?\s*\.then\(\s*\(?\s*(" + IDENT + r")\s*\)?\s*=>", scope):
                paths |= collect_paths(scope[jm.end():jm.end() + 1500], jm.group(1), funcs, 0)
        else:
            paths |= collect_paths(scope, var, funcs, 0)
    return paths


# ----------------------------------------------------------------------
# JavaScript: API calls in one source
# ----------------------------------------------------------------------
def api_callers(funcs, src_by_name):
    """Functions that wrap fetch: name -> (url_idx, options_idx). Helpers that build options: name -> METHOD."""
    wrappers, helpers = {}, {}
    for name, d in funcs.items():
        body = d["src"][d["start"]:d["end"] + 1]
        fm = re.search(r"\bfetch\(\s*(" + IDENT + r")\s*(?:,\s*(" + IDENT + r"))?", body)
        if fm and fm.group(1) in d["params"]:
            wrappers[name] = (d["params"].index(fm.group(1)), d["params"].index(fm.group(2)) if fm.group(2) in d["params"] else None)
        mm = re.search(r"""\bmethod\s*:\s*['"](\w+)['"]""", body)
        if mm and "fetch(" not in body:
            helpers[name] = mm.group(1).upper()
    return wrappers, helpers


def js_calls(src, funcs, consts, maps, wrappers, helpers):
    calls = []
    targets = [("fetch", 0, 1, True)] + [(n, u, o, False) for n, (u, o) in wrappers.items()]
    for name, url_idx, opt_idx, raw in targets:
        for m in re.finditer(r"(?<![\w$.])" + re.escape(name) + r"\s*\(", src):
            if re.search(r"function\s*\*?\s*$", src[max(0, m.start() - 14):m.start()]):
                continue
            op = m.end() - 1
            cl = match_close(src, op, "(", ")")
            if cl < 0:
                continue
            args = split_top(src[op + 1:cl])
            if len(args) <= url_idx:
                continue
            url = resolve_url(args[url_idx], consts, maps)
            if not url or url.startswith("{"):
                continue
            opts = args[opt_idx] if opt_idx is not None and len(args) > opt_idx else ""
            calls.append(make_call(src, m.start(), cl + 1, url, opts, helpers, funcs, raw))
    for m in re.finditer(r"(?<![\w$.])(?:axios|\$|http|api)\.(get|post|put|delete|patch|getJSON)\s*\(", src):
        op = m.end() - 1
        cl = match_close(src, op, "(", ")")
        args = split_top(src[op + 1:cl]) if cl > 0 else []
        if not args:
            continue
        url = resolve_url(args[0], consts, maps)
        if not url or url.startswith("{"):
            continue
        verb = m.group(1).upper().replace("GETJSON", "GET")
        sends, dyn = ({}, False)
        if verb in ("POST", "PUT", "PATCH") and len(args) > 1:
            sends, dyn = object_from(args[1], src)
        call = make_call(src, m.start(), cl + 1, url, "", helpers, funcs, False)
        call.update({"method": verb, "sends": sends, "dynamic": dyn})
        calls.append(call)
    for m in re.finditer(r"(?<![\w$.])\$\.ajax\s*\(\s*\{", src):
        cl = match_close(src, m.end() - 1, "{", "}")
        obj = src[m.end() - 1:cl + 1] if cl > 0 else ""
        um = re.search(r"\burl\s*:\s*([^,}]+)", obj)
        tm = re.search(r"""\b(?:type|method)\s*:\s*['"](\w+)['"]""", obj)
        url = resolve_url(um.group(1), consts, maps) if um else None
        if url:
            sends, dyn = ({}, False)
            dm = re.search(r"\bdata\s*:\s*(\{[^}]*\})", obj)
            if dm:
                sends, dyn = object_shape(dm.group(1))
            call = make_call(src, m.start(), cl + 1, url, "", helpers, funcs, False)
            call.update({"method": (tm.group(1) if tm else "GET").upper(), "sends": sends, "dynamic": dyn})
            calls.append(call)
    return calls


def make_call(src, start, end, url, opts, helpers, funcs, raw):
    query = []
    if "?" in url:
        url, qs = url.split("?", 1)
        query = [k for k in re.findall(r"(?:^|&)([\w\-\[\]]+)=", qs)]
    sends, dynamic = js_body(opts, helpers, src)
    headers = re.findall(r"""['"]?([A-Za-z\-]+)['"]?\s*:""", re.search(r"headers\s*:\s*(\{[^}]*\})", opts).group(1)) if re.search(r"headers\s*:\s*\{[^}]*\}", opts) else []
    return {
        "method": js_method(opts, helpers, src),
        "url": url,
        "query": query,
        "sends": sends,
        "dynamic": dynamic,
        "headers": [h for h in headers if h.lower() not in ("content-type",)],
        "reads": to_shape(trace_response(src, start, end, funcs, raw)),
    }


# ----------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------
def html_facts(src):
    facts = {"title": "", "forms": [], "inputs": [], "views": [], "libs": [], "scripts": []}
    m = re.search(r"<title[^>]*>(.*?)</title>", src, re.S | re.I)
    if m:
        facts["title"] = re.sub(r"\s+", " ", m.group(1)).strip()
    for m in re.finditer(r"<script\b([^>]*)>(.*?)</script>", src, re.S | re.I):
        if "src=" not in m.group(1).lower() and m.group(2).strip():
            facts["scripts"].append(m.group(2))
    body = re.sub(r"<script\b.*?</script>|<style\b.*?</style>", " ", src, flags=re.S | re.I)

    def fields(chunk):
        out = []
        for fm in re.finditer(r"<(input|textarea|select)\b([^>]*)>", chunk, re.I):
            attrs = fm.group(2)
            name = (re.search(r"""\bname=["']([^"']+)["']""", attrs) or re.search(r"""\bid=["']([^"']+)["']""", attrs))
            kind = fm.group(1).lower()
            if kind == "input":
                tm = re.search(r"""\btype=["']([^"']+)["']""", attrs)
                kind = tm.group(1).lower() if tm else "text"
            if name and kind not in ("submit", "button", "reset", "image"):
                req = ", required" if re.search(r"\brequired\b", attrs) else ""
                out.append(f"{name.group(1)}({kind}{req})")
        seen, uniq = set(), []
        for f in out:
            if f not in seen:
                seen.add(f)
                uniq.append(f)
        return uniq

    spans = []
    for fm in re.finditer(r"<form\b([^>]*)>(.*?)</form>", body, re.S | re.I):
        spans.append((fm.start(), fm.end()))
        attrs = fm.group(1)
        fid = re.search(r"""\bid=["']([^"']+)["']""", attrs)
        action = re.search(r"""\baction=["']([^"']+)["']""", attrs)
        method = re.search(r"""\bmethod=["']([^"']+)["']""", attrs)
        facts["forms"].append({
            "id": fid.group(1) if fid else "",
            "action": action.group(1) if action else "",
            "method": method.group(1).upper() if method else "",
            "fields": fields(fm.group(2)),
        })
    rest = body
    for s, e in reversed(spans):
        rest = rest[:s] + " " * (e - s) + rest[e:]
    facts["inputs"] = fields(rest)
    seen = []
    for hm in re.finditer(r"""href=["']#([A-Za-z][\w\-]*)["']""", body):
        if hm.group(1) not in seen:
            seen.append(hm.group(1))
    facts["views"] = seen[:14]
    known = ["react", "vue", "svelte", "angular", "tailwind", "bootstrap", "jquery", "chart.js", "d3", "alpine", "htmx", "three", "gsap", "lodash", "axios"]
    for um in re.finditer(r"""(?:src|href)=["'](https?://[^"']+)["']""", src):
        url = um.group(1).lower()
        for k in known:
            if k in url and k not in facts["libs"]:
                facts["libs"].append(k)
        if "fonts.googleapis" in url:
            fam = re.search(r"family=([A-Za-z+]+)", um.group(1))
            if fam and "Google Fonts " + fam.group(1).replace("+", " ") not in facts["libs"]:
                facts["libs"].append("Google Fonts " + fam.group(1).replace("+", " "))
    return facts


# ----------------------------------------------------------------------
# Python backends (Flask, FastAPI)
# ----------------------------------------------------------------------
def py_functions(src):
    funcs = {}
    lines = src.split("\n")
    for i, line in enumerate(lines):
        m = re.match(r"^(\s*)(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*[^:]+)?:", line)
        if not m:
            continue
        indent = len(m.group(1))
        body = []
        for ln in lines[i + 1:]:
            if ln.strip() and len(ln) - len(ln.lstrip()) <= indent:
                break
            body.append(ln)
        funcs.setdefault(m.group(2), {"params": [p.split(":")[0].split("=")[0].strip() for p in split_top(m.group(3))], "annot": split_top(m.group(3)), "body": "\n".join(body), "line": i})
    return funcs


def py_dict_shape(text, funcs_shapes, depth=0):
    """Shape of a dict literal at the start of `text`, or {} when it is not one."""
    text = text.strip()
    if not text.startswith("{"):
        m = re.match(r"^(\w+)\s*\(", text)
        return dict(funcs_shapes.get(m.group(1), {})) if m else {}
    end = match_close(text, 0, "{", "}")
    if end < 0:
        return {}
    shape = {}
    for part in split_top(text[1:end]):
        part = part.strip()
        if part.startswith("**"):
            m = re.match(r"\*\*\s*(\w+)\s*\(", part)
            if m and m.group(1) in funcs_shapes:
                shape_merge(shape, funcs_shapes[m.group(1)])
            else:
                shape["*"] = {}  # unknown extra keys
            continue
        pieces = split_top(part, ":")
        if len(pieces) < 2:
            continue
        km = re.match(r"""^['"]([^'"]+)['"]$""", pieces[0].strip())
        if not km:
            continue
        node = shape.setdefault(km.group(1), {})
        value = ":".join(pieces[1:]).strip()
        if depth < 3:
            if value.startswith("{"):
                shape_merge(node, py_dict_shape(value, funcs_shapes, depth + 1))
            elif value.startswith("["):
                inner = value[1:-1].strip()
                if inner.startswith("{"):
                    shape_merge(node.setdefault("[]", {}), py_dict_shape(inner, funcs_shapes, depth + 1))
                elif "for" in inner and "{" in inner:
                    br = inner.index("{")
                    shape_merge(node.setdefault("[]", {}), py_dict_shape(inner[br:], funcs_shapes, depth + 1))
            elif re.match(r"^\w+\s*\(", value):
                shape_merge(node, py_dict_shape(value, funcs_shapes, depth + 1))
    return shape


def py_return_shapes(funcs):
    shapes = {}
    for name, f in funcs.items():
        m = re.search(r"\breturn\s*(\{)", f["body"])
        if m:
            s = py_dict_shape(f["body"][m.start(1):], {})
            if s:
                shapes[name] = s
    return shapes


_PY_TUPLES = {}


def py_tuple_consts(src):
    out = {}
    for m in re.finditer(r"^(\w+)\s*=\s*[(\[]([^()\[\]]*)[)\]]\s*$", src, re.M):
        items = re.findall(r"""['"]([\w\-]+)['"]""", m.group(2))
        if items and len(items) == len([x for x in split_top(m.group(2)) if x.strip()]):
            out[m.group(1)] = items
    return out


def py_reads(body, funcs, depth=0, var_hint=None):
    """Keys a handler reads from the request body, and from the query string."""
    reads, query = [], []
    variables = set(var_hint or [])
    for m in re.finditer(r"\b(\w+)\s*=\s*(?:await\s+)?(?:request\.(?:get_json|json|form|values)\b|json_body\s*\()", body):
        variables.add(m.group(1))
    for v in variables:
        for lm in re.finditer(r"^[ \t]*for\s+(\w+)\s+in\s+([^:\n]+):", body, re.M):
            loop = lm.group(1)
            if re.search(r"\b" + v + r"\s*(?:\.get\(\s*|\[\s*)" + loop + r"\b", body[lm.end():]):
                reads += re.findall(r"""['"]([\w\-]+)['"]""", lm.group(2))
                for name in re.findall(r"\b([A-Za-z_]\w*)\b", lm.group(2)):
                    reads += _PY_TUPLES.get(name, [])
        for m in re.finditer(r"\b" + v + r"""\s*\.get\(\s*['"]([\w\-]+)['"]""", body):
            reads.append(m.group(1))
        for m in re.finditer(r"\b" + v + r"""\[\s*['"]([\w\-]+)['"]\s*\]""", body):
            reads.append(m.group(1))
        for m in re.finditer(r"""['"]([\w\-]+)['"]\s+(?:not\s+)?in\s+""" + v + r"\b", body):
            reads.append(m.group(1))
        if depth < 2:
            for m in re.finditer(r"\b(\w+)\s*\(([^()]*)\)", body):
                if m.group(1) in funcs and re.search(r"\b" + v + r"\b", m.group(2)):
                    args = split_top(m.group(2))
                    for idx, a in enumerate(args):
                        if a.strip() == v and idx < len(funcs[m.group(1)]["params"]):
                            r2, q2 = py_reads(funcs[m.group(1)]["body"], funcs, depth + 1, [funcs[m.group(1)]["params"][idx]])
                            reads += r2
                            query += q2
    for m in re.finditer(r"""request\.(?:json|form|values)\s*(?:\.get\(\s*|\[\s*)['"]([\w\-]+)['"]""", body):
        reads.append(m.group(1))
    for m in re.finditer(r"""request\.get_json\([^)]*\)\s*(?:\.get\(\s*|\[\s*)['"]([\w\-]+)['"]""", body):
        reads.append(m.group(1))
    for m in re.finditer(r"""request\.args(?:\.get\(\s*|\[\s*)['"]([\w\-]+)['"]""", body):
        query.append(m.group(1))
    return reads, query


def py_returns(body, funcs_shapes, funcs):
    shape = {}
    statuses = set()
    for m in re.finditer(r"\bjsonify\(", body):
        cl = match_close(body, m.end() - 1, "(", ")")
        if cl < 0:
            continue
        arg = body[m.end():cl].strip()
        if arg.startswith("{") or re.match(r"^\w+\s*\(", arg):
            shape_merge(shape, py_dict_shape(arg, funcs_shapes))
        elif re.match(r"^\w+$", arg):
            shape_merge(shape, py_var_shape(body, arg, funcs_shapes))
        sm = re.match(r"\s*\)\s*,\s*(\d{3})", body[cl:cl + 20])
        if sm:
            statuses.add(int(sm.group(1)))
    for m in re.finditer(r"\breturn\s+(\{|\w+\s*\(|\w+\s*$)", body, re.M):
        rest = body[m.start(1):]
        if rest.startswith("{"):
            shape_merge(shape, py_dict_shape(rest, funcs_shapes))
        elif re.match(r"^\w+\s*$", rest.split("\n")[0]) and rest.split("\n")[0].strip() not in ("None", "True", "False"):
            shape_merge(shape, py_var_shape(body, rest.split("\n")[0].strip(), funcs_shapes))
    return shape, sorted(statuses)


def py_var_shape(body, var, funcs_shapes):
    shape = {}
    for m in re.finditer(r"\b" + re.escape(var) + r"\s*=\s*(\{|\w+\s*\()", body):
        shape_merge(shape, py_dict_shape(body[m.start(1):], funcs_shapes))
    for m in re.finditer(r"\b" + re.escape(var) + r"((?:\[\s*['\"][\w\-]+['\"]\s*\])+)\s*=(?!=)\s*([^\n]*)", body):
        path = re.findall(r"""\[\s*['"]([\w\-]+)['"]\s*\]""", m.group(1))
        node = shape
        for seg in path:
            node = node.setdefault(seg, {})
        expr_start = m.start(2)
        shape_merge(node, py_dict_shape(body[expr_start:], funcs_shapes))
    return shape


def py_routes(src, fname, funcs, funcs_shapes, classes):
    routes = []
    lines = src.split("\n")
    pats = [
        re.compile(r"""^[ \t]*@(\w+)\.(get|post|put|delete|patch)\(\s*(['"])(.*?)\3""", re.M),
        re.compile(r"""^[ \t]*@(\w+)\.route\(\s*(['"])(.*?)\2([^)]*)\)""", re.M),
    ]
    found = []
    for m in pats[0].finditer(src):
        found.append((m.start(), [m.group(2).upper()], m.group(4)))
    for m in pats[1].finditer(src):
        methods = re.findall(r"""['"](\w+)['"]""", m.group(4).split("methods")[1]) if "methods" in m.group(4) else ["GET"]
        found.append((m.start(), [x.upper() for x in methods], m.group(3)))
    for pos, methods, path in found:
        line_no = src.count("\n", 0, pos)
        fm = None
        for j in range(line_no + 1, min(len(lines), line_no + 8)):
            fm = re.match(r"^\s*(?:async\s+)?def\s+(\w+)", lines[j])
            if fm:
                break
        if not fm or fm.group(1) not in funcs:
            continue
        f = funcs[fm.group(1)]
        reads, query = py_reads(f["body"], funcs)
        for a in f["annot"]:
            am = re.match(r"\s*\w+\s*:\s*(\w+)", a)
            if am and am.group(1) in classes:
                reads += classes[am.group(1)]
        returns, statuses = py_returns(f["body"], funcs_shapes, funcs)
        norm = re.sub(r"<(?:\w+:)?(\w+)>", r"{\1}", path)
        for method in methods:
            if method in ("HEAD", "OPTIONS"):
                continue
            routes.append({
                "method": method, "path": norm, "file": fname, "handler": fm.group(1),
                "reads": list(dict.fromkeys(reads)), "query": list(dict.fromkeys(query)),
                "returns": returns, "statuses": statuses,
            })
    return routes


def before_request_reads(src, funcs):
    """{path: reads} for hooks like `if request.path == "/shorten": data.get("level")`."""
    out = {}
    for m in re.finditer(r"@\w+\.before_request\s*\n\s*def\s+(\w+)", src):
        f = funcs.get(m.group(1))
        if not f:
            continue
        reads, _ = py_reads(f["body"], funcs)
        extra = []
        for v in set(re.findall(r"\b(\w+)\s*=\s*request\.get_json", f["body"])):
            extra += re.findall(r"\b" + v + r"""\.get\(\s*['"]([\w\-]+)['"]""", f["body"])
        paths = re.findall(r"""request\.path\s*==\s*['"]([^'"]+)['"]""", f["body"])
        for p in paths:
            out.setdefault(p, []).extend(reads + extra)
    return out


def py_classes(src):
    classes = {}
    for m in re.finditer(r"^class\s+(\w+)\(([^)]*)\):\s*\n((?:[ \t]+.*\n?)*)", src, re.M):
        if "BaseModel" in m.group(2):
            classes[m.group(1)] = re.findall(r"^[ \t]+(\w+)\s*:", m.group(3), re.M)
    return classes


# ----------------------------------------------------------------------
# JavaScript servers (Express)
# ----------------------------------------------------------------------
def express_routes(src, fname):
    routes = []
    for m in re.finditer(r"""\b(?:app|router|server)\.(get|post|put|delete|patch)\(\s*(['"`])(.*?)\2\s*,""", src):
        cl = match_close(src, src.index("(", m.start()), "(", ")")
        body = src[m.end():cl] if cl > 0 else src[m.end():m.end() + 2500]
        reads = re.findall(r"\breq\.body\.(\w+)", body)
        for dm in re.finditer(r"\b(?:const|let|var)\s*\{([^}]*)\}\s*=\s*req\.body", body):
            reads += [re.match(r"\s*(\w+)", p).group(1) for p in split_top(dm.group(1)) if re.match(r"\s*(\w+)", p)]
        query = re.findall(r"\breq\.query\.(\w+)", body)
        for dm in re.finditer(r"\b(?:const|let|var)\s*\{([^}]*)\}\s*=\s*req\.query", body):
            query += [re.match(r"\s*(\w+)", p).group(1) for p in split_top(dm.group(1)) if re.match(r"\s*(\w+)", p)]
        shape, statuses = {}, set()
        for rm in re.finditer(r"\bres(?:\.status\((\d{3})\))?\.(?:json|send)\(\s*(\{)", body):
            e = match_close(body, rm.end(2) - 1, "{", "}")
            if e > 0:
                shape_merge(shape, object_shape(body[rm.end(2) - 1:e + 1])[0])
            if rm.group(1):
                statuses.add(int(rm.group(1)))
        routes.append({
            "method": m.group(1).upper(), "path": re.sub(r":(\w+)", r"{\1}", m.group(3)), "file": fname, "handler": "",
            "reads": list(dict.fromkeys(reads)), "query": list(dict.fromkeys(query)), "returns": shape, "statuses": sorted(statuses),
        })
    return routes


# ----------------------------------------------------------------------
# Config, database, environment
# ----------------------------------------------------------------------
def sql_tables(src):
    tables = []
    for m in re.finditer(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"]?(\w+)[`\"]?\s*\(", src, re.I):
        e = match_close(src, m.end() - 1, "(", ")")
        if e < 0:
            continue
        cols = []
        for part in split_top(src[m.end():e]):
            cm = re.match(r"^[`\"]?(\w+)[`\"]?\s+(\w+)", part.strip())
            if cm and cm.group(1).upper() not in ("PRIMARY", "FOREIGN", "UNIQUE", "CONSTRAINT", "KEY", "CHECK", "INDEX"):
                cols.append(f"{cm.group(1)} {cm.group(2).upper()}")
        tables.append({"name": m.group(1), "columns": cols})
    for m in re.finditer(r"^class\s+(\w+)\(([^)]*Model[^)]*)\):\s*\n((?:[ \t]+.*\n?)*)", src, re.M):
        cols = [f"{c[0]} {c[1]}" for c in re.findall(r"^[ \t]+(\w+)\s*=\s*(?:db\.)?Column\(\s*(?:db\.)?(\w+)", m.group(3), re.M)]
        if cols:
            tables.append({"name": m.group(1), "columns": cols})
    return tables


def env_names(src):
    names = re.findall(r"""os\.getenv\(\s*['"](\w+)['"]""", src)
    names += re.findall(r"""os\.environ(?:\.get\(\s*|\[\s*)['"](\w+)['"]""", src)
    names += re.findall(r"\bprocess\.env\.(\w+)", src)
    names += re.findall(r"\bimport\.meta\.env\.(\w+)", src)
    return [n for n in names if n.isupper() or "_" in n]


def js_modules(src):
    """Packages a JavaScript file imports or requires (relative files are left out)."""
    mods = re.findall(r"""\bimport\s+(?:[^'"\n;]*?\s+from\s+)?['"]([^'".\/][^'"]*)['"]""", src)
    mods += re.findall(r"""\brequire\(\s*['"]([^'".\/][^'"]*)['"]\s*\)""", src)
    return [m for m in mods if not m.startswith("node:")]


def storage_keys(src):
    keys = re.findall(r"""(?:localStorage|sessionStorage)\.(?:get|set|remove)Item\(\s*['"]([^'"]+)['"]""", src)
    keys += re.findall(r"""\bstore\.(?:get|set)\(\s*['"]([\w.\-]+)['"]""", src)
    keys += re.findall(r"""\b[A-Z_]*KEY\s*=\s*['"]([\w.\-]+)['"]""", src)
    return keys


# ----------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------
def norm_name(name):
    name = str(name or "").replace("\\", "/").strip()
    name = re.sub(r"^(?:\./)+", "", name).lstrip("/")
    return re.sub(r"/+", "/", name)


def accept(name, content):
    parts = name.lower().split("/")
    base = parts[-1]
    if any(p in SKIP_PARTS for p in parts[:-1]):
        return "inside a folder that is skipped (" + next(p for p in parts[:-1] if p in SKIP_PARTS) + ")"
    if base in SKIP_NAMES or re.search(r"\.min\.(js|css)$", base):
        return "generated or minified file"
    ext = os.path.splitext(base)[1]
    if not (ext in TEXT_EXT or base in NAMED_OK or base.startswith(".env")):
        return "not a code or text file"
    if len(content) > MAX_FILE_CHARS:
        return "too large (over 300,000 characters)"
    if "\x00" in content[:2000]:
        return "binary file"
    return ""


def lang_of(name):
    base = name.lower().split("/")[-1]
    ext = os.path.splitext(base)[1]
    if base.startswith(".env"):
        return "env"
    return {
        ".html": "html", ".htm": "html", ".css": "css", ".scss": "css", ".py": "py", ".sql": "sql", ".md": "docs", ".txt": "docs",
        ".json": "config", ".yml": "config", ".yaml": "config", ".toml": "config", ".ini": "config", ".cfg": "config",
        ".js": "js", ".jsx": "js", ".ts": "js", ".tsx": "js", ".mjs": "js", ".cjs": "js", ".vue": "js", ".svelte": "js",
    }.get(ext, "other")


def norm_path(p):
    p = p.split("?")[0].lower().rstrip("/") or "/"
    return re.sub(r"\{[^}]*\}", "{}", p)


def top_defs(funcs, limit=8):
    ranked = sorted(funcs.items(), key=lambda kv: -(kv[1]["end"] - kv[1]["start"]))
    return [n for n, _ in ranked if len(n) > 2][:limit]


def analyze(files, count_tokens):
    facts = {
        "files": [], "api_calls": [], "routes": [], "forms": [], "inputs": [], "views": [], "storage": [], "env": [],
        "tables": [], "stack": {"frontend": [], "backend": [], "database": [], "libs": [], "deps": [], "modules": []}, "vars": [],
        "base_url": "", "port": "", "mismatches": [], "skipped": [], "masked": [], "warnings": [], "readme": "",
        "errors_convention": False, "title": "",
    }
    seen, accepted, total = {}, [], 0
    for f in files or []:
        name = norm_name(f.get("name"))
        content = f.get("content")
        if not name or not isinstance(content, str):
            continue
        seen[name] = content
    for name, content in seen.items():
        reason = accept(name, content)
        if not reason and len(accepted) >= MAX_FILES:
            reason = f"over the limit of {MAX_FILES} files"
        if not reason and total + len(content) > MAX_TOTAL_CHARS:
            reason = "over the total size limit"
        if reason:
            facts["skipped"].append({"name": name, "reason": reason})
            continue
        total += len(content)
        accepted.append((name, content))

    items = []  # parsed per file
    for name, content in accepted:
        lang = lang_of(name)
        tokens = count_tokens(content)
        if lang == "env":
            names = re.findall(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", content, re.M)
            facts["env"] += names
            facts["warnings"].append(f"{name}: only the variable names were read. The values were ignored.")
            items.append({"name": name, "lang": lang, "tokens": tokens, "text": "", "raw": ""})
            continue
        clean, secrets = scrub(content)
        for label in secrets:
            facts["masked"].append({"type": label, "file": name})
        items.append({"name": name, "lang": lang, "tokens": tokens, "text": clean, "raw": content})

    js_sources = []  # (file, stripped source)
    py_sources = []
    for it in items:
        t, lang = it["text"], it["lang"]
        if lang == "html":
            hf = html_facts(t)
            facts["forms"] += [dict(f, file=it["name"]) for f in hf["forms"]]
            facts["inputs"] += hf["inputs"]
            facts["views"] += [v for v in hf["views"] if v not in facts["views"]]
            facts["stack"]["libs"] += [x for x in hf["libs"] if x not in facts["stack"]["libs"]]
            facts["title"] = facts["title"] or hf["title"]
            if hf["scripts"]:
                js_sources.append((it["name"], strip_js("\n".join(hf["scripts"]))))
            it["role"] = "frontend"
        elif lang == "js":
            js_sources.append((it["name"], strip_js(t)))
        elif lang == "py":
            py_sources.append((it["name"], strip_py(t)))
        elif lang == "sql":
            facts["tables"] += sql_tables(t)
            it["role"] = "database"
        elif lang == "docs":
            it["role"] = "docs"
            if it["name"].lower().endswith("readme.md") and not facts["readme"]:
                facts["readme"] = re.sub(r"\s+", " ", t)[:500]
        elif lang == "config":
            it["role"] = "config"
            base = it["name"].lower().split("/")[-1]
            if base == "package.json":
                try:
                    pj = json.loads(t)
                    deps = list((pj.get("dependencies") or {}).keys()) + list((pj.get("devDependencies") or {}).keys())
                    facts["stack"]["deps"] += deps
                    if pj.get("scripts"):
                        facts["scripts"] = list(pj["scripts"].keys())[:8]
                except ValueError:
                    pass
            if base == "requirements.txt":
                facts["stack"]["deps"] += [m.group(1) for m in re.finditer(r"^\s*([A-Za-z0-9_.\-]+)", t, re.M) if not m.group(1).startswith("-")]
        elif lang == "css":
            it["role"] = "frontend"
        if base_is_requirements(it["name"]):
            it["role"] = "config"

    # one pass over everything JavaScript knows, so a constant in one file resolves in another
    consts, maps = {}, {}
    for _, src in js_sources:
        c, m = js_constants(src)
        consts.update(c)
        maps.update(m)
    all_funcs = {}
    per_file_funcs = {}
    for fname, src in js_sources:
        funcs = js_functions(src)
        for d in funcs.values():
            d["src"] = src
        per_file_funcs[fname] = funcs
        for k, v in funcs.items():
            all_funcs.setdefault(k, v)
    wrappers, helpers = api_callers(all_funcs, None)

    calls, server_routes = [], []
    for fname, src in js_sources:
        funcs = dict(all_funcs)
        funcs.update(per_file_funcs[fname])
        is_server = bool(re.search(r"require\(\s*['\"]express['\"]\s*\)|from\s+['\"]express['\"]", src))
        if is_server:
            server_routes += express_routes(src, fname)
            facts["stack"]["backend"].append("Express")
            role = "backend"
        else:
            for c in js_calls(src, funcs, consts, maps, wrappers, helpers):
                c["file"] = fname
                calls.append(c)
            role = "frontend"
        facts["storage"] += storage_keys(src)
        facts["env"] += env_names(src)
        facts["stack"]["modules"] += js_modules(src)
        pm = re.search(r"\blisten\(\s*(?:process\.env\.PORT\s*(?:\|\||\?\?)\s*)?(\d{4})", src)
        if pm:
            facts["port"] = pm.group(1)
        for it in items:
            if it["name"] == fname and "role" not in it:
                it["role"] = role
                it["defines"] = top_defs(per_file_funcs.get(fname, {}))
        fw = []
        if re.search(r"from\s+['\"]react['\"]|\buseState\b|React\.", src):
            fw.append("React")
        if re.search(r"\bcreateApp\(|from\s+['\"]vue['\"]|\bnew Vue\(", src):
            fw.append("Vue")
        if re.search(r"from\s+['\"]svelte|\$:\s", src):
            fw.append("Svelte")
        if re.search(r"@angular/", src):
            fw.append("Angular")
        facts["stack"]["frontend"] += [f for f in fw if f not in facts["stack"]["frontend"]]
    for it in items:
        if it["lang"] == "html" and "defines" not in it:
            it["defines"] = []
            for fname, _src in js_sources:
                if fname == it["name"]:
                    it["defines"] = top_defs(per_file_funcs.get(fname, {}))

    local_modules = {os.path.splitext(n.split("/")[-1])[0] for n, _ in accepted}
    py_deps = set()
    hook_reads = {}
    for fname, src in py_sources:
        funcs = py_functions(src)
        fshapes = py_return_shapes(funcs)
        classes = py_classes(src)
        _PY_TUPLES.clear()
        _PY_TUPLES.update(py_tuple_consts(src))
        file_routes = py_routes(src, fname, funcs, fshapes, classes)
        for p, extra in before_request_reads(src, funcs).items():
            hook_reads.setdefault(p, []).extend(extra)
        server_routes += file_routes
        facts["env"] += env_names(src)
        facts["tables"] += sql_tables(src)
        for m in re.finditer(r"^\s*(?:from|import)\s+([A-Za-z_]\w*)", src, re.M):
            mod = m.group(1)
            if mod not in STDLIB and mod not in local_modules:
                py_deps.add(mod)
        if re.search(r"\bflask\b", src, re.I):
            facts["stack"]["backend"].append("Flask")
        if re.search(r"\bfastapi\b", src, re.I):
            facts["stack"]["backend"].append("FastAPI")
        if re.search(r"\bdjango\b", src, re.I):
            facts["stack"]["backend"].append("Django")
        if re.search(r"sqlite3", src):
            facts["stack"]["database"].append("SQLite")
        if re.search(r"sqlalchemy|psycopg|asyncpg", src, re.I):
            facts["stack"]["database"].append("SQL database (SQLAlchemy or Postgres)")
        if re.search(r"pymongo|mongoengine", src, re.I):
            facts["stack"]["database"].append("MongoDB")
        if re.search(r"\bCORS\(", src):
            facts["warnings"].append("The backend enables CORS.")
        pm = re.search(r"""getenv\(\s*["']PORT["']\s*,\s*["'](\d+)["']""", src) or re.search(r"\bport\s*=\s*(\d{4})", src)
        if pm:
            facts["port"] = pm.group(1)
        if re.search(r"""jsonify\(\s*\{\s*["']error["']""", src) or re.search(r"""["']error["']\s*:\s*err""", src):
            facts["errors_convention"] = True
        for it in items:
            if it["name"] == fname:
                it["role"] = "backend"
                it["defines"] = [n for n in funcs if len(n) > 2 and not n.startswith("_")][:8]
    facts["stack"]["deps"] += sorted(py_deps)
    for r in server_routes:
        if r["path"] in hook_reads:
            r["reads"] = list(dict.fromkeys(r["reads"] + hook_reads[r["path"]]))

    # merge duplicate API calls
    merged = {}
    for c in calls:
        key = (c["method"], c["url"])
        if key not in merged:
            merged[key] = {"method": c["method"], "url": c["url"], "files": [c["file"]], "sends": {}, "dynamic": False, "query": [], "headers": [], "reads": {}}
        m = merged[key]
        shape_merge(m["sends"], c["sends"])
        shape_merge(m["reads"], c["reads"])
        m["dynamic"] = m["dynamic"] or c["dynamic"]
        m["query"] = list(dict.fromkeys(m["query"] + c["query"]))
        m["headers"] = list(dict.fromkeys(m["headers"] + c["headers"]))
        if c["file"] not in m["files"]:
            m["files"].append(c["file"])
    facts["api_calls"] = list(merged.values())
    rmerged = {}
    for r in server_routes:
        key = (r["method"], r["path"])
        if key not in rmerged:
            rmerged[key] = r
        else:
            rmerged[key]["reads"] = list(dict.fromkeys(rmerged[key]["reads"] + r["reads"]))
            shape_merge(rmerged[key]["returns"], r["returns"])
    facts["routes"] = list(rmerged.values())

    # base url: an absolute URL in the calls or a constant named like one
    for name, value in consts.items():
        if re.match(r"(?i)^(api|base|backend|server)[_]?(url|base|origin|host)?$|.*api.*(url|base).*", name) and re.match(r"https?://", value):
            facts["base_url"] = value
    for c in facts["api_calls"]:
        m = re.match(r"^(https?://[^/]+)(/.*)?$", c["url"])
        if m:
            facts["base_url"] = facts["base_url"] or m.group(1)
            c["url"] = m.group(2) or "/"

    facts["stack"]["frontend"] = list(dict.fromkeys(facts["stack"]["frontend"]))
    if not facts["stack"]["frontend"] and facts["api_calls"]:
        facts["stack"]["frontend"] = ["plain HTML and JavaScript"] if any(it["lang"] == "html" for it in items) else ["JavaScript"]
    for name, value in consts.items():
        if len(facts["vars"]) < 25 and not re.search(r"(?i)secret|token|password|key", name):
            facts["vars"].append(f"{name} = {value!r}")
    for name, entries in maps.items():
        if len(facts["vars"]) < 25 and not re.search(r"(?i)secret|token|password|key", name):
            facts["vars"].append(f"{name} = {{" + ", ".join(f"{k}: {v!r}" for k, v in list(entries.items())[:12]) + "}")
    for k in ("backend", "database", "libs", "deps", "modules"):
        facts["stack"][k] = list(dict.fromkeys(facts["stack"][k]))
    facts["env"] = list(dict.fromkeys(facts["env"]))
    facts["storage"] = list(dict.fromkeys(facts["storage"]))
    if any(x["name"] == "error" for x in []):
        pass
    for c in facts["api_calls"]:
        if "error" in c["reads"]:
            facts["errors_convention"] = True
    facts["mismatches"] = find_mismatches(facts)

    for it in items:
        facts["files"].append({
            "name": it["name"], "role": it.get("role", "other"), "tokens": it["tokens"], "defines": it.get("defines", []),
            "lang": it["lang"],
        })
    facts["files"].sort(key=lambda f: ({"frontend": 0, "backend": 1, "database": 2, "config": 3}.get(f["role"], 4), f["name"]))
    facts["files_tokens"] = sum(f["tokens"] for f in facts["files"])
    facts["has_frontend"] = bool(facts["api_calls"] or facts["forms"] or any(f["role"] == "frontend" for f in facts["files"]))
    facts["has_backend"] = bool(facts["routes"] or any(f["role"] == "backend" for f in facts["files"]))
    return facts


def base_is_requirements(name):
    return name.lower().split("/")[-1] == "requirements.txt"


def find_mismatches(facts):
    out = []
    calls, routes = facts["api_calls"], facts["routes"]
    if not calls or not routes:
        return out
    by_path = {}
    for r in routes:
        by_path.setdefault(norm_path(r["path"]), []).append(r)
    used = set()
    for c in calls:
        key = norm_path(c["url"])
        candidates = by_path.get(key, [])
        if not candidates:
            out.append(f"The frontend calls {c['method']} {c['url']} but the backend has no route for that path.")
            continue
        match = next((r for r in candidates if r["method"] == c["method"]), None)
        if not match and c["method"] != "?":
            out.append(f"The frontend calls {c['method']} {c['url']} but the backend only accepts {', '.join(sorted({r['method'] for r in candidates}))} there.")
            continue
        match = match or candidates[0]
        used.add((match["method"], match["path"]))
        sent = set(shape_paths(c["sends"]) and {p[0] for p in shape_paths(c["sends"])})
        read_by_backend = set(match["reads"]) | set(match["query"]) | set(c["query"])
        extra = sorted(sent - read_by_backend) if (match["reads"] or match["query"]) else []
        if extra:
            out.append(f"{c['method']} {c['url']}: the frontend sends {', '.join(extra)}, which the backend does not read.")
        if not c["dynamic"]:
            missing = sorted(set(match["reads"]) - sent)
            if missing and sent:
                out.append(f"{c['method']} {c['url']}: the backend reads {', '.join(missing)}, which the frontend does not send.")
        back = shape_paths(match["returns"])
        wild = {p[:-1] for p in back if p and p[-1] == "*"}
        if back:
            lacking = []
            for p in sorted(shape_paths(c["reads"])):
                if any(p[:len(w)] == w for w in wild if len(w) <= len(p) - 1 or w == ()):
                    continue
                if p[0] not in {q[0] for q in back}:
                    if len(p) == 1:
                        lacking.append(p[0])
                elif len(p) > 1 and ((p[0],) in back) and any(q[:1] == (p[0],) and len(q) > 1 for q in back) and p not in back:
                    lacking.append(".".join(p))
            if lacking:
                out.append(f"{c['method']} {c['url']}: the frontend reads {', '.join(lacking[:8])} from the answer, which the backend does not return.")
    for r in routes:
        if (r["method"], r["path"]) not in used and r["path"] != "/" and norm_path(r["path"]) not in {norm_path(c["url"]) for c in calls}:
            out.append(f"The backend has {r['method']} {r['path']}, which the frontend never calls.")
    return out[:30]


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------
GOALS = {
    "backend": "Build the backend for the existing frontend described below.",
    "frontend": "Build the frontend for the existing backend described below.",
    "continue": "Continue this project from where I stopped.",
    "fix": "Review this project and fix its problems, starting with any mismatch between the frontend and the backend.",
    "tests": "Write tests for this project.",
}

GOAL_RULES = {
    "backend": [
        "Do not change the frontend. Match every path, method, field name and response shape in the contract exactly.",
        "Write every backend file in full, with its file name above it, so I can paste them next to my frontend and run it.",
        "Use only the modules named in the contract plus what the stack needs. List the install command and the run command.",
        "Serve the frontend files from the same server, or enable CORS if the base URL or port differs. Read secrets from environment variables.",
    ],
    "frontend": ["Do not change the backend. Call the routes in the contract with exactly these paths, methods and field names."],
    "continue": ["Keep the existing file names, function names and field names. Do not rename or restructure anything unless I ask."],
    "fix": ["Change as little as possible. Explain each fix in one line."],
    "tests": ["Do not change the source code. Test the behaviour in the contract."],
}


def sends_text(shape, dynamic):
    s = shape_render(shape)
    return ("{" + s + (",..." if dynamic and s else "") + "}") if s else ""


def call_line(c):
    path = c["url"] + ("?" + "&".join(k + "=" for k in c["query"]) if c["query"] else "")
    parts = [f"{c['method']} {path}"]
    s = sends_text(c["sends"], c["dynamic"])
    if s:
        parts.append("sends " + s)
    if c["headers"]:
        parts.append("headers " + ",".join(c["headers"]))
    r = shape_render(c["reads"])
    if r:
        parts.append("reads {" + clip(r) + "}")
    return "  ".join(parts)


def route_line(r):
    path = r["path"] + ("?" + "&".join(k + "=" for k in r["query"]) if r["query"] else "")
    parts = [f"{r['method']} {path}"]
    if r["reads"]:
        parts.append("reads {" + ",".join(r["reads"]) + "}")
    rs = shape_render(r["returns"])
    if rs:
        parts.append("returns {" + clip(rs) + "}")
    if r["statuses"]:
        parts.append("status " + ",".join(str(x) for x in r["statuses"]))
    return "  ".join(parts)


def found_lists(facts):
    """The same facts as short lines, for the page to show."""
    inputs = []
    for f in facts["forms"]:
        label = "form" + (" #" + f["id"] if f["id"] else "") + (f" {f['method']} {f['action']}" if f["action"] else "")
        inputs.append(f"{label}: " + ", ".join(f["fields"]) if f["fields"] else label)
    if facts["inputs"]:
        inputs.append("inputs outside forms: " + ", ".join(facts["inputs"][:30]))
    return {
        "api_calls": [call_line(c) for c in facts["api_calls"]],
        "routes": [route_line(r) for r in facts["routes"]],
        "inputs": inputs,
        "views": facts["views"],
        "tables": [f"{t['name']}({', '.join(t['columns'])})" for t in facts["tables"]],
        "env": facts["env"],
        "storage": facts["storage"],
        "stack": {k: v for k, v in facts["stack"].items() if v},
        "files": facts["files"],
        "mismatches": facts["mismatches"],
    }


def contract_sections(facts, goal):
    """[(title, [lines])] in the order they appear in the prompt."""
    sections = []
    lines = []
    if facts["base_url"]:
        lines.append(f"Base URL: {facts['base_url']}")
    if facts["api_calls"]:
        head = "The frontend calls these endpoints. Build them." if goal == "backend" else "The frontend calls these endpoints:"
        lines.append(head)
        lines += [call_line(c) for c in facts["api_calls"]]
    if facts["routes"]:
        lines.append("Backend routes that already exist:" if facts["api_calls"] else "Backend routes:")
        lines += [route_line(r) for r in facts["routes"]]
    if facts["errors_convention"]:
        lines.append("Errors: JSON {error: message} with a 4xx or 5xx status.")
    if lines:
        sections.append(("Contract (read from my code, exact)", lines))
    ui = []
    for f in facts["forms"]:
        label = "Form" + (" #" + f["id"] if f["id"] else "")
        ui.append(f"{label}: " + (", ".join(f["fields"]) if f["fields"] else "no fields"))
    if facts["inputs"]:
        ui.append("Inputs: " + ", ".join(facts["inputs"][:30]))
    if facts["views"]:
        ui.append("Pages (hash routes): " + ", ".join("#" + v for v in facts["views"]))
    if facts["storage"]:
        ui.append("Browser storage keys: " + ", ".join(facts["storage"][:20]))
    if ui:
        sections.append(("Frontend inputs and pages", ui))
    data = [f"{t['name']}({', '.join(t['columns'])})" for t in facts["tables"]]
    if data:
        sections.append(("Database tables", data))
    cfg = []
    st = facts["stack"]
    for label, key in (("Frontend", "frontend"), ("Backend", "backend"), ("Database", "database")):
        if st[key]:
            cfg.append(f"{label}: " + ", ".join(st[key]))
    if st["libs"]:
        cfg.append("Libraries on the page: " + ", ".join(st["libs"][:12]))
    if st["modules"]:
        cfg.append("JavaScript modules imported: " + ", ".join(dict.fromkeys(st["modules"])))
    if facts["vars"]:
        cfg.append("Frontend constants: " + "; ".join(facts["vars"][:15]))
    if st["deps"]:
        cfg.append("Packages: " + ", ".join(st["deps"][:25]))
    if facts["env"]:
        cfg.append("Environment variables: " + ", ".join(facts["env"]))
    if facts["port"]:
        cfg.append(f"Port: {facts['port']}")
    if cfg:
        sections.append(("Stack and config", cfg))
    files = []
    for f in facts["files"]:
        if f["role"] == "docs":
            continue
        line = f"{f['name']} ({f['role']}, {f['tokens']:,} tokens)"
        if f.get("defines"):
            line += ": " + ", ".join(f["defines"][:8])
        files.append(line)
    if files:
        sections.append(("Files", files))
    if facts["mismatches"]:
        sections.append(("Known mismatches between frontend and backend", facts["mismatches"]))
    return sections


SECTION_TAGS = {
    "Contract": "contract",
    "Frontend inputs and pages": "frontend_ui",
    "Database tables": "database",
    "Stack and config": "stack",
    "Files": "files",
    "Known mismatches between frontend and backend": "mismatches",
}


def render_prompt(facts, narrative, goal, target_ai):
    use_xml = target_ai == "claude"
    goal = goal if goal in GOALS else "continue"
    rules = list(dict.fromkeys(list(GOAL_RULES[goal]) + list(narrative.get("rules") or []) + ["Give complete files I can paste, not fragments."]))
    parts = []

    def block(title, tag, lines, bullets=True):
        lines = [str(x) for x in lines if str(x).strip()]
        if not lines:
            return
        if use_xml:
            body = "\n".join(("- " + x if bullets else x) for x in lines)
            parts.append(f"<{tag}>\n{body}\n</{tag}>")
        else:
            body = "\n".join(("- " + x if bullets else x) for x in lines)
            parts.append(f"## {title}\n{body}")

    summary = (narrative.get("summary") or "").strip()
    if use_xml:
        parts.append("<project>\n" + (summary or "A web project.") + "\n</project>")
    else:
        parts.append("# Project handoff\n" + (summary or "A web project."))
    block("Task", "task", [(GOALS[goal] + " " + (narrative.get("task") or "")).strip()], bullets=False)
    block("Where I am", "status", narrative.get("done") or [])
    block("Rules", "rules", rules)
    for title, lines in contract_sections(facts, goal):
        tag = SECTION_TAGS.get(title.split(" (")[0], "section")
        block(title.split(" (")[0] if use_xml is False else title, tag, lines, bullets=False)
    block("Please confirm or double check", "assumptions", narrative.get("assumptions") or [])
    text = "\n\n".join(parts)
    return f"<handoff>\n{text}\n</handoff>" if use_xml else text
