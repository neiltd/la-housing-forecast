"""Send every prompt in llm/prompts to an LLM API and save the raw answers in llm/answers.

Keys are read from .env (GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY); never commit that file.
Every call is a fresh single-turn request: our prompt only, no system prompt, no tools, default settings.

    python llm_run_api.py list   gemini          -> models this key can use
    python llm_run_api.py run    gemini <model> [workers]  -> llm/answers/<model>_<variant>_<origin>.txt
    python llm_run_api.py run    openai <model>
    python llm_run_api.py run    claude claude-opus-5-5
"""
import json, os, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
PROMPTS, ANSWERS = ROOT / "llm" / "prompts", ROOT / "llm" / "answers"
for line in ((ROOT / ".env").read_text().splitlines() if (ROOT / ".env").exists() else []):
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"'))


def post(url, body, headers):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def gemini(model, text):
    key = os.environ["GEMINI_API_KEY"]
    r = post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
             {"contents": [{"parts": [{"text": text}]}]}, {"x-goog-api-key": key})
    return "".join(p.get("text", "") for p in r["candidates"][0]["content"]["parts"])


def openai(model, text):
    key = os.environ["OPENAI_API_KEY"]
    r = post("https://api.openai.com/v1/responses", {"model": model, "input": text},
             {"Authorization": f"Bearer {key}"})
    return r["output_text"] if "output_text" in r else "".join(
        c.get("text", "") for o in r["output"] if o["type"] == "message" for c in o["content"])


def claude(model, text):
    import anthropic
    r = anthropic.Anthropic().messages.create(model=model, max_tokens=16000,
                                              messages=[{"role": "user", "content": text}])
    if r.stop_reason == "refusal":
        raise RuntimeError("refused")
    return "".join(b.text for b in r.content if b.type == "text")


def list_models(provider):
    if provider == "gemini":
        req = urllib.request.Request("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                                     headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
        with urllib.request.urlopen(req) as r:
            for m in json.load(r)["models"]:
                if "generateContent" in m.get("supportedGenerationMethods", []):
                    print(m["name"].removeprefix("models/"))


def run(provider, model, workers=1):
    from concurrent.futures import ThreadPoolExecutor
    ANSWERS.mkdir(parents=True, exist_ok=True)
    call = {"gemini": gemini, "openai": openai, "claude": claude}[provider]

    def one(p):
        out = ANSWERS / f"{model}_{p.stem}.txt"
        if out.exists() and out.stat().st_size:
            return
        for attempt in range(4):
            try:
                out.write_text(call(model, p.read_text()))
                print("ok", out.name, flush=True)
                return
            except Exception as e:  # rate limits: back off and retry
                print("retry", p.stem, e, flush=True)
                time.sleep(20 * (attempt + 1))

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, sorted(PROMPTS.glob("*.txt"))))


if __name__ == "__main__":
    if sys.argv[1] == "list":
        list_models(sys.argv[2])
    else:
        run(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 1)
