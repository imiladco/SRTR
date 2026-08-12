#!/usr/bin/env python3
"""
SRT → Persian subtitle translator (GapGPT / OpenAI-compatible endpoint).

Improvements over the original:
  • Recursive bisection around content-filter blocks — isolates only the
    offending line(s) instead of dropping a whole 100-line chunk to source.
  • Model-driven glossary so proper nouns stay consistent across chunks
    (the old heuristic extractor produced empty values and was inert).
  • Crash-safe + validated cache: atomic writes, and the cache is invalidated
    automatically if the input file or chunk-size changes (the old cache keyed
    on chunk index, so changing --chunk-size silently corrupted resumes).
  • Per-request timeout, token accounting, and cleaner CLI / config handling.
  • Reasoning-model aware request building — GPT-5.6 family models (Sol /
    Terra / Luna) reject a custom `temperature` on Chat Completions and use
    `reasoning_effort` instead; regular models keep the old temperature knob.
"""

from __future__ import annotations

import os
import re
import json
import time
import hashlib
import argparse
from pathlib import Path

import chardet
from tqdm import tqdm
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE = 100
CONTEXT_TAIL_SIZE = 3
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2          # seconds, doubled each attempt
REQUEST_TIMEOUT = 90          # seconds per API call
DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_BASE_URL = "https://api.gapgpt.app/v1"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_REASONING_EFFORT = "low"
CACHE_VERSION = 2

PROMPTS_FILE = Path(__file__).parent / "prompts.txt"

# API key resolution order:  --api-key  >  env GAPGPT_API_KEY
# The key is never embedded in this file — a key that has ever been pasted
# into a script, chat, or ticket should be treated as compromised and rotated.
EMBEDDED_API_KEY = ""

# GPT-5.6 family (Sol / Terra / Luna) — and reasoning models generally — only
# accept the default temperature (1) on /v1/chat/completions and expose
# `reasoning_effort` instead. Keep this pattern list narrow and explicit
# rather than guessing from the model string in general.
REASONING_MODEL_PREFIXES = ("gpt-5.6-", "o1", "o3", "o4")


def is_reasoning_model(model: str) -> bool:
    return model.startswith(REASONING_MODEL_PREFIXES)


class ContentFilterError(Exception):
    """Raised when the provider blocks a request for content policy reasons."""


# ── Prompt Loader ─────────────────────────────────────────────────────────────
def load_prompts() -> tuple[str, str]:
    content = PROMPTS_FILE.read_text(encoding="utf-8")
    system = re.search(
        r"\[SYSTEM_PROMPT\]\n(.*?)\n\[USER_PROMPT_TEMPLATE\]", content, re.DOTALL
    )
    user = re.search(r"\[USER_PROMPT_TEMPLATE\]\n(.*)", content, re.DOTALL)
    if not system or not user:
        raise ValueError(
            "prompts.txt is malformed. Check [SYSTEM_PROMPT] and "
            "[USER_PROMPT_TEMPLATE] sections."
        )
    return system.group(1).strip(), user.group(1).strip()


# ── SRT Parser ────────────────────────────────────────────────────────────────
def detect_encoding(path: Path) -> str:
    result = chardet.detect(path.read_bytes())
    return result.get("encoding") or "utf-8"


def parse_srt(path: Path) -> list[dict]:
    encoding = detect_encoding(path)
    content = path.read_text(encoding=encoding, errors="replace")
    content = content.replace("\r\n", "\n").replace("\r", "\n").strip()

    blocks = re.split(r"\n{2,}", content)
    subtitles: list[dict] = []

    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        number_line = lines[0].strip()
        timecode_line = lines[1].strip()
        if not re.match(r"^\d+$", number_line):
            continue
        if "-->" not in timecode_line:
            continue
        text_lines = lines[2:] if len(lines) > 2 else [""]
        subtitles.append(
            {
                "index": len(subtitles),
                "number": number_line,
                "timecode": timecode_line,
                "text": "\n".join(text_lines),
            }
        )

    return subtitles


# ── SRT Writer ────────────────────────────────────────────────────────────────
def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file + os.replace so an interrupt never leaves a
    half-written / corrupted output or cache file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_srt(subtitles: list[dict], translations: dict[str, str], output_path: Path) -> None:
    lines: list[str] = []
    for sub in subtitles:
        translated = translations.get(sub["number"], sub["text"])
        # Normalise any literal "\n" the model returned into a real line break;
        # real newlines are left untouched.
        translated = translated.replace("\\n", "\n")
        lines.extend((sub["number"], sub["timecode"], translated, ""))
    _atomic_write(output_path, "\n".join(lines))


# ── Chunker ───────────────────────────────────────────────────────────────────
def chunk_subtitles(subtitles: list[dict], size: int) -> list[list[dict]]:
    return [subtitles[i : i + size] for i in range(0, len(subtitles), size)]


# ── Glossary ──────────────────────────────────────────────────────────────────
def merge_glossary(glossary: dict[str, str], new_entries: dict[str, str]) -> None:
    """First seen translation of a name wins, keeping it consistent thereafter."""
    if not isinstance(new_entries, dict):
        return
    for name, persian in new_entries.items():
        if name and isinstance(persian, str) and persian and name not in glossary:
            glossary[name] = persian


# ── JSON / Response Extractor ───────────────────────────────────────────────--
def extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from response:\n{text[:300]}")


def split_result(result: dict) -> tuple[dict[str, str], dict[str, str]]:
    """Accept either the wrapped format {"translations": {...}, "glossary": {...}}
    or a flat {"1": "...", ...} dict (backward compatible)."""
    glossary: dict[str, str] = {}
    if isinstance(result, dict) and "translations" in result:
        glossary = result.get("glossary") or {}
        result = result["translations"]
    translations = {k: v for k, v in result.items() if isinstance(v, str)}
    return translations, glossary


# ── API Caller ────────────────────────────────────────────────────────────────
def call_api(
    client: OpenAI,
    system_prompt: str,
    user_prompt_template: str,
    chunk: list[dict],
    context_tail: list[str],
    glossary: dict[str, str],
    model: str,
    temperature: float,
    reasoning_effort: str | None,
) -> tuple[dict[str, str], dict[str, str], int]:
    """Translate one chunk. Returns (translations, glossary_part, tokens_used)."""
    tokens_used = 0
    reasoning = reasoning_effort if is_reasoning_model(model) else None

    def _single(subs: list[dict]) -> tuple[dict[str, str], dict[str, str]]:
        nonlocal tokens_used
        sub_dict = {s["number"]: s["text"] for s in subs}
        payload = {
            "context_tail": context_tail,
            "glossary": glossary,
            "subtitles": sub_dict,
        }
        user_message = user_prompt_template.replace(
            "{input_json}", json.dumps(payload, ensure_ascii=False, indent=2)
        )

        request_kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            timeout=REQUEST_TIMEOUT,
            response_format={"type": "json_object"},
        )
        # Reasoning models (GPT-5.6 Sol/Terra/Luna, o1/o3/o4, ...) reject a
        # non-default `temperature` on chat.completions and use
        # `reasoning_effort` instead. Non-reasoning models keep temperature.
        if reasoning:
            request_kwargs["reasoning_effort"] = reasoning
        else:
            request_kwargs["temperature"] = temperature

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = client.chat.completions.create(**request_kwargs)
                usage = getattr(response, "usage", None)
                if usage:
                    tokens_used += getattr(usage, "total_tokens", 0) or 0

                translations, glossary_part = split_result(
                    extract_json(response.choices[0].message.content)
                )

                missing = [k for k in sub_dict if k not in translations]
                if missing:
                    tqdm.write(
                        f"  ⚠ Missing keys {missing[:5]} — falling back to source text."
                    )
                    for k in missing:
                        translations[k] = sub_dict[k]

                return translations, glossary_part

            except (ContentFilterError, ValueError):
                raise
            except Exception as e:
                err = str(e)
                if "content_filter" in err or ("400" in err and "content management" in err):
                    raise ContentFilterError(err)
                # Some OpenAI-compatible endpoints reject an unsupported
                # `temperature` outright instead of silently coercing it —
                # fall back to the reasoning_effort path once and retry.
                if "temperature" in err and "unsupported" in err.lower() and "temperature" in request_kwargs:
                    tqdm.write("  ⚠ Model rejected custom temperature — retrying without it.")
                    del request_kwargs["temperature"]
                    request_kwargs.setdefault("reasoning_effort", reasoning_effort or DEFAULT_REASONING_EFFORT)
                    continue
                last_error = e
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                tqdm.write(
                    f"  ✗ Attempt {attempt}/{MAX_RETRIES} failed: {e}. Retrying in {delay}s…"
                )
                time.sleep(delay)

        raise RuntimeError(f"All {MAX_RETRIES} attempts failed. Last error: {last_error}")

    def _recurse(subs: list[dict]) -> tuple[dict[str, str], dict[str, str]]:
        try:
            return _single(subs)
        except ContentFilterError:
            if len(subs) == 1:
                num = subs[0]["number"]
                tqdm.write(f"  ⚠ Content filter on subtitle {num} — keeping source text.")
                return {num: subs[0]["text"]}, {}
            mid = len(subs) // 2
            tqdm.write(f"  ⚠ Content filter — bisecting {len(subs)} lines to isolate it…")
            t1, g1 = _recurse(subs[:mid])
            t2, g2 = _recurse(subs[mid:])
            t1.update(t2)
            g1.update(g2)
            return t1, g1

    translations, glossary_part = _recurse(chunk)
    return translations, glossary_part, tokens_used


# ── Cache ─────────────────────────────────────────────────────────────────────
def cache_path(output_path: Path) -> Path:
    return output_path.with_suffix(".cache.json")


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_cache(output_path: Path, chunk_size: int, src_hash: str) -> dict:
    cp = cache_path(output_path)
    if not cp.exists():
        return {}
    try:
        data = json.loads(cp.read_text(encoding="utf-8"))
    except Exception:
        return {}

    meta = data.get("meta", {})
    if (
        meta.get("version") != CACHE_VERSION
        or meta.get("chunk_size") != chunk_size
        or meta.get("source_hash") != src_hash
    ):
        print("  ⚠ Existing cache is stale (input or chunk-size changed) — starting fresh.")
        return {}
    return data


def save_cache(
    output_path: Path,
    chunk_size: int,
    src_hash: str,
    translations: dict,
    glossary: dict,
    completed_chunks: set[int],
) -> None:
    payload = {
        "meta": {"version": CACHE_VERSION, "chunk_size": chunk_size, "source_hash": src_hash},
        "translations": translations,
        "glossary": glossary,
        "completed_chunks": sorted(completed_chunks),
    }
    _atomic_write(cache_path(output_path), json.dumps(payload, ensure_ascii=False, indent=2))


def clear_cache(output_path: Path) -> None:
    cp = cache_path(output_path)
    if cp.exists():
        cp.unlink()


# ── Main Translator ───────────────────────────────────────────────────────────
def translate_srt(
    input_path: Path,
    output_path: Path,
    api_key: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = DEFAULT_TEMPERATURE,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    fresh: bool = False,
) -> None:
    print(f"\n📥 Input:  {input_path}")
    print(f"📤 Output: {output_path}")
    if is_reasoning_model(model):
        print(f"🤖 Model:  {model}  |  Chunk size: {chunk_size}  |  reasoning_effort={reasoning_effort}\n")
    else:
        print(f"🤖 Model:  {model}  |  Chunk size: {chunk_size}  |  temperature={temperature}\n")

    system_prompt, user_prompt_template = load_prompts()
    client = OpenAI(api_key=api_key, base_url=base_url)

    subtitles = parse_srt(input_path)
    if not subtitles:
        raise ValueError("No valid subtitles found in input file.")
    print(f"Found {len(subtitles)} subtitle entries.")

    src_hash = source_hash(input_path)
    if fresh:
        clear_cache(output_path)

    chunks = chunk_subtitles(subtitles, chunk_size)
    print(f"Processing in {len(chunks)} chunk(s) of up to {chunk_size} subtitles each.\n")

    cache = load_cache(output_path, chunk_size, src_hash)
    translations: dict[str, str] = cache.get("translations", {})
    glossary: dict[str, str] = cache.get("glossary", {})
    completed_chunks: set[int] = set(cache.get("completed_chunks", []))

    context_tail: list[str] = []
    total_tokens = 0

    with tqdm(total=len(chunks), desc="Translating", unit="chunk") as pbar:
        for idx, chunk in enumerate(chunks):
            if idx in completed_chunks:
                context_tail = [
                    translations[s["number"]]
                    for s in chunk[-CONTEXT_TAIL_SIZE:]
                    if s["number"] in translations
                ]
                pbar.update(1)
                continue

            tqdm.write(
                f"\nChunk {idx + 1}/{len(chunks)}: "
                f"subtitles {chunk[0]['number']}–{chunk[-1]['number']}"
            )
            try:
                result, glossary_part, tokens = call_api(
                    client, system_prompt, user_prompt_template,
                    chunk, context_tail, glossary, model,
                    temperature, reasoning_effort,
                )
            except (RuntimeError, ValueError) as e:
                tqdm.write(f"  ✖ Failed to translate chunk {idx + 1}: {e}")
                save_cache(output_path, chunk_size, src_hash,
                           translations, glossary, completed_chunks)
                write_srt(subtitles, translations, output_path)
                print("\nPartial translation saved before failure.")
                raise SystemExit(1)

            translations.update(result)
            merge_glossary(glossary, glossary_part)
            total_tokens += tokens

            context_tail = [
                translations[s["number"]]
                for s in chunk[-CONTEXT_TAIL_SIZE:]
                if s["number"] in translations
            ]

            completed_chunks.add(idx)
            save_cache(output_path, chunk_size, src_hash,
                       translations, glossary, completed_chunks)
            pbar.update(1)

    write_srt(subtitles, translations, output_path)

    # Final integrity check before discarding the cache.
    untranslated = [s["number"] for s in subtitles if s["number"] not in translations]
    if untranslated:
        print(f"\n⚠ {len(untranslated)} subtitle(s) had no translation: {untranslated[:10]}")
        print("  Cache kept so you can re-run to retry only the missing ones.")
    else:
        clear_cache(output_path)
        print("\n✔ Translation complete.")

    if total_tokens:
        print(f"Tokens used this run: ~{total_tokens:,}")
    if glossary:
        names = ", ".join(f"{k}→{v}" for k, v in list(glossary.items())[:10])
        print(f"Glossary ({len(glossary)}): {names}{' …' if len(glossary) > 10 else ''}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def build_output_path(input_path: Path) -> Path:
    return input_path.with_name(input_path.stem + "_persian" + input_path.suffix)


def resolve_api_key(cli_key: str | None) -> str:
    key = cli_key or os.environ.get("GAPGPT_API_KEY") or EMBEDDED_API_KEY
    if not key or key.startswith("sk-...") or "REPLACE" in key:
        raise SystemExit(
            "No API key found. Pass --api-key or set the GAPGPT_API_KEY "
            "environment variable."
        )
    return key


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate SRT subtitles to Persian via a GapGPT/OpenAI endpoint.",
        epilog="""Examples:
  python translate.py movie.srt
  python translate.py movie.srt --output movie_fa.srt
  python translate.py movie.srt --chunk-size 80 --model gpt-5.6-terra
  python translate.py movie.srt --model gpt-5.6-luna --reasoning-effort low
  python translate.py movie.srt --fresh            # ignore any cached progress
  GAPGPT_API_KEY=sk-... python translate.py movie.srt
  """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Input .srt file path")
    parser.add_argument("--api-key", help="API key (overrides GAPGPT_API_KEY env var)")
    parser.add_argument("--output", help="Output .srt path (default: <input>_persian.srt)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model (default: {DEFAULT_MODEL})")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base URL")
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"Subtitles per request (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature for non-reasoning models (default: {DEFAULT_TEMPERATURE}). "
             "Ignored for reasoning models (GPT-5.6 Sol/Terra/Luna, o1/o3/o4), which only "
             "accept the default and use --reasoning-effort instead.",
    )
    parser.add_argument(
        "--reasoning-effort", default=DEFAULT_REASONING_EFFORT,
        choices=["none", "low", "medium", "high", "xhigh", "max"],
        help=f"Reasoning effort for reasoning models like gpt-5.6-luna (default: {DEFAULT_REASONING_EFFORT}). "
             "Ignored for non-reasoning models.",
    )
    parser.add_argument("--fresh", action="store_true", help="Ignore cache and re-translate")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"Input file does not exist: {input_path}")
    if input_path.suffix.lower() != ".srt":
        parser.error("Input file must be an .srt subtitle file")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be >= 1")

    output_path = Path(args.output) if args.output else build_output_path(input_path)
    api_key = resolve_api_key(args.api_key)

    translate_srt(
        input_path, output_path, api_key,
        chunk_size=args.chunk_size,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        fresh=args.fresh,
    )


if __name__ == "__main__":
    main()
