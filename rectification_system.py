"""
Article Rectification System

This is where you implement your article rectification logic.
The run() function receives AI-generated content and should return the corrected version.

Feel free to:
- Add additional modules, classes, or helper functions
- Load and compare with source articles
- Implement multi-step validation and correction strategies
- Use multiple LLM calls or different models
- Add confidence scoring and logging
"""

import re
import difflib
import os
import time
import logging
from typing import Tuple
from rapidfuzz import process, fuzz
from dotenv import load_dotenv
import requests

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "httplite")
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-85ZASE6Lt4osaIz7uc7x2Q")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://recllm.brahmastra.tech/")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-oss-120b")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))

NUMERIC_RE = re.compile(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b|\b\d+(?:\.\d+)?\s*(million|billion|thousand)\b", flags=re.I)
YEAR_RE = re.compile(r"\b(17|18|19|20)\d{2}\b")

logger = logging.getLogger(__name__)

def get_ai_generated_article(article_id: str) -> str:
    path = f"ai_generated_articles/{article_id}.txt"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def get_source_article(article_id: str) -> str:
    path = f"source_articles/{article_id}.txt"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def save_rectified_article(article_id: str, text: str):
    path = f"rectified_articles/{article_id}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# --- Minimal LLM caller (pluggable) ---
def call_llm_surgical(prompt: str) -> str:
    """
    Small wrapper to call a lightweight HTTP LLM endpoint or an OpenAI compatible one.
    This is deliberately minimal; you can swap in the SDK you have.
    """
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {}
    payload = {"model": LLM_MODEL, "prompt": prompt, "max_tokens": 512}
    try:
        if LLM_PROVIDER == "httplite" and LLM_BASE_URL:
            r = requests.post(LLM_BASE_URL, json=payload, headers=headers, timeout=LLM_TIMEOUT)
            r.raise_for_status()
            return r.json().get("text", "").strip()
        elif LLM_PROVIDER == "openai":
            # If using OpenAI SDK, replace this with a direct SDK call.
            r = requests.post(LLM_BASE_URL or "https://api.openai.com/v1/completions", json={
                "model": LLM_MODEL, "prompt": prompt, "max_tokens": 512, "temperature": 0
            }, headers={**headers, "Content-Type": "application/json"}, timeout=LLM_TIMEOUT)
            r.raise_for_status()
            return r.json()["choices"][0]["text"].strip()
        else:
            raise RuntimeError("No LLM provider configured")
    except Exception as e:
        logger.exception("LLM call failed: %s", e)
        return ""

def _split_sentences(text: str):
    # simple splitter; preserves abbreviations poorly but is fast and dependency-free
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if s.strip()]

def _best_source_candidates(ai_sentence: str, source_sentences: list, k: int = 3):
    # use rapidfuzz process.extract for fast fuzzy matches
    choices = {i: s for i, s in enumerate(source_sentences)}
    results = process.extract(ai_sentence, choices, scorer=fuzz.ratio, limit=k)
    # results: list of tuples (match, score, key)
    return [choices[r[2]] for r in results]

def _numeric_rule_replace(ai_sentence: str, src_sentence: str) -> Tuple[str, bool]:
    """
    If the ai_sentence contains numeric tokens and source sentence contains different numeric tokens,
    and the counts match, perform a deterministic replacement (one-to-one).
    Returns (corrected_sentence, changed_flag)
    """
    ai_nums = [m.group(0) for m in NUMERIC_RE.finditer(ai_sentence)]
    src_nums = [m.group(0) for m in NUMERIC_RE.finditer(src_sentence)]
    if ai_nums and src_nums and len(ai_nums) == len(src_nums) and any(a != b for a, b in zip(ai_nums, src_nums)):
        corrected = ai_sentence
        for a, b in zip(ai_nums, src_nums):
            corrected = corrected.replace(a, b, 1)
        return corrected, True
    return ai_sentence, False

def _minimal_edit_valid(original: str, candidate: str, max_fraction: float = 0.4) -> bool:
    # simple validation that candidate is not a large rewrite
    s = difflib.SequenceMatcher(None, original, candidate)
    changed_frac = 1 - s.ratio()
    return changed_frac <= max_fraction

def _build_surgical_prompt(ai_sentence_marked: str, source_sentence: str) -> str:
    # Strict prompt: single sentence output, no explanation.
    return (
        "You are an editor that must make exactly one minimal factual edit. "
        "Output exactly one sentence and nothing else.\n"
        "AI sentence:\n"
        f"\"{ai_sentence_marked}\"\n\n"
        "Source sentence (ground truth):\n"
        f"\"{source_sentence}\"\n\n"
        "Task: Replace only the text between << and >> if it is factually incorrect. "
        "Do not change any other words, punctuation, or capitalization. "
        "If the marked text is correct, return the AI sentence exactly as given.\n"
        "Output only the corrected sentence.\n"
    )

def surgical_rectify(ai_text: str, source_text: str, article_id: str = None) -> Tuple[str, dict]:
    """
    Main pipeline that returns (rectified_text, diagnostics)
    diagnostics includes: llm_calls, edits
    """
    ai_sents = _split_sentences(ai_text)
    src_sents = _split_sentences(source_text)
    corrected_sents = []
    diagnostics = {"llm_calls": 0, "edits": 0}

    for ai_s in ai_sents:
        # find candidates
        candidates = _best_source_candidates(ai_s, src_sents, k=3) if src_sents else []

        # 1) Try numeric deterministic rule with best candidate(s)
        applied = False
        for cand in candidates:
            corrected, changed = _numeric_rule_replace(ai_s, cand)
            if changed:
                corrected_sents.append(corrected)
                diagnostics["edits"] += 1
                applied = True
                break
        if applied:
            continue

        # 2) detect a suspicious span (number/year/percent) to mark
        span_m = NUMERIC_RE.search(ai_s) or YEAR_RE.search(ai_s)
        if span_m and candidates:
            marked = ai_s[:span_m.start()] + "<<" + span_m.group(0) + ">>" + ai_s[span_m.end():]
            prompt = _build_surgical_prompt(marked, candidates[0])
            resp = call_llm_surgical(prompt)
            diagnostics["llm_calls"] += 1
            if resp:
                # quick validation
                if _minimal_edit_valid(ai_s, resp):
                    corrected_sents.append(resp)
                    if resp != ai_s:
                        diagnostics["edits"] += 1
                    continue
            # fallback to original if validation fails
            corrected_sents.append(ai_s)
            continue

        # 3) If no obvious span but low similarity to best candidate -> ask for a surgical check
        if candidates:
            sim = difflib.SequenceMatcher(None, ai_s, candidates[0]).ratio()
            if sim < 0.75:
                prompt = _build_surgical_prompt(ai_s, candidates[0])
                resp = call_llm_surgical(prompt)
                diagnostics["llm_calls"] += 1
                if resp and _minimal_edit_valid(ai_s, resp):
                    corrected_sents.append(resp)
                    if resp != ai_s:
                        diagnostics["edits"] += 1
                    continue
        # default: keep sentence as-is
        corrected_sents.append(ai_s)

    rectified = " ".join(corrected_sents)
    return rectified, diagnostics

def run(ai_generated_content: str) -> str:
    """
    Rectify an AI-generated article.
    
    Args:
        ai_generated_content: The AI-generated article text to be corrected
        
    Returns:
        str: The rectified article content
    """
    # Create a simple prompt to fix issues
    prompt = (
        "Fix all issues in the following article:\n\n"
        f"{ai_generated_content}\n\n"
        
        "Return only the corrected article text."
    )

    # Call LLM to rectify the article
    response = completion(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        api_key=os.getenv('LLM_API_KEY'),
        api_base=os.getenv('LLM_API_BASE')
    )
    
    rectified_content = response.choices[0].message.content.strip()
    return rectified_content    

