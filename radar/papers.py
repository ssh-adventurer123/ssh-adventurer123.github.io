#!/usr/bin/env python3
"""Append a daily 'AI papers' section to today's blog post.

Picks 3-5 arXiv papers from the last 7 days, most valuable for the user's
research interests (LLM, multi-agent systems, context engineering / ACE,
Vibe Coding, AI toolchain), with no repeats.

How it works:
  1. Query the arXiv API per category (last 7 days, newest first, capped).
  2. Drop papers already published in the last 8 days of blog posts.
  3. One Groq call ranks the rest and picks the top 3-5 (Chinese titles +
     one-line reasons). Falls back to recency order if ranking fails.
  4. Append a '## AI 论文' section to today's post. Idempotent: skips if
     the section is already there.

Stdlib only. Reads GROQ_API_KEY from the environment for ranking.
"""
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

ARXIV_API = "https://export.arxiv.org/api/query"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"

# (arXiv category, max results per query) — ordered by relevance to the user.
CATEGORIES = [
    ("cs.AI", 15),
    ("cs.CL", 12),
    ("cs.LG", 12),
    ("cs.MA", 10),
    ("cs.SE", 10),
]

DAYS = 7            # pool: papers submitted in the last 7 days
SEEN_DAYS = 8       # dedup against posts from the last 8 days (1 day margin)
PICK_COUNT = 5      # aim for 3-5 papers
MIN_COUNT = 3
SECTION_MARKER = "## AI 论文"

INTERESTS = (
    "大语言模型 (LLM)：从基础理论到实际应用；"
    "多代理系统 (MAS)：智能体协作、通信机制、设计模式；"
    "上下文工程：提升模型理解和生成能力，以及 ACE (Agentic Context Engineering)；"
    "Vibe Coding：使用 AI 编程提升个人全方位效能；"
    "AI 工具链：构建高效的 AI 开发工具"
)

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
ARXIV_ID_RE = re.compile(r"arxiv\.org/abs/(\d{4}\.\d{4,5})")


def beijing_today() -> datetime:
    return datetime.now(timezone(timedelta(hours=8)))


def fetch_category(category: str, max_results: int, since: datetime, until: datetime):
    """Query the arXiv API for one category; return list of paper dicts."""
    start = since.strftime("%Y%m%d%H%M")
    end = until.strftime("%Y%m%d%H%M")
    query = f"cat:{category} AND submittedDate:[{start} TO {end}]"
    params = {
        "search_query": query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "start": 0,
        "max_results": max_results,
    }
    url = ARXIV_API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "ai-news-radar-papers/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    root = ET.fromstring(data)
    papers = []
    for entry in root.findall("atom:entry", ATOM_NS):
        id_url = (entry.findtext("atom:id", "", ATOM_NS) or "").strip()
        m = re.search(r"(\d{4}\.\d{4,5})(?:v\d+)?$", id_url)
        if not m:
            continue
        arxiv_id = m.group(1)
        title = re.sub(r"\s+", " ", entry.findtext("atom:title", "", ATOM_NS) or "").strip()
        summary = re.sub(r"\s+", " ", entry.findtext("atom:summary", "", ATOM_NS) or "").strip()
        published = (entry.findtext("atom:published", "", ATOM_NS) or "").strip()
        cats = [c.get("term", "") for c in entry.findall("atom:category", ATOM_NS)]
        papers.append({
            "id": arxiv_id,
            "title": title,
            "abstract": summary,
            "published": published,
            "categories": cats,
        })
    return papers


def load_seen_ids(posts_dir: str, today: datetime) -> set:
    """arXiv IDs already published in recent daily posts."""
    seen = set()
    if not os.path.isdir(posts_dir):
        return seen
    cutoff = (today - timedelta(days=SEEN_DAYS)).strftime("%Y-%m-%d")
    for name in os.listdir(posts_dir):
        if not (name.startswith("daily-") and name.endswith(".md")):
            continue
        date = name[len("daily-"):-len(".md")]
        if date < cutoff:
            continue
        try:
            with open(os.path.join(posts_dir, name), encoding="utf-8") as f:
                seen.update(ARXIV_ID_RE.findall(f.read()))
        except OSError:
            continue
    return seen


def rank_with_groq(candidates, api_key: str):
    """One Groq call: pick the top 3-5 most valuable papers. Returns list of dicts."""
    items_text = []
    for i, p in enumerate(candidates):
        abstract = p["abstract"][:400]
        items_text.append(
            f"[{i}] id={p['id']} published={p['published'][:10]} "
            f"categories={','.join(p['categories'])}\n"
            f"Title: {p['title']}\nAbstract: {abstract}"
        )
    prompt = (
        "你是 AI 研究编辑。用户的研究兴趣：" + INTERESTS + "\n\n"
        "下面是候选论文（均为一周内 arXiv 新论文）。请选出对该用户最有价值的 3-5 篇："
        "优先与用户兴趣高度相关、有实质性方法或结果、最好有开源代码的论文；"
        "避免纯综述、边角料、明显水文。同等条件下优先更新的论文。\n\n"
        + "\n\n".join(items_text) + "\n\n"
        "只返回 JSON 数组，不要其他文字，格式：\n"
        '[{"id": "arxiv id", "title_zh": "中文标题（一句话）", '
        '"reason": "一句话推荐理由（中文）"}]'
    )
    body = json.dumps({
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 2048,
    }).encode("utf-8")
    req = urllib.request.Request(
        GROQ_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"]
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    picked = json.loads(content)
    by_id = {p["id"]: p for p in candidates}
    result = []
    for entry in picked:
        paper = by_id.get(entry.get("id", ""))
        if not paper:
            continue
        result.append({
            "id": paper["id"],
            "title_zh": entry.get("title_zh") or paper["title"],
            "reason": entry.get("reason") or "",
            "published": paper["published"],
        })
        if len(result) >= PICK_COUNT:
            break
    return result


def fallback_pick(candidates):
    """No ranking available: take the newest papers, reason from abstract."""
    result = []
    for p in sorted(candidates, key=lambda c: c["published"], reverse=True)[:PICK_COUNT]:
        first_sentence = re.split(r"(?<=[.!?])\s+", p["abstract"], maxsplit=1)[0]
        result.append({
            "id": p["id"],
            "title_zh": p["title"],
            "reason": first_sentence[:100],
            "published": p["published"],
        })
    return result


def format_section(picked) -> str:
    lines = [SECTION_MARKER, ""]
    for p in picked:
        line = f"- [{p['title_zh']}](https://arxiv.org/abs/{p['id']})"
        if p.get("reason"):
            line += f" — {p['reason']}"
        lines.append(line)
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    no_rank = "--no-rank" in sys.argv
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    posts_dir = os.path.join(repo_root, "content", "posts")
    today = beijing_today()
    date_str = today.strftime("%Y-%m-%d")
    post_path = os.path.join(posts_dir, f"daily-{date_str}.md")

    if not os.path.isfile(post_path):
        print(f"NO_POST {date_str} (no digest post yet, skipping papers)")
        return 0
    with open(post_path, encoding="utf-8") as f:
        post = f.read()
    if SECTION_MARKER in post:
        print(f"SKIP {date_str} (papers section already present)")
        return 0

    until = datetime.now(timezone.utc)
    since = until - timedelta(days=DAYS)
    candidates = []
    seen_ids = set()
    for category, max_results in CATEGORIES:
        try:
            papers = fetch_category(category, max_results, since, until)
        except Exception as exc:  # one bad category must not kill the run
            print(f"WARN fetch {category} failed: {exc}")
            continue
        for p in papers:
            if p["id"] in seen_ids:
                continue
            seen_ids.add(p["id"])
            candidates.append(p)
    print(f"FETCHED {len(candidates)} candidates from arXiv (last {DAYS}d)")

    seen = load_seen_ids(posts_dir, today)
    fresh = [p for p in candidates if p["id"] not in seen]
    print(f"DEDUP removed {len(candidates) - len(fresh)} already-published, {len(fresh)} fresh")
    if not fresh:
        print("NO_CANDIDATES (nothing fresh to add)")
        return 0

    api_key = os.environ.get("GROQ_API_KEY", "")
    picked = []
    if not no_rank and api_key:
        try:
            picked = rank_with_groq(fresh, api_key)
            print(f"RANKED {len(picked)} papers via Groq")
        except Exception as exc:
            print(f"WARN Groq ranking failed ({exc}); falling back to recency")
    if not picked:
        picked = fallback_pick(fresh)
        print(f"FALLBACK picked {len(picked)} newest papers")

    section = format_section(picked)
    with open(post_path, "a", encoding="utf-8") as f:
        f.write("\n" + section)
    print(f"ADDED {len(picked)} papers to {post_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
