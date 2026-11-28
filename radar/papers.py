#!/usr/bin/env python3
"""Add a daily 'AI papers' section to today's blog post, as a peer of the
'tech news' section: a group in the summary list at the top, and a detail
section right after the tech-news detail section.

Picks 3-5 arXiv papers from the last 7 days, most valuable for the user's
research interests (LLM, multi-agent systems, context engineering / ACE,
Vibe Coding, AI toolchain), with no repeats.

How it works:
  1. Query the arXiv API per category (last 7 days, newest first, capped).
  2. Drop papers already published in the last 8 days of blog posts.
  3. One Groq call ranks the rest and picks the top 3-5 (Chinese titles +
     one-line reasons). Falls back to recency order if ranking fails.
  4. Insert the section into today's post. Idempotent: any previous papers
     section is removed first, then re-inserted.

Stdlib only. Reads GROQ_API_KEY from the environment for ranking.
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

ARXIV_API = "https://export.arxiv.org/api/query"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"
USER_AGENT = "ai-news-radar-papers/1.0"

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
SECTION_TITLE = "AI 论文"
DETAIL_HEADING = "## AI 论文"
INDEX_HEADING = "**AI 论文**"

INTERESTS = (
    "大语言模型 (LLM)：从基础理论到实际应用；"
    "多代理系统 (MAS)：智能体协作、通信机制、设计模式；"
    "上下文工程：提升模型理解和生成能力，以及 ACE (Agentic Context Engineering)；"
    "Vibe Coding：使用 AI 编程提升个人全方位效能；"
    "AI 工具链：构建高效的 AI 开发工具"
)

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
ARXIV_ID_RE = re.compile(r"arxiv\.org/abs/(\d{4}\.\d{4,5})")
NUMBERED_RE = re.compile(r"^\d+\.\s")


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
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
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


def groq_rank(candidates, api_key: str):
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
            "User-Agent": USER_AGENT,
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
            "categories": paper["categories"],
        })
        if len(result) >= PICK_COUNT:
            break
    return result


def rank_with_groq(candidates, api_key: str):
    """Rank with retries; log the provider's error body for diagnosis.

    Groq's free tier also caps tokens per MINUTE (8,000 TPM) on top of the
    daily 200k budget, and a ranking request is ~6k tokens. The news
    enrichment just before this step uses most of that minute's budget, so
    wait 60s first to let the per-minute window clear, and wait 60s between
    retries (the provider suggests ~35s; 60s is a safe margin).
    """
    time.sleep(60)
    last_exc = None
    for attempt in (1, 2):
        try:
            return groq_rank(candidates, api_key)
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", "replace")[:500]
            except Exception:
                detail = ""
            print(f"WARN Groq ranking attempt {attempt} failed: HTTP {exc.code} {detail}")
            last_exc = exc
        except Exception as exc:
            print(f"WARN Groq ranking attempt {attempt} failed: {exc}")
            last_exc = exc
        if attempt == 1:
            time.sleep(60)
    raise last_exc


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
            "categories": p["categories"],
        })
    return result


def format_date(iso: str) -> str:
    try:
        dt = datetime.strptime(iso[:10], "%Y-%m-%d")
        return f"{dt.month}月{dt.day}日"
    except ValueError:
        return iso[:10]


def format_meta(paper) -> str:
    cats = [c for c in paper.get("categories", []) if c.startswith("cs.")][:3]
    parts = ["arXiv"]
    if cats:
        parts.append(", ".join(cats))
    if paper.get("published"):
        parts.append(format_date(paper["published"]))
    return " · ".join(parts)


def build_index_group(picked) -> list:
    lines = [INDEX_HEADING]
    for i, p in enumerate(picked, start=1):
        lines.append(f"{i}. [{p['title_zh']}](#item-ai-papers-{i})")
    return lines


def build_detail_section(picked) -> list:
    lines = [DETAIL_HEADING, ""]
    for i, p in enumerate(picked, start=1):
        lines.append(f'<a id="item-ai-papers-{i}"></a>')
        lines.append(f"### [{p['title_zh']}](https://arxiv.org/abs/{p['id']})")
        lines.append("")
        if p.get("reason"):
            lines.append(p["reason"])
            lines.append("")
        lines.append(format_meta(p))
        lines.append("")
    return lines


def remove_existing_papers(lines: list) -> list:
    """Drop any previous papers index group and detail section."""
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line == INDEX_HEADING:
            # Skip the heading, its numbered items, and one trailing blank line.
            i += 1
            while i < len(lines) and NUMBERED_RE.match(lines[i]):
                i += 1
            if i < len(lines) and lines[i] == "":
                i += 1
            continue
        if line == DETAIL_HEADING:
            # Skip until the next H2 heading or end of file.
            i += 1
            while i < len(lines) and not lines[i].startswith("## "):
                i += 1
            continue
        out.append(line)
        i += 1
    return out


def insert_papers(lines: list, picked) -> list:
    """Insert the index group after the tech-news index group and the detail
    section right after the tech-news detail section."""
    # --- index group: after the "**科技新闻**" group's numbered items ---
    index_group = build_index_group(picked)
    idx = None
    first_h2 = None
    for j, line in enumerate(lines):
        if first_h2 is None and line.startswith("## "):
            first_h2 = j
        if line == "**科技新闻**":
            idx = j
            break
    if idx is not None:
        k = idx + 1
        while k < len(lines) and NUMBERED_RE.match(lines[k]):
            k += 1
        # k is the blank line after the numbered items (or the next heading).
        insert_at = k
        block = [""] + index_group
        if insert_at >= len(lines) or lines[insert_at] != "":
            block.append("")
        lines = lines[:insert_at] + block + lines[insert_at:]
    # --- detail section: right after the "## 科技新闻" section ---
    detail = build_detail_section(picked)
    h2 = None
    for j, line in enumerate(lines):
        if line == "## 科技新闻":
            h2 = j
            break
    if h2 is None:
        # No tech-news section (e.g. empty news day): append at the end.
        if lines and lines[-1] != "":
            lines.append("")
        lines.extend(detail)
        return lines
    nxt = None
    for j in range(h2 + 1, len(lines)):
        if lines[j].startswith("## "):
            nxt = j
            break
    if nxt is None:
        # Strip a dangling divider left by a removed papers section.
        while lines and lines[-1] == "":
            lines.pop()
        if lines and lines[-1] == "---":
            lines.pop()
        if lines and lines[-1] != "":
            lines.append("")
        lines.extend(["---", ""] + detail)
        return lines
    # Insert before the next H2, keeping the "---" divider that precedes it.
    insert_at = nxt
    if insert_at >= 2 and lines[insert_at - 1] == "" and lines[insert_at - 2] == "---":
        insert_at -= 2
    block = detail + ["---", ""]
    lines = lines[:insert_at] + block + lines[insert_at:]
    return lines


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
        # Still remove a stale section if the post has one but nothing fresh.
        lines = remove_existing_papers(post.split("\n"))
        if lines != post.split("\n"):
            with open(post_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print("REMOVED stale papers section (no fresh candidates)")
        else:
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

    lines = remove_existing_papers(post.split("\n"))
    lines = insert_papers(lines, picked)
    with open(post_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"ADDED {len(picked)} papers to {post_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
