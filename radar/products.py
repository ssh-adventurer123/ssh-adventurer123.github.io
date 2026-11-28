#!/usr/bin/env python3
"""Add a daily 'Product Hunt new products' section to today's blog post,
as the third section after tech news and AI papers: a group in the summary
list at the top, and a detail section right after the AI-papers detail
section.

Picks 3-5 products launched in the last 24 hours from Product Hunt's
public Atom feed (https://www.producthunt.com/feed, no auth), most relevant
to the user's interests (AI tools, coding agents, LLM apps, Vibe Coding,
developer tools), with no repeats.

How it works:
  1. Fetch the Product Hunt Atom feed (50 entries, ordered by recent
     activity); keep entries published in the last 24 hours.
  2. Drop products already published in the last 7 days of blog posts
     (matched by product page slug).
  3. One Groq call ranks the rest and picks the top 3-5 (Chinese names +
     one-line reasons). Falls back to feed order if ranking fails.
  4. Insert the section into today's post. Idempotent: any previous
     products section is removed first, then re-inserted.

Stdlib only. Reads GROQ_API_KEY from the environment for ranking.
"""
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

PH_FEED = "https://www.producthunt.com/feed"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"
USER_AGENT = "ai-news-radar-products/1.0"

HOURS = 24            # pool: products launched in the last 24 hours
SEEN_DAYS = 7         # dedup against posts from the last 7 days
PICK_COUNT = 5        # aim for 3-5 products
DETAIL_HEADING = "## Product Hunt 新品"
INDEX_HEADING = "**Product Hunt 新品**"

INTERESTS = (
    "用户关注好产品，尤其是 AI 相关：AI 工具、编程智能体 / 代码助手、"
    "大语言模型应用、Vibe Coding、AI 开发工具链、开发者工具。"
    "其次是有新意、真正实用的新产品。"
)

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
PH_SLUG_RE = re.compile(r"producthunt\.com/products/([a-z0-9][a-z0-9-]*)")
NUMBERED_RE = re.compile(r"^\d+\.\s")


def beijing_today() -> datetime:
    return datetime.now(timezone(timedelta(hours=8)))


def strip_tags(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_products():
    """Fetch the Product Hunt Atom feed; return list of product dicts."""
    req = urllib.request.Request(PH_FEED, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    root = ET.fromstring(data)
    products = []
    for entry in root.findall("atom:entry", ATOM_NS):
        url = ""
        for link in entry.findall("atom:link", NS := ATOM_NS):
            if link.get("rel") in (None, "alternate"):
                url = link.get("href", "")
                break
        m = PH_SLUG_RE.search(url or "")
        if not m:
            continue
        slug = m.group(1)
        name = re.sub(r"\s+", " ", entry.findtext("atom:title", "", ATOM_NS) or "").strip()
        content = entry.findtext("atom:content", "", ATOM_NS) or ""
        # Tagline is the first <p> in the content HTML.
        pm = re.search(r"<p[^>]*>(.*?)</p>", content, re.DOTALL | re.IGNORECASE)
        tagline = strip_tags(pm.group(1)) if pm else ""
        published = (entry.findtext("atom:published", "", ATOM_NS) or "").strip()
        products.append({
            "slug": slug,
            "name": name,
            "url": url,
            "tagline": tagline,
            "published": published,
        })
    return products


def parse_time(value: str):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def load_seen_slugs(posts_dir: str, today: datetime) -> set:
    """Product Hunt slugs already published in recent daily posts."""
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
                seen.update(PH_SLUG_RE.findall(f.read()))
        except OSError:
            continue
    return seen


def groq_rank(candidates, api_key: str):
    """One Groq call: pick the top 3-5 most relevant products. Returns list of dicts."""
    items_text = []
    for i, p in enumerate(candidates):
        items_text.append(
            f"[{i}] slug={p['slug']} published={p['published'][:10]}\n"
            f"Name: {p['name']}\nTagline: {p['tagline']}"
        )
    prompt = (
        "你是 AI 产品编辑。" + INTERESTS + "\n\n"
        "下面是候选新品（均为 24 小时内在 Product Hunt 上线的产品）。"
        "请选出对该用户最有价值的 3-5 个：优先 AI 相关、有实质功能、"
        "能解决真实问题的产品；避免纯营销、换皮、明显水货。"
        "同等条件下优先更新的产品。\n\n"
        + "\n\n".join(items_text) + "\n\n"
        "只返回 JSON 数组，不要其他文字，格式：\n"
        '[{"slug": "product slug", "name_zh": "中文产品名（一句话，可保留英文原名）", '
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
    by_slug = {p["slug"]: p for p in candidates}
    result = []
    for entry in picked:
        product = by_slug.get(entry.get("slug", ""))
        if not product:
            continue
        result.append({
            "slug": product["slug"],
            "name_zh": entry.get("name_zh") or product["name"],
            "reason": entry.get("reason") or "",
            "url": product["url"],
            "published": product["published"],
        })
        if len(result) >= PICK_COUNT:
            break
    return result


def rank_with_groq(candidates, api_key: str):
    """Rank with retries; log the provider's error body for diagnosis.

    Groq's free tier also caps tokens per MINUTE (8,000 TPM) on top of the
    daily 200k budget. The papers step just before this one uses most of
    that minute's budget, so wait 60s first to let the per-minute window
    clear, and wait 60s between retries.
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
    """No ranking available: take the first products in feed order."""
    result = []
    for p in candidates[:PICK_COUNT]:
        result.append({
            "slug": p["slug"],
            "name_zh": p["name"],
            "reason": p["tagline"],
            "url": p["url"],
            "published": p["published"],
        })
    return result


def format_date(iso: str) -> str:
    dt = parse_time(iso)
    if dt is None:
        return iso[:10]
    return f"{dt.month}月{dt.day}日"


def build_index_group(picked) -> list:
    lines = [INDEX_HEADING]
    for i, p in enumerate(picked, start=1):
        lines.append(f"{i}. [{p['name_zh']}](#item-ph-{i})")
    return lines


def build_detail_section(picked) -> list:
    lines = [DETAIL_HEADING, ""]
    for i, p in enumerate(picked, start=1):
        lines.append(f'<a id="item-ph-{i}"></a>')
        lines.append(f"### [{p['name_zh']}]({p['url']})")
        lines.append("")
        if p.get("reason"):
            lines.append(p["reason"])
            lines.append("")
        lines.append(f"Product Hunt · {format_date(p['published'])}")
        lines.append("")
    return lines


def remove_existing_products(lines: list) -> list:
    """Drop any previous products index group and detail section."""
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line == INDEX_HEADING:
            i += 1
            while i < len(lines) and NUMBERED_RE.match(lines[i]):
                i += 1
            if i < len(lines) and lines[i] == "":
                i += 1
            continue
        if line == DETAIL_HEADING:
            i += 1
            while i < len(lines) and not lines[i].startswith("## "):
                i += 1
            continue
        out.append(line)
        i += 1
    return out


def insert_products(lines: list, picked) -> list:
    """Insert the index group after the AI-papers index group (else after
    tech news) and the detail section right after the AI-papers detail
    section (else after tech news)."""
    index_group = build_index_group(picked)
    # --- index group ---
    anchor = None
    for j, line in enumerate(lines):
        if line == "**AI 论文**":
            anchor = j
            break
    if anchor is None:
        for j, line in enumerate(lines):
            if line == "**科技新闻**":
                anchor = j
                break
    if anchor is not None:
        k = anchor + 1
        while k < len(lines) and NUMBERED_RE.match(lines[k]):
            k += 1
        insert_at = k
        block = [""] + index_group
        if insert_at >= len(lines) or lines[insert_at] != "":
            block.append("")
        lines = lines[:insert_at] + block + lines[insert_at:]
    # --- detail section ---
    detail = build_detail_section(picked)
    h2 = None
    for j, line in enumerate(lines):
        if line == "## AI 论文":
            h2 = j
            break
    if h2 is None:
        for j, line in enumerate(lines):
            if line == "## 科技新闻":
                h2 = j
                break
    if h2 is None:
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
        while lines and lines[-1] == "":
            lines.pop()
        if lines and lines[-1] == "---":
            lines.pop()
        if lines and lines[-1] != "":
            lines.append("")
        lines.extend(["---", ""] + detail)
        return lines
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
        print(f"NO_POST {date_str} (no digest post yet, skipping products)")
        return 0
    with open(post_path, encoding="utf-8") as f:
        post = f.read()

    try:
        all_products = fetch_products()
    except Exception as exc:
        print(f"WARN fetch Product Hunt feed failed: {exc}")
        return 0
    print(f"FETCHED {len(all_products)} products from Product Hunt feed")
    cutoff = datetime.now(timezone.utc) - timedelta(hours=HOURS)
    recent = []
    for p in all_products:
        pub = parse_time(p["published"])
        if pub is not None and pub >= cutoff:
            recent.append(p)
    print(f"WINDOW {len(recent)} launched in the last {HOURS}h")

    seen = load_seen_slugs(posts_dir, today)
    fresh = [p for p in recent if p["slug"] not in seen]
    print(f"DEDUP removed {len(recent) - len(fresh)} already-published, {len(fresh)} fresh")
    if not fresh:
        lines = remove_existing_products(post.split("\n"))
        if lines != post.split("\n"):
            with open(post_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print("REMOVED stale products section (no fresh candidates)")
        else:
            print("NO_CANDIDATES (nothing fresh to add)")
        return 0

    api_key = os.environ.get("GROQ_API_KEY", "")
    picked = []
    if not no_rank and api_key:
        try:
            picked = rank_with_groq(fresh, api_key)
            print(f"RANKED {len(picked)} products via Groq")
        except Exception as exc:
            print(f"WARN Groq ranking failed ({exc}); falling back to feed order")
    if not picked:
        picked = fallback_pick(fresh)
        print(f"FALLBACK picked {len(picked)} products in feed order")

    lines = remove_existing_products(post.split("\n"))
    lines = insert_products(lines, picked)
    with open(post_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"ADDED {len(picked)} products to {post_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
