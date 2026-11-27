#!/usr/bin/env bash
# Convert the newest Horizon zh digest (radar/docs/_posts/*-summary-zh.md)
# into a blog post (content/posts/daily-YYYY-MM-DD.md).
# Idempotent: rewrites the post only when the digest body changed.
set -euo pipefail
cd "$(dirname "$0")/.."

strip_frontmatter() {
  # Skip only the first two "---" lines (front matter); keep "---" rules in the body.
  awk 'c<2 && /^---$/ {c++; next} c>=2 {print}' "$1"
}

newest=$(ls radar/docs/_posts/*-summary-zh.md 2>/dev/null | sort | tail -1 || true)
if [ -z "$newest" ]; then
  echo "NO_DIGEST"
  exit 0
fi

date=$(basename "$newest" | cut -c1-10)
target="content/posts/daily-$date.md"
new_body=$(strip_frontmatter "$newest")

if [ -f "$target" ]; then
  old_body=$(strip_frontmatter "$target")
  if [ "$new_body" = "$old_body" ]; then
    echo "NO_CHANGE $date"
    exit 0
  fi
fi

{
  printf -- '---\ntitle: "每日AI动态 - %s"\ndate: %s\ncategories: ["daily-ai-news"]\n---\n\n' "$date" "$date"
  printf '%s\n' "$new_body"
} > "$target"
echo "WROTE $target"
