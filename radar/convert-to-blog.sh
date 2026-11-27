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

# The digest filename uses the UTC date, but the run happens at 07:23 Beijing
# (23:23 UTC the day before), so the UTC date is one day behind what the
# reader sees. Label the post with the Beijing date instead.
date=$(TZ=Asia/Shanghai date +%Y-%m-%d)
target="content/posts/daily-$date.md"
new_body=$(strip_frontmatter "$newest")

# Build the full post and compare whole files: the front matter template
# adds its own blank line, so comparing bare bodies gives false positives.
tmp=$(mktemp)
{
  printf -- '---\ntitle: "每日AI动态 - %s"\ndate: %s\ncategories: ["daily-ai-news"]\n---\n\n' "$date" "$date"
  printf '%s\n' "$new_body"
} > "$tmp"

if [ -f "$target" ] && cmp -s "$tmp" "$target"; then
  rm -f "$tmp"
  echo "NO_CHANGE $date"
  exit 0
fi
mv "$tmp" "$target"
echo "WROTE $target"
