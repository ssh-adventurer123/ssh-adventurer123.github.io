# Wang 的 AI 博客

Hugo + PaperMod + GitHub Pages，不用服务器。

- 线上地址：https://ssh-adventurer123.github.io/
- 新闻雷达（独立站，不动它）：https://ssh-adventurer123.github.io/ai-news-radar/

## 本地跑起来

```bash
hugo server -D
# 打开 http://localhost:1313
```

## 发一篇文章

在 `content/posts/` 新建一个 `.md` 文件，头上写：

```yaml
---
title: "文章标题"
date: 2026-11-27
categories: ["my-insights"]
tags: ["AI"]
---
```

正文用 Markdown 写，推到 GitHub 的 main 分支就自动上线。

栏目对应关系（categories 的值）：

- 个人洞察 `my-insights`
- 名人洞察 `celebrity-insights`
- 智能体与编程 `ai-programming`
- 模型与研究 `large-models`
- 开源项目 `projects`
- 产品与大厂 `products`

## 新闻雷达是怎么“放进来”的

没有把雷达的代码搬进这个仓库。雷达还是它自己的仓库
`ssh-adventurer123/ai-news-radar`，每天 07:23 自动跑、单独上线。

这个博客只是在顶部菜单加了一个外链「每日AI动态」，点过去就是雷达站。
这样两边互不影响：博客坏了雷达照跑，雷达改了博客不用动。
