# Rohith Behera - Portfolio

Personal portfolio website with integrated blog system.

## Setup

Host on GitHub Pages:

1. Push to repository
2. Go to Settings → Pages
3. Source: Deploy from branch `main`
4. Folder: `/ (root)`

## Adding Blog Posts

1. Create a markdown file in `blogs/` folder (e.g., `my-new-post.md`)
2. Add frontmatter:

```markdown
---
title: "Your Post Title"
date: "2025-01-20"
author: "Rohith Behera"
readTime: 5
tags: ["Tag1", "Tag2"]
---

Your content here...
```

3. Update `blogs/index.json`:

```json
{
  "slug": "my-new-post",
  "title": "Your Post Title",
  "date": "2025-01-20",
  "excerpt": "Brief description...",
  "tags": ["Tag1", "Tag2"],
  "readTime": 5
}
```

## Structure

```
/
├── index.html       # Main portfolio
├── blog.html        # Blog post viewer
├── css/style.css    # Styles
├── js/main.js       # Scripts
└── blogs/
    ├── index.json   # Blog index
    └── *.md         # Blog posts
```

