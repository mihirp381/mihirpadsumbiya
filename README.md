# Mihir Padsumbiya · portfolio

Personal site of Mihir Padsumbiya, data scientist, live at https://mihirp381.github.io/mihirpadsumbiya/.

GitHub Pages builds it with Jekyll from the root of `main`. There is no Gemfile, no Sass and no custom plugin; only the plugins GitHub Pages ships with (`jekyll-feed`, `jekyll-seo-tag`, `jekyll-sitemap`) are used.

## Structure

```
_config.yml        site settings, permalinks, plugins
_data/             page content as YAML: experience, education, publications, projects, exhibits, navigation
_layouts/          default, page, exhibit (one Lab figure), post (one note)
_includes/         head, masthead, footer, contents list, figure, pull quote, title, reading time
_posts/            notes, one Markdown file each
_drafts/           post-template.md, a note that shows every supported element
index.html         home (folio 1)
lab/               Laboratory index and one folder per exhibit (folio 2)
work/              experience, education, certifications, skills (folio 3)
research/          papers and projects (folio 4)
notes/             notes index and the Atom feed (folio 5)
about/             biography, recommendations, contact (folio 6)
404.html           page not found
assets/css/        site.css, the whole stylesheet
assets/js/         site.js (theme, ranges), hero.js (home figure), lab.js and demos/ (exhibits), lib/ (models)
assets/models/     exported models loaded by the exhibits
assets/notes/      images used in notes
tools/             training, export and check scripts; not published
```

Every internal link goes through `relative_url`, so the site works under `/mihirpadsumbiya/`.

## Preview locally

```
tools/preview.sh [port]
```

It builds the site with the same image GitHub Pages uses (`ghcr.io/actions/jekyll-build-pages`, needs Docker) into a `_preview` folder next to the repository and serves it at http://127.0.0.1:8731/mihirpadsumbiya/ (or the port given). The warnings about GitHub API authentication and a missing git repository are harmless.

## Writing a note

1. Copy `_drafts/post-template.md` to `_posts/YYYY-MM-DD-short-title.md`. The date and the file name set the address, `/notes/YYYY/MM/short-title/`.
2. Fill in the front matter: `title`, `summary` (one or two sentences, shown on the notes index, in the feed and as the page description) and `tags`.
3. Write the body in Markdown. The first paragraph gets the drop cap and the last one gets the end mark.
4. Link to pages with `{{ '/lab/' | relative_url }}` and to other notes with `{{ site.baseurl }}{% post_url 2026-11-24-three-models-and-no-server %}`.
5. Put images in `assets/notes/` and use `{% include figure.html src="/assets/notes/name.jpg" alt="..." caption="..." %}`.
6. Preview, then push to `main`. A note dated in the future is not published until that date.

Tables, footnotes (`[^name]`), fenced code with a language, block quotes and lists are all styled. Reading time is the word count divided by 220, at least one minute.

## Checks

```
node tools/parity.mjs
node tools/hero_check.mjs
node tools/demo_check.mjs
```

`parity.mjs` compares the exported tree models with the Python predictions, `hero_check.mjs` checks the network on the home page, and `demo_check.mjs` checks the logic behind each exhibit. Run all three before changing anything in the Lab. How the models are trained and exported is in [tools/README.md](tools/README.md).
