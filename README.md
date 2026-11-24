# Mihir Padsumbiya · portfolio

Personal site of Mihir Padsumbiya, data scientist, live at https://mihirp381.github.io/mihirpadsumbiya/.
It is a static page (HTML, one stylesheet, ES modules) with no build step and no dependencies.

## Preview locally

```
python3 -m http.server 8000
```

Run it from the repo root and open http://localhost:8000/. ES modules do not load from `file://`.

## Models

Every model on the page runs client-side: the hero network trains in the browser (`assets/js/lib/mlp.js`, checked by `node tools/hero_check.mjs`), and the Lab figures load small exported models from `assets/models/`.
All of them are reproducible from `tools/`; see [tools/README.md](tools/README.md).
