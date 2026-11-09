# LLM Atlas

**Watch an AI learn to write.** Type any text, and a real tiny transformer (a GPT-style model) trains on it live in your browser.
You can watch the training loop run, the loss fall, and the model's writing turn from noise into your words.

**Live site:** https://purvanshsingh.github.io/LLM-Atlas/

## Pages

| Page | What you can do |
|---|---|
| **Train a model** (`index.html`) | Pick a preset or type your own text and press *Start training*. Then explore, all live: **(1)** the Guess → Score → Blame → Nudge loop with a loss curve on training vs. unseen text (overfitting shows up on its own) and writing samples at each milestone; **(2) Inside the model**, a Transformer-Explainer-style flow of *your* model (tokens → embeddings → Q·K·V → masked multi-head attention → residual → GELU MLP → block output → next-letter odds), with every square a real activation, a guided step-by-step walkthrough, and a live backward-pass view of gradient size per layer; **(3)** one prediction up close; **(4)** a live 2D map of the character embeddings; **(5)** the model finishing your sentence. |
| **Under the hood** (`lab.html`) | Playgrounds for each piece: BPE tokenizer, attention heatmaps (causal mask, scaling, heads), GPT-2 architecture with live parameter counts, and temperature / top-k sampling. |

## How it works

- `tinygpt.js` is a dependency-free, character-level GPT: token and position embeddings, pre-LayerNorm blocks with causal multi-head attention and a GELU MLP, its own small autograd, and Adam with gradient clipping. About 28K parameters by default. Its gradients were checked against numerical gradients.
- `trainer-worker.js` runs training in a Web Worker so the page stays smooth. It streams weights back to the page about 8 times a second.
- Everything is static: no build step, no server, no API keys. Fonts come from Google Fonts.

## Run locally

```bash
python3 -m http.server 8000
```

Then open http://localhost:8000. Web Workers need `http://`. Opening the file directly falls back to training on the main thread.

## Deploy to GitHub Pages

1. Push this folder to `main` of a public repo (this one: `Purvanshsingh/LLM-Atlas`).
2. Repo → **Settings → Pages → Build and deployment → Source: Deploy from a branch** → `main` / `/ (root)` → Save.
3. About a minute later it's live at https://purvanshsingh.github.io/LLM-Atlas/

## Credits

Made with ♥ by Purvansh ([LinkedIn](https://www.linkedin.com/in/purvanshsingh/)).

An independent educational project inspired by Sebastian Raschka's *Build a Large Language Model (From Scratch)* (Manning).
Not affiliated with or endorsed by the author or publisher. Official book code: https://github.com/rasbt/LLMs-from-scratch.
