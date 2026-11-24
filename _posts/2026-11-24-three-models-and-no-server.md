---
title: Three models and no server
summary: How the figures in my Laboratory run trained models inside the browser tab, with no server and no machine learning library.
tags: [lab, javascript, machine learning]
---

The [Laboratory]({{ '/lab/' | relative_url }}) on this site has five exhibits: three trained models, a searchable dataset and a Streamlit app. None of them talks to a server. When you move a slider on the concrete page, the new prediction is computed by your own machine, inside the tab you are reading. This note is about how that works.

<!--more-->

## Why no server

The usual route would be to wrap each model in a small API and call it from the page. This site is static files on GitHub Pages, so an API would mean hosting, cold starts and one more thing to keep alive. The models are also small. Once that was clear, the real question was how to run them without loading a machine learning library that would outweigh the models themselves.

## Trees are arrays

Two of the models, [concrete strength]({{ '/lab/concrete-strength/' | relative_url }}) and [customer churn]({{ '/lab/customer-churn/' | relative_url }}), are gradient-boosted trees trained with scikit-learn. A fitted tree is a handful of parallel arrays: for each node, the feature it splits on, the threshold, the left child, the right child, and a value at the leaves. I export those arrays to JSON, one object per tree, and walk them in the browser. The whole evaluator in `assets/js/lib/gbm.js` is 20 lines. This is the loop that does the work:

```js
for (const tr of model.trees) {
  const f = tr.f, t = tr.t, l = tr.l, r = tr.r;
  let n = 0;
  while (f[n] !== -1) n = x32[f[n]] <= t[n] ? l[n] : r[n];
  s += lr * tr.v[n];
}
```

A node whose feature is `-1` is a leaf. Each tree adds its leaf value, scaled by the learning rate, to a running score. For the churn model the score then goes through a sigmoid to become a probability.

The detail that needed care is `x32`. scikit-learn casts its inputs to 32-bit floats before comparing them with a threshold, while JavaScript numbers are 64-bit. For most inputs that changes nothing, but a value very close to a threshold can land on the other side of it, and then a whole tree returns a different leaf. So the evaluator casts every input with `Math.fround` first, and the exporter writes each threshold as the shortest decimal that splits 32-bit inputs exactly where scikit-learn splits them. With both in place, the browser reproduces scikit-learn's own decisions and matches its predictions to within 1e-9.[^parity]

## A network written out by hand

The [crack detection]({{ '/lab/crack-detection/' | relative_url }}) exhibit runs the convolutional network from my 2022 paper, retrained in PyTorch. It has 954,241 parameters. As 32-bit floats that is about 3.8 MB, which is a lot to ask of someone who only came to read a page. Two weight tensors hold nearly all of the parameters, so I store those two as 8-bit integers with one scale factor each and keep the rest as floats. That puts 99.6% of the weights in 8 bits and the file at 0.97 MB. On the 4,000 test images the quantised model scores exactly what the 32-bit one did.

There is no library on the JavaScript side either. The forward pass, two convolutions with ReLU and max pooling followed by two dense layers, is written out by hand. It runs in a Web Worker so the page stays responsive, and one photo takes about 200 ms. Its output agrees with PyTorch, using the same shipped weights, to 3.9e-8.

{% include figure.html src="/assets/notes/crack-sample.jpg" alt="A grey concrete surface with a dark crack running across it, shown as enlarged pixels." caption="One of the sample photos on the crack-detection page, enlarged. The network reads every photo at 128 × 128 pixels." width="256" height="256" pixelated=true %}

## What it costs

| Model | Size on disk | Held-out result |
| --- | ---: | --- |
| Concrete strength, 400 trees | 260 kB | R² 0.94 on 201 mixes |
| Customer churn, 250 trees | 94 kB | ROC-AUC 0.82 on 4,882 customers |
| Crack detection, CNN | 0.97 MB | 98.8% accuracy on 4,000 images |

Nothing loads until you open the page that needs it, so the home page never pays for the network. The [drug–transporter]({{ '/lab/drug-transporters/' | relative_url }}) exhibit works the same way: its 8,094 compounds arrive as one JSON file when you open that page, and the search runs locally. Nothing you type, pick or upload is sent anywhere.

The fifth exhibit is the exception to the no-library rule. [ChurnGuard, the app]({{ '/lab/churnguard-app/' | relative_url }}) is my team's hackathon entry, and it runs real Streamlit on Python compiled to WebAssembly. That runtime is about 31 MB, so it is fetched from a CDN only when you press Start. It scores with the same 250 trees as the churn exhibit, walked in Python instead of JavaScript.

## Checking the copy against the original

A model that runs in a different language is a copy, and a copy can be wrong in ways that look right. The repository has a small Node script, `tools/parity.mjs`, that loads the exported files, runs the same inputs Python ran, and exits with an error if any prediction drifts past its tolerance. Three more scripts check the network on the home page, the logic behind each exhibit and the Python copy of the churn trees. All four run before any change to the Lab goes out. A model that is right almost everywhere is the hardest kind of wrong to notice, and a check like this is the cheapest way I know to rule it out.

[^parity]: The tolerance in the check is 1e-9. The largest difference it currently finds is about 2e-11, for both tree models.
