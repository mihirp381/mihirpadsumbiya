---
title: Title of the note
summary: One or two sentences. They appear on the notes index, in the RSS feed and as the page description.
tags: [example, template]
---

The opening paragraph. It gets the drop cap. Link to another page with the relative_url filter, for example the [project demos]({{ '/lab/' | relative_url }}). Link to another note with the post_url tag and the note's file name without the extension, as described in the README.

<!--more-->

## A section heading

Body text with *italic*, **bold** and `inline code`. A footnote goes here.[^one]

### A smaller heading

- A bulleted item
- Another item

1. A numbered item
2. Another item

> A quotation set apart from the text.

| Column | Number |
| --- | ---: |
| First row | 1.0 |
| Second row | 22.5 |

```python
def add(a, b):
    return a + b
```

Images go in `assets/notes/` and use the figure include:

{% include figure.html src="/assets/notes/crack-sample.jpg" alt="Describe the image for someone who cannot see it." caption="The caption, set in italic under the frame." %}

---

The last paragraph ends with the end mark.

[^one]: The footnote text.
