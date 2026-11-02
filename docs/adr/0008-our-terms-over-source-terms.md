# Our terms over the terms of the sources

The two glossaries are the authority for the words of this crate. We do not
rename a term to match a paper or a reference implementation. The sources
disagree with each other: "layer" can mean our block, our plant or our family.
So no source gives one vocabulary to follow. A reader of the papers must learn
our words, and `docs/source-terms.md` maps the source words to them.

## Consequences

- A rename needs one of two reasons. The term is wrong by our own definitions,
  or its risk of a wrong port is worth the cost.
- `Block` stays, although the official class `Block` is our layer. The official
  `mixer` is our block, but Multi-Gate Residuals uses "mixer" for its stream
  update. So that rename would only move the clash.
- `_Avoid_` applies to our own words. A doc can use a source word in quotes or
  backticks when it cites that source.
