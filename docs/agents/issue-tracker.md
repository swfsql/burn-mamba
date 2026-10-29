# Issue tracker: Local Markdown

Issues and specs for this repo are markdown files in `.scratch/`.

## Conventions

- Each feature has one directory: `.scratch/<feature-slug>/`.
- The spec is `.scratch/<feature-slug>/spec.md`.
- Each implementation ticket is one file: `.scratch/<feature-slug>/issues/<NN>-<slug>.md`. The numbers start at `01`. Never put all the tickets in one combined file.
- A `Status:` line near the top of each issue file records the triage state. `triage-labels.md` gives the role strings.
- Comments and conversation history go at the bottom of the file, under a `## Comments` heading.

## When a skill says "publish to the issue tracker"

Create a new file under `.scratch/<feature-slug>/`. Create the directory if it does not exist.

## When a skill says "fetch the relevant ticket"

Read the file at the given path. The user usually gives the path or the issue number.

## Wayfinding operations

`/wayfinder` uses these. The **map** is one file, with one **child** file per ticket.

- **Map**: `.scratch/<effort>/map.md` (the Notes / Decisions-so-far / Fog body).
- **Child ticket**: `.scratch/<effort>/issues/NN-<slug>.md`, numbered from `01`, with the question in the body. A `Type:` line records the ticket type (`research`/`prototype`/`grilling`/`task`). A `Status:` line records `claimed`/`resolved`.
- **Blocking**: a `Blocked by: NN, NN` line near the top. A ticket is unblocked when each file in that line is `resolved`.
- **Frontier**: scan `.scratch/<effort>/issues/` for the files that are open, unblocked and unclaimed. The lowest number wins.
- **Claim**: set `Status: claimed` and save the file before any other work.
- **Resolve**: write the answer under an `## Answer` heading and set `Status: resolved`. Then add a context pointer (gist + link) to the Decisions-so-far of `map.md`.
