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

## Changes in both repos

Some changes touch this repo and `../burn-stack`. Start the session in this repo: `domain.md` loads both glossaries, and `CLAUDE.md` permits writes to `../burn-stack`.

- **Spec**: one `spec.md` in this repo.
- **Tickets**: each ticket changes one repo. A burn-stack ticket goes to `../burn-stack/.scratch/<feature-slug>/issues/`, with the same slug.
- **Blocking**: the `Blocked by:` line gives the path of a blocker in the other repo.
- **Review**: `/code-review` reads the diff of one repo. Run it once in each changed repo, with the tickets of that repo as the spec source.

### The burn-stack pin

`Cargo.toml` pins burn-stack to a GitHub rev, in `[dependencies]` and in `[dev-dependencies]`. While a change is in progress, use one of these:

- **No other active session or parallel agent** (`ListAgents` shows them): enable the `[patch]` at the end of `Cargo.toml`. The build then uses the working tree of `../burn-stack`.
- **Another active session or parallel agent**: it can change that working tree. Push each burn-stack commit, then move the pin to that commit.

Each burn-mamba commit builds against the pin. Before you commit in this repo:

1. Push the burn-stack commits.
2. Set both `rev` values to the pushed commit.
3. Comment out the `[patch]`.
4. Run `cargo check`, which updates `Cargo.lock`.

## Wayfinding operations

`/wayfinder` uses these. The **map** is one file, with one **child** file per ticket.

- **Map**: `.scratch/<effort>/map.md` (the Notes / Decisions-so-far / Fog body).
- **Child ticket**: `.scratch/<effort>/issues/NN-<slug>.md`, numbered from `01`, with the question in the body. A `Type:` line records the ticket type (`research`/`prototype`/`grilling`/`task`). A `Status:` line records `claimed`/`resolved`.
- **Blocking**: a `Blocked by: NN, NN` line near the top. A ticket is unblocked when each file in that line is `resolved`.
- **Frontier**: scan `.scratch/<effort>/issues/` for the files that are open, unblocked and unclaimed. The lowest number wins.
- **Claim**: set `Status: claimed` and save the file before any other work.
- **Resolve**: write the answer under an `## Answer` heading and set `Status: resolved`. Then add a context pointer (gist + link) to the Decisions-so-far of `map.md`.
