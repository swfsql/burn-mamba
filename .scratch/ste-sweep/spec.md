# STE sweep: the prose of both repos follows the updated `asd-ste100` skill

Status: ready-for-agent

## Problem Statement

The linter of the `asd-ste100` skill has new rules for long paragraphs,
phrasal verbs with a particle and change history. It can also read the strings
of panic, assert and expect calls. It reads only the prose of a `.rs` file. The
last prose sweeps of burn-mamba and burn-stack came before these changes.

The updated linter reads every tracked `.rs` and `.md` file outside `tmp/`:

| Repo | Hard findings | In `.scratch/` | In messages | Advisory findings |
|---|---|---|---|---|
| burn-mamba | 656 | 46 | 24 | ~1,100 |
| burn-stack | 189 | 54 | 11 | ~380 |

Most hard findings are long sentences, semicolons (burn-mamba only), long
paragraphs, phrasal verbs and synonym rotation.

The prose step of `scripts/check.sh` fails when a changed file gets worse than
on `origin/main`. But the step skips `.scratch/` and the message strings. So
nothing guards the tickets or the messages.

## Solution

- Each repo sweeps its prose in batches. A batch fixes each hard finding of
  its files. It reads the paragraph around each finding, and it judges each
  advisory finding.
- Each hard finding that stays has a reason in its ticket.
- In both repos, the prose step of `scripts/check.sh` also reads `.scratch/`
  and the message strings. It keeps its rule: no changed file gets worse than
  on `origin/main`.

## User Stories

1. As a reader agent, I want each sentence to have one structure, so that I
   parse it the way the author meant.
2. As a reader agent, I want at most 25 words in each sentence of a
   description, so that I keep its subject.
3. As a reader agent, I want no semicolons, so that each sentence holds one
   idea.
4. As a reader agent, I want one topic and at most six sentences in each
   paragraph, so that its first sentence gives the topic.
5. As a reader agent, I want one plain verb instead of a phrasal verb, so that
   no meaning depends on an idiom.
6. As a reader agent, I want one word for one action in each file, so that two
   words never seem to name two actions.
7. As a reader agent, I want each doc to describe the code as it is, so that I
   do not act on an old state.
8. As a crate user, I want strict STE in each panic, assert and expect message,
   so that I can act on a failure.
9. As an implementer, I want each ticket body in the same style, so that I
   read the ticket the way its author meant.
10. As an implementer, I want one method for all batches, so that each batch
    applies the same rules.
11. As an implementer, I want a read budget for each batch, so that a batch
    fits in one context.
12. As an implementer, I want a closed list of reasons to keep a finding, so
    that I never rewrite correct prose for the linter.
13. As an implementer, I want a list of the text that I must not touch, so
    that no build, link or anchor breaks.
14. As an implementer, I want to search the tests of both repos before a
    message changes, so that no matching test breaks.
15. As a maintainer, I want each hedge to keep its strength, so that no rewrite
    changes a claim.
16. As a maintainer, I want each fact, number, condition and scope qualifier
    kept, so that the sweep changes only the form.
17. As a maintainer, I want a report when a comment and the code disagree, so
    that I decide which side is the intent.
18. As a maintainer, I want a list of the false positives of the linter, so
    that I can fix the linter in the skill.
19. As a maintainer, I want the records of each ticket to keep their facts, so
    that the history of a decision stays readable.
20. As a maintainer, I want the resolved tickets in the sweep, so that all text
    in the repos has one style.
21. As a maintainer, I want `scripts/check.sh` to read `.scratch/`, so that a
    new ticket adds no violation.
22. As a maintainer, I want `scripts/check.sh` to read the message strings, so
    that each new message follows strict STE.
23. As a maintainer, I want the change of `scripts/check.sh` to land first, so
    that it guards each batch of the sweep.
24. As a maintainer, I want all copies of a verbatim paragraph to change
    together, so that the copies stay equal.
25. As a maintainer, I want the terms and identifiers kept, so that the sweep
    does not start a rename.
26. As a maintainer, I want the line width of each file kept, so that the diff
    shows only the prose change.
27. As a reviewer, I want each ticket to change one repo, so that
    `/mattpocock-skills:code-review` reads one diff.
28. As a reviewer, I want the count of hard findings before and after each
    batch, so that I see the result of the batch.

## Implementation Decisions

### Terms

- **Hard finding**: a finding that the linter counts toward its exit code. The
  prose step counts these.
- **Advisory finding**: a finding that never fails the linter: passive voice,
  compound tense, em dash and technical phrasal verb.
- **Kept finding**: a hard finding that stays, with a reason from the list in
  [Kept findings](#kept-findings).

### Scope

- Every tracked `.rs` and `.md` file of both repos, outside each `tmp/`
  directory at any depth.
- This includes all of `.scratch/` (resolved tickets too), the `CLAUDE.md`
  files, the READMEs, `docs/`, `info/`, and the notes and handoff files of the
  examples.
- In a `.rs` file, the linter reads the doc comments, the comments and the
  strings of `panic!`, `expect` and `assert!` calls. This is the option
  `--parts docs,comments,messages`.

### Modes

- Comments, rustdoc, Markdown and tickets are STE-flavored. A description has
  at most 25 words in each sentence.
- The strings of `panic!`, `expect` and `assert!` are Strict. A sentence has at
  most 20 words.

### The method of a batch

1. Run the linter of `scripts/check.sh` with `--enable history --parts
   docs,comments,messages` on the files of the batch. These are its tracked
   `.rs` and `.md` files, outside each `tmp/` directory. Record the count of
   hard findings in the ticket.
2. Rewrite each sentence or paragraph that has a hard finding.
3. Read the full paragraph of each finding for the lexical rules:
   - synonym rotation, nominalization, hedge stacking and marketing
     adjectives,
   - the terms of the two glossaries, and their _Avoid_ entries. ADR-0008
     permits a source word in quotes or backticks.
4. Judge each advisory finding in the files of the batch:
   - Change a passive to active voice when you can name a relevant actor.
   - Split an em dash that joins two sentences into two sentences.
   - Change a compound tense to a simple tense, unless it gives current
     relevance or a hedge.
   - Change a technical phrasal verb (`set up`, `back up`, `shut down`) to one
     verb when one verb has the same meaning.
   - An advisory finding that stays needs no note.
5. Run the linter again. The batch ends with zero hard findings, except the
   kept findings that the ticket lists.

### Text that does not change

- Code, identifiers, code spans, link targets and heading text. Headings are
  link anchors, and the title of a ticket is a heading.
- Each fact, number, condition and scope qualifier, and the strength of each
  hedge. The rewrite adds no fact.
- The line width of each file.
- The terms and the identifiers. When a finding shows a problem with a term,
  record it under `## Comments`. A rename follows the rename rule of
  `CLAUDE.md`, in a different change.
- In a ticket: the state of each checkbox, and each fact of `## Comments` and
  `## Answer`.

### Kept findings

A hard finding stays only for one of these reasons. The ticket lists each kept
finding under `## Comments`, with its location, rule, match and reason.

- **False positive**: the linter misreads the text. An example is a phrasal
  verb that the linter reads across a gap of spaces.
- **Quoted word**: the text cites a source word or a banned phrase, in quotes
  or backticks (ADR-0008). An example is a rule that quotes a banned phrase of
  the history rule.
- **Ticket record**: a history phrase in `## Comments` or `## Answer` of a
  ticket, when a rewrite loses a fact of the record.
- **Precision**: the rewrite loses a necessary precision. This is the "Kept
  as-is" rule of the skill.

The linter is part of the skill, outside the access allowlist. The sweep does
not change it. Do not rewrite correct prose only to silence a false positive.

### Messages and copies

- Before a message changes, search the tests of both repos for each part of
  the message. Examples are `should_panic(expected = …)`, a `contains(…)`
  check and a string compare. Keep each substring that a test matches.
- A burn-mamba test can match a burn-stack message after the next pin move.
  So a burn-stack ticket also searches the tests of burn-mamba.
- Change all verbatim copies of a message or a paragraph in the same ticket.
  When a copy is in the other repo, record it under `## Comments` of the
  ticket of that repo.

### A doc that disagrees with the code

Change neither the doc nor the code. Record the evidence under `## Comments`:
the code line, a test or the output of a small run. Ask the user at the end of
the ticket.

### The prose step of `scripts/check.sh`

In both repos:

- The step also reads the files under `.scratch/`.
- The step reads each `.rs` file with `--parts docs,comments,messages`, on
  both sides of the compare.
- The rule stays. A changed file fails when it has more hard findings than at
  the merge base with `origin/main`. A new file starts at zero.
- The comments of the script and each doc that names the prose step describe
  what the step reads.

So a new ticket starts at zero hard findings. Its `## Comments` follow the
rules too, and the history rule applies.

### Batches

- Each ticket changes one repo (`docs/agents/issue-tracker.md`). A burn-stack
  ticket goes to `../burn-stack/.scratch/ste-sweep/issues/`.
- The read budget is about 2–2.5k lines for each ticket. The estimate counts
  10 lines on each side of each finding.

| Repo | Area | Estimated lines | Proposed tickets |
|---|---|---|---|
| burn-mamba | `src/mamba3/` | 6,700 | 3, cut by subdirectory |
| burn-mamba | `examples/reset/` | 2,800 | 1 |
| burn-mamba | the other examples | 2,300 | 1 |
| burn-mamba | the other `src/` modules | 1,600 | 1 |
| burn-mamba | `info/`, `docs/`, `benches/` and the root files | 2,100 | 1 |
| burn-mamba | `.scratch/` | 2,500 | 1 |
| burn-stack | `src/modules/` | 1,700 | 1 |
| burn-stack | `src/examples/` | 1,600 | 1 |
| burn-stack | `src/utils/`, `src/optim/`, `src/reference/` and the root of `src/` | 1,500 | 1 |
| burn-stack | `.scratch/`, `docs/` and the root files | 2,500 | 1 |

### Order

- The `scripts/check.sh` ticket of each repo goes first, with no blocker.
- The `scripts/check.sh` ticket of a repo blocks each sweep ticket of that
  repo. The sweep tickets do not block each other.
- The `.scratch/` ticket of each repo goes last in its repo, so that it also
  reads the comments that the earlier batches wrote.

### Commits

- Each ticket has one commit in its repo.
- The sweep does not move the burn-stack pin. A prose change in burn-stack
  does not change what burn-mamba builds.

## Testing Decisions

- A good check tests the result for the reader: the count of the linter and
  the build. It does not test the words that the implementer selected.
- The sweep adds no test. Prose has no seam for `/mattpocock-skills:tdd`.
- The sweep has no coverage gate and no bench gate, because it changes no code
  path.

The seams:

1. **The prose step of `scripts/check.sh`.** No changed file gets more hard
   findings than at the merge base. The `scripts/check.sh` tickets plant no
   violation to prove their change.
2. **The linter on the files of each batch.** The batch ends with zero hard
   findings, except its kept findings. The ticket records the count before and
   after.
3. **The build of the changed repo, on `flex`.** `cargo check`,
   `cargo test --lib`, `cargo test --examples` and `cargo doc --all --no-deps`
   pass with no warnings. They find a changed message that a test matches, and
   a broken doc link.

Prior art: tickets 02–05 of `../burn-stack/.scratch/glossary-followups/`. Each
one lists the hits that stay, with their reasons, under `## Comments`.

## Out of Scope

- A change of the linter or the skill. The user fixes the false positives that
  the tickets list.
- The comments of `.sh`, `.py` and `.toml` files. The linter and the prose step
  read only `.rs` and `.md` files.
- Each `tmp/` directory, at any depth.
- `bench.md` and `kernels.md`. `bench.sh` and `kernels.sh` write these reports,
  so a hand edit is lost at the next run.
- Error strings that the linter does not read, for example `format!` errors
  and CLI help.
- A lexical review of a paragraph that has no finding.
- A rename of a term or an identifier. The sweep rewrites the prose of a
  glossary, but its terms, their meanings and its _Avoid_ entries stay.
- A rule of zero hard findings in `scripts/check.sh`. The linter cannot accept
  a kept finding.
- A move of the burn-stack pin.

## Further Notes

- The counts come from burn-mamba `9a8958ec`, burn-stack `e3a6c7b` and the
  skill at `88d216a`. The measure files are in `tmp/ste-sweep/`.
- The history rule also flags a passive in the past tense, for example "were
  added".
- Two known false positives: `CLAUDE.md:65` quotes a banned phrase of the
  history rule. At `src/mamba3/positive/tests.rs:825`, the linter reads
  "keeps … out" across a gap of spaces.
