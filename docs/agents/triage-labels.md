# Triage Labels

The skills use five canonical triage roles, plus one closed state. This file maps each role and the closed state to the label string of this repo. The tracker is local markdown, so a label is the value of the `Status:` line near the top of an issue file.

| Label in mattpocock/skills | Label in our tracker | Meaning                                       |
| -------------------------- | -------------------- | --------------------------------------------- |
| `needs-triage`             | `needs-triage`       | The maintainer must evaluate this issue       |
| `needs-info`               | `needs-info`         | It waits for more information from the reporter |
| `ready-for-agent`          | `ready-for-agent`    | It is fully specified, ready for an AFK agent |
| `ready-for-human`          | `ready-for-human`    | A human must implement it                     |
| `wontfix`                  | `wontfix`            | No one will do it                             |
| `resolved` (wayfinder)     | `resolved`           | It is complete, or its question has an answer |

When a skill names a role (for example, "apply the AFK-ready triage label"), use the string of that role from this table.

To use a different vocabulary, edit the "Label in our tracker" column.
