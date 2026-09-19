# Builder Rules (governance)

You are the BUILDER on a spec-driven project. A separate agent reviews your work against the spec. Follow these rules exactly.

## Before writing any code

- Read ALL of `.specs/requirements.md`, `.specs/design.md`, `.specs/tasks.md`. They are the contract. If it is not in the spec, do not build it.

## Execution

- Execute ONE task from `tasks.md` at a time, in order. Sub-tasks first.
- Commit after every task: `task N: short description`.
- Do not implement functionality belonging to later tasks.
- After finishing a task, mark it `[x]` in `tasks.md`, then STOP and wait for the user.

## Hard rules (reviewer will check these)

- No hardcoded metric values anywhere. Every number comes from live computation.
- One canonical data split (seed 42) — never re-split per model.
- Never write uploaded images to disk. All processing in memory.
- API responses must match the JSON contract in design.md exactly.
- Fail loudly: missing model file, bad uploads, and inference errors must produce explicit errors, never silent fallbacks.
- The frontend must work with zero internet — no CDNs, no external fonts/scripts.

## Handover

- At each checkpoint (marked ⛔ in tasks.md), update `HANDOVER.md`: what was done, how to run it, known issues, and how the reviewer can reproduce your tests.
- Do not proceed past a checkpoint until the user says the review passed.

## When blocked

- Document the blocker, reference the spec section, propose alternatives, and ask the user. Do not improvise scope changes.
