# Probe Bench: agent instructions

## Working rules

- Defend technical positions with evidence; do not flip-flop to placate. When recommending, weigh trade-offs and failure modes first.
- Pin every dependency to an exact version. MIT-compatible licenses only.
- Run only the tests relevant to a change; type-check the frontend with `npx tsc --noEmit`.
- One named implementation per behavior; delete code a change makes unused.
- Comments describe the present in present tense, never the history of a change.
- No em dashes anywhere, including user-facing copy. Braces on every control-flow body.
- Generic examples only: invented names and organizations, `user@example.com`, no real people.
- One commit per logical change.

## Project rules

- `backend/contract.py` and `frontend/src/types.ts` are the wire contract and must change together.
- Do not read files outside this repository except public documentation and the model card.
- Secrets live only in the git-ignored `.env` and on the serving host; never paste a key into a transcript, a commit, or a log.
- Every generated example is read by a human before it becomes a prebuilt example.
- Releases ship the committed tree (`scripts/ship.sh`), never the working tree.
