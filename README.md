# SemanticDisputeRouter

**Precedent-Aware AI Arbitration on GenLayer**

A staking dispute-resolution contract where an on-chain LLM judge rules on
disagreements between two parties — and, unlike a stateless judge, gets
*more consistent over time* by consulting semantically similar prior rulings
before deciding a new case.

---

## What it does

Two parties can lock a GEN stake against a dispute (e.g. a freelance
delivery disagreement, a contract breach claim, any binary/three-way
disagreement). When both sides have staked and the dispute is ready for
judgment, the contract:

1. Embeds the dispute's description into a vector and searches a running
   `VecDB` of previously resolved disputes for the most semantically similar
   past cases.
2. Feeds those precedent cases — their descriptions, rulings, and reasoning —
   into the LLM's arbitration prompt as context.
3. Asks the LLM to rule `PARTY_A`, `PARTY_B`, or `SPLIT`, with GenLayer's
   validator consensus (`gl.eq_principle.prompt_comparative`) ensuring
   multiple independent LLM runs agree on the categorical outcome before
   it's accepted on-chain.
4. **Explicitly validates the judge's output before any state change**:
   the raw response must parse as JSON, be a JSON object, contain both a
   `ruling` and `reasoning` field of the correct type, and `ruling` must be
   exactly one of `PARTY_A`, `PARTY_B`, or `SPLIT`. Any malformed, missing,
   wrongly typed, or out-of-domain output causes the transaction to fail
   safely (revert) rather than silently falling through to a default payout.
5. Pays out the staked pool according to the ruling, and records the new
   case (with its ruling and reasoning) as a precedent for future disputes.

The result is a growing, on-chain "case law" bank: as more disputes get
resolved, new similar disputes are ruled on *consistently* with how similar
cases were decided before — rather than each ruling being judged completely
independently with no memory of prior decisions.

## Why this is different

Most AI-arbitrated escrow contracts treat every dispute as an isolated
judgment call — the LLM sees only the current case and rules from scratch
every time, with no way to stay consistent with its own past decisions.
SemanticDisputeRouter treats every resolved dispute as a reusable precedent,
giving the arbitration layer memory and consistency the way a real legal
system builds on prior rulings rather than reinventing judgment each time.

## Architecture

| Concern | Mechanism |
|---|---|
| Dispute state | Parallel `TreeMap[str, ...]` fields per dispute ID (description, parties, status, ruling, reasoning, stakes) |
| Precedent memory | `VecDB[float32, 384, Precedent]` — vector search over prior resolved cases |
| Embeddings | `SentenceTransformer("all-MiniLM-L6-v2")`, called via `get_embedding_generator()(text)` |
| AI judgment | `gl.nondet.exec_prompt` inside a closure, reconciled across validators via `gl.eq_principle.prompt_comparative` |
| Output safety | Judge output is validated (JSON structure, field presence, types, and domain of `ruling`) before any state mutation, fund transfer, or precedent write is allowed to proceed |
| Fund safety | Stakes only move on `RESOLVED` (to winner(s)) or `CANCELLED` (refund to party_a, only before party_b accepts) |

Dispute records are stored as parallel primitive `TreeMap`s (one map per
field) rather than a single `TreeMap` of a `@allow_storage` dataclass. This
was a deliberate workaround: constructing a fresh `@allow_storage` dataclass
via keyword arguments and assigning it directly into a `TreeMap` produced a
reproducible GenVM storage-serialization error in Studio. The same dataclass
pattern works fine as a `VecDB` value type (used here for `Precedent`), so
`VecDB` values use dataclasses while `TreeMap` values are kept as
primitives.

## Contract methods

### Write

- **`open_dispute(description: str, party_b: Address) -> str`** *(payable)*
  Party A opens a dispute against Party B, staking GEN. Returns the new
  `dispute_id` (e.g. `"D0"`).
- **`accept_dispute(dispute_id: str)`** *(payable)*
  Party B accepts by staking the exact same amount Party A staked. Dispute
  moves to `ACCEPTED`.
- **`resolve_dispute(dispute_id: str) -> str`**
  Triggers the AI arbitration flow: fetches similar precedents, prompts the
  LLM for a `PARTY_A` / `PARTY_B` / `SPLIT` ruling with reasoning, reaches
  validator consensus, validates the parsed output's structure and domain,
  pays out the stake pool accordingly, and records the case as a new
  precedent. Returns `"<RULING> - <reasoning>"`. Reverts safely if the
  judge's output fails validation.
- **`cancel_dispute(dispute_id: str)`**
  Party A can cancel and reclaim their stake, but only while the dispute is
  still `OPEN` (i.e. before Party B has accepted).

### View

- **`find_similar_precedents(description: str, top_n: int) -> list[dict]`**
  Preview which past rulings would be considered as precedent for a given
  dispute description, before actually resolving anything.
- **`get_dispute(dispute_id: str) -> dict`**
  Full current state of a dispute (parties, stakes, status, ruling, reasoning).
- **`get_dispute_count() -> int`**
- **`get_precedent_count() -> int`**

## Tested end-to-end in GenLayer Studio

- `open_dispute` → `accept_dispute` → `resolve_dispute` — full lifecycle
  confirmed with real GEN stakes and real validator consensus (5-validator
  set, multiple LLM policies agreeing on the categorical ruling).
- First dispute resolved with **no precedent** available (first-principles
  judgment path) — contract correctly paid out per the LLM's ruling and
  recorded the case as `P0`.
- `find_similar_precedents` confirmed to correctly surface prior rulings by
  semantic similarity before a second, related dispute was resolved.
- `cancel_dispute` confirmed to correctly refund Party A when invoked before
  Party B accepts.
- **Output-validation fix retested**: after adding explicit JSON/type/domain
  validation to `resolve_dispute` (in response to reviewer feedback about
  the SPLIT branch being an unsafe silent catch-all for malformed judge
  output), the full lifecycle was re-run end-to-end and confirmed to behave
  identically for valid judge output — the fix closes the unsafe fallback
  path without changing normal-case behavior.
- All test transactions reached `FINALIZED` / `SUCCESS` with supermajority
  validator agreement.

## Deployment

```
# v0.2.16
# { "Seq": [
#     { "Depends": "py-lib-genlayer-embeddings:09h0i209wrzh4xzq86f79c60x0ifs7xcjwl53ysrnw06i54ddxyi" },
#     { "Depends": "py-genlayer:1jb45aa8ynh2a9c9xn3b7qqh8sm5q93hwfp7jqmwsfhh8jpz09h6" }
# ] }
```

Constructor takes no arguments. Deploy directly in GenLayer Studio or via
`genlayer-js` against Studionet / Bradbury testnet.

## Notes on GenVM quirks discovered while building this

- An `Address`-typed method **parameter** arrives inside the contract as a
  raw integer, not a proper `Address` object (unlike `gl.message.sender_address`,
  which *is* already a proper `Address`). To store it safely:
  `Address(param.to_bytes(20, byteorder="big"))`.
- `gle.SentenceTransformer("all-MiniLM-L6-v2")` returns a **callable**, not
  an object with `.encode()`. Call it directly: `model(text)`, not
  `model.encode(text)`.
- Constructing a new `@allow_storage` dataclass instance via keyword
  arguments and assigning it directly into a `TreeMap` value slot can throw
  a GenVM storage serialization error; using `VecDB` as the dataclass's
  container works fine, and parallel primitive `TreeMap`s work as a
  reliable workaround when a `TreeMap` of structured records is needed.
- LLM-judge output must never be trusted implicitly: always explicitly
  validate structure, field presence, types, and the allowed domain of any
  categorical field *before* using it to gate state changes or fund
  transfers. An unvalidated `else`/default branch on judge output is a
  silent-failure risk, not a safe fallback.

## License

MIT
