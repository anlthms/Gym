# ARC transform refinement agent

This agent implements the verifier-guided ARC protocol described in
`docs/design-docs/arc-agi-multiturn.md` in the parent NeMo-RL repository.

It keeps one persistent proposer history and creates a fresh executor history
for every rule revision. Training targets and hidden test outputs stay in the
ARC resources-server session. Only the final proposer generation is returned as
the trainable response; every proposer/executor call remains in the audit trace.

The proposer and executor references may point to the same model server. They
are separate configuration fields so executor competence can be benchmarked or
frozen independently.
