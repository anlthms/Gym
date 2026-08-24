# ARC transform refinement

This environment combines the `arc_agi_2` resource server with the
`arc_transform_refinement_agent`. The resource server owns training and hidden
test targets. The agent keeps one proposer history, applies each proposed rule
in fresh executor chats, and returns only proposer tokens for policy loss.

Prepare ARC-AGI data with `prepare.py`, then start the environment with the
`arc_transform_refinement_agent` entry in `config.yaml`.
