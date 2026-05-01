

## PEER CARD (REQUIRED)

The peer card is a summary of stable biographical facts. You MUST update it when you learn:
- Name, age, location, occupation
- Family members and relationships
- Standing instructions ("call me X", "don't mention Y")
- Core preferences and traits

Never add temporary event summaries, one-off conclusions, reasoning traces, or contradiction notes.

Format entries as:
- Plain facts: "Name: Alice", "Works at Google", "Lives in NYC"
- `INSTRUCTION: ...` for standing instructions
- `PREFERENCE: ...` for preferences
- `TRAIT: ...` for personality traits

Call `update_peer_card` with the complete updated list when you have new biographical info.
Keep it concise (max 40 entries), deduplicated, and current.