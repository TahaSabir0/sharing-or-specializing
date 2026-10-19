---
name: UGRIP-transcript
description: Summarize an auto-transcribed UGRIP/FLARE meeting transcript into the standard SUMMARY.md block format and append it to transcripts/SUMMARY.md. Use when the user points at a new transcript .txt file or asks to summarize/log a meeting.
---

# UGRIP-transcript

Turn a raw, auto-transcribed meeting transcript into a tight, agent-readable summary block and
append it to `transcripts/SUMMARY.md`. The raw `.txt` stays untouched as ground-truth backup.

## When to use
- The user drops a new file into `transcripts/` and asks to summarize/log it.
- The user says "summarize the meeting" / "add this to the summaries" / invokes `/UGRIP-transcript`.

## Inputs
- A transcript file path (the user will usually name it). If they don't, list `transcripts/*.txt`,
  compare against the headings already in `transcripts/SUMMARY.md`, and summarize the one(s) not yet
  covered — confirm with the user if ambiguous.

## Context you need (read first if not already in context)
- The team & roles from `AGENTS.md` "Who's who" — so you can attribute action items to real names:
  Taha Sabir, Alberto Montero Molina, Thinh Quang Ly ("Alex"), Veronika Dmitrenko; mentors Dr. Mai
  Shaaban, Ahmed Aly, Dr. Tausifa Jan Saleem, Prof. Mohammad Yaqub.
- The existing `transcripts/SUMMARY.md` to match the house style and avoid duplicating a meeting.

## Caveats about the transcripts
- **Speaker labels are unreliable** — treat "UNKNOWN" as mentor-or-intern; infer the speaker from
  content, don't trust the label. Don't invent attributions you can't support.
- Transcription garbles names (e.g. "Dr. May/Maie" = Dr. Mai; "Veronica" = Veronika). Normalize to
  the correct names above.

## How to summarize
For a large transcript, delegate the read+summarize to a subagent (Explore/general-purpose) so it
doesn't bloat the main context — pass it the section spec below and have it return ONLY the markdown
block. For several transcripts at once, spawn one subagent each in parallel.

Produce a block with this exact shape (omit any section that's genuinely empty):

```
### <YYYY-MM-DD> — <meeting type: kickoff / brainstorm / technical KT / Q&A-setup / check-in / ...>

**Decisions:** committed choices.
**Mentor guidance / technical:** specific recommendations — methods, tools, gotchas.
**Action items:** who does what (real names; "All" if everyone).
**Open questions / blockers:** unresolved items.
```

For a non-project session (e.g. cohort orientation) adapt headings to fit (logistics / advice /
action items / blockers) — match how the existing blocks in `SUMMARY.md` already handle it.

Rules: bullet points, factual, no filler. Convert relative dates ("tomorrow") to absolute where the
transcript date makes it unambiguous. Keep each block well under one screen.

## Output
Append the new block to the **bottom** of `transcripts/SUMMARY.md` (chronological, newest last).
Don't touch the raw `.txt`. Then tell the user one line: which meeting was added and any notable
new decision/blocker worth surfacing (e.g. something that should also update the AGENTS.md snapshot,
a new action item for Taha, or a fact worth saving to memory).
