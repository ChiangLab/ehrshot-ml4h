# Next action

Status date: 2026-09-01

Prompt 3 is closed by verified fit-recovery job 451. Preserve jobs 437, 444, 445_0 through 445_3,
failed job 446, and recovery job 451 unchanged. The active prompt is
`prompts/next/04_IMPLEMENT_M2_AND_M3.md`.

Implement and locally validate the Prompt 4 M2/M3 source using synthetic fixtures only. M2 must use
the authorized on-premise instruction endpoint, never an external API. M3 must retain the frozen
architecture and seeds. The DGX execution order is C1 reference cells first, the prespecified
validation-only approach-selection gate second, then only the fixed within-approach sensitivity
cells needed for the selected approach, followed by a cryptographically hashed frozen experiment
manifest. Do not access held-out test data or performance.
