# Zotero and Google Docs workflow

The LaTeX manuscript (`main.tex` plus `references.bib`) is the submission source of truth. Use the citation-ready DOCX as a collaboration copy for Jeff, then import it into Google Docs and replace its plain-text author–year citations with active Zotero citations.

## One-time setup

1. Install and open the Zotero desktop application.
2. Install the Zotero Connector for the browser used with Google Docs.
3. In Zotero, choose **File → Import**, select `paper/ml4h2026/references.bib`, and import it into a collection named `Temporal EHR ML4H 2026`.
4. Resolve any duplicates in that new collection before inserting citations.
5. Open the citation-ready DOCX in Google Drive with **Open with → Google Docs**. Keep the original DOCX and LaTeX files unchanged.

## Citation insertion

Use **Zotero → Add/Edit Citation** in Google Docs. Select an author–year style accepted for collaborative review; the conference LaTeX will continue to use its own natbib style. Insert active Zotero fields at these locations:

| Manuscript location | Zotero items |
|---|---|
| Introduction: long context, repetition, irregular intervals | Wornow et al., *Context Clues* |
| Introduction: EHR-specific learned temporal components | Pang et al., *CEHR-BERT* |
| Introduction: task-dependent tokenization | Liu et al., *Temporal Tokenization Strategies*; Al Attrach et al., *Rethinking Tokenization* |
| Introduction: closest general-purpose embedding predecessor | Hegselmann et al., *Large Language Models Are Powerful EHR Encoders* |
| Related work: benchmark | Wornow et al., *EHRSHOT* |
| Related work: data standard | McDermott et al., *MEDS* |
| Related work: embedding family | Zhang et al., *Qwen3 Embedding* |
| Related work: one-pass abstraction boundary | Sarthi et al., *RAPTOR* |
| Related work: explicit temporal pooling | Yoon et al., *PaReGTA* |
| Related work: causal timing boundary | Wald et al., *Time After Time* |
| M3 methods: recurrent architecture | Cho et al., original GRU paper |
| M3 methods: optimizer | Loshchilov and Hutter, AdamW |
| Evaluation: family-wise correction | Holm (1979) |
| Evaluation: false-discovery correction | Benjamini and Hochberg (1995) |

After the citations are active, place the cursor under **References** and choose **Zotero → Add/Edit Bibliography**. Use **Zotero → Refresh** after edits.

## Collaboration rules

- Jeff and every coauthor editing citations should have Zotero Connector installed. Editing or dragging citation text without Zotero can break active fields.
- Do not use **Unlink Citations** in the working document. When a clean final copy is needed, first make a duplicate, unlink only the duplicate, and retain the linked master.
- Do not manually change the spelling, year, or author list in an active citation. Correct the Zotero library record and refresh.
- Treat the Google Doc as a review surface. Apply accepted scientific edits back to `paper/ml4h2026/main.tex`; update `references.bib` when a citation is added or corrected.
- Before ML4H submission, compile in the official 2026 template and verify every citation, bibliography entry, DOI/URL, anonymity requirement, and page limit.

## Items that Zotero cannot resolve for the authors

Before submission, the authors must supply the final author list, the applicable IRB/exemption/NHSR language, and the truthful code-availability statement. These are explicit placeholders and must not be inferred or generated from citation metadata.
