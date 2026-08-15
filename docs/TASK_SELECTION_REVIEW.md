# EHRSHOT task selection for the first complete run

## Recommendation

Use a three-family panel for the initial experiment:

1. **Anemia** — laboratory result.
2. **New acute myocardial infarction** — incident diagnosis.
3. **Long length of stay** — operational outcome.

This makes the paper a methods study rather than a claim about one disease. It asks whether the
value of recurrence and correct time alignment generalizes across three substantially different
prediction settings. Run the same preregistered representation conditions for every task. Do not
replace tasks after viewing test performance.

The language in the paper should be **sources of label ambiguity, observation-process bias, and
transportability limits**, not “confounders,” because this study estimates predictive performance,
not a causal treatment effect.

## Laboratory options

| Task | Public-release labels / positive prevalence | Strength for this study | Main interpretation threat | Decision |
|---|---:|---|---|---|
| Anemia | 184,880 / 68.96% | Large, clinically intuitive, repeated hemoglobin history should make recurrence and interval spacing relevant | Testing frequency, transfusion, bleeding, dialysis, and cancer treatment affect both history and result | **Primary** |
| Thrombocytopenia | 179,618 / 33.25% | More balanced; platelet decline/recovery is a compelling temporal trajectory | Chemotherapy, infection, transfusion, and marrow disease create heterogeneous mechanisms | Best prespecified substitute or later sensitivity task |
| Hyponatremia | 212,837 / 28.52% | Large and moderately balanced | Fluids, medications, acute illness, and measurement intensity make timing hard to interpret | Secondary only |
| Hyperkalemia | 200,170 / 2.38% | Clinically consequential | Rare positives, treatment before measurement, kidney disease, and pseudohyperkalemia | Do not use in the first run |
| Hypoglycemia | 318,164 / 1.48% | Very large raw label count | Severe imbalance and meal/insulin timing not represented cleanly in structured data | Do not use in the first run |

EHRSHOT supplies four anemia severity levels but evaluates the benchmark as binary normal versus
abnormal. The code preserves the raw counts for validation and uses `integer_value != 0` for the
binary target.

## Incident-diagnosis options

| Task | Public-release labels / positive prevalence | Strength for this study | Main interpretation threat | Decision |
|---|---:|---|---|---|
| Acute MI | 6,837 / 6.79% | Clinically compelling and more objectively defined than most chronic coding outcomes | Diagnosis can still occur outside the system or reflect coding/documentation timing | **Primary** |
| Hypertension | 3,764 / 13.71% | More common outcome | Long preclinical period and delayed coding weaken “incident” interpretation | Secondary only |
| Hyperlipidemia | 4,442 / 12.74% | Adequate prevalence | Screening and documentation intensity dominate assignment | Secondary only |
| Pancreatic cancer | 7,011 / 3.77% | High clinical interest and a genuine longitudinal-risk question | Only 264 positive labels; referral and outside diagnosis threaten power/ascertainment | Future extension, not first run |
| Lupus | 7,038 / 2.23% | Complex longitudinal prodrome | Rare, diagnostic delay, phenotype heterogeneity | Do not use initially |
| Celiac disease | 7,129 / 1.32% | Long diagnostic pathway | Too few positives for the compressed schedule | Do not use initially |

## Operational options

| Task | Public-release labels / positive prevalence | Strength for this study | Main interpretation threat | Decision |
|---|---:|---|---|---|
| Long length of stay | 6,995 / 25.26% | Objective threshold, adequate positives, statistically stable, easy to explain | Discharge practice, social barriers, and hospital operations contribute to outcome | **Primary** |
| 30-day readmission | 7,003 / 13.01% | Strongly longitudinal and clinically familiar | Access, death, care coordination, and outside-hospital readmission affect label | Prespecified secondary if schedule permits |
| ICU transfer | 6,491 / 4.47% | Acute deterioration is clinically important | Low prevalence and institution-specific triage/capacity decisions | Do not use initially |

## The plain-English medical question

Can a frozen general-purpose text embedding make better predictions when it sees not only which
events occurred, but also how often they recurred and how much time separated them? We will test
that question for a next abnormal hemoglobin result, a first acute MI diagnosis within one year,
and a hospitalization lasting at least seven days.

These are not causal predictions. A positive result would show that elapsed-time information is
useful to the representation under controlled input perturbations; it would not show that changing
the timing of care changes the outcome.

Source for task definitions and published aggregate counts: the official
[EHRSHOT benchmark documentation](https://som-shahlab.github.io/ehrshot-website/docs/intro/benchmark/).
