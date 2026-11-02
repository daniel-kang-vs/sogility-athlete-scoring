# Athlete Scoring — Sogility

> How can a training score explain both peer standing and progress without penalizing younger athletes?

**Industry practicum · public methodology**

A dual scoring framework separates cohort-relative standing from fixed performance benchmarks, with distinct parent and trainer views.

[Case study](https://daniel-kang-vs.github.io/projects/sogility.html) · [Source evidence](https://github.com/daniel-kang-vs/sogility-athlete-scoring/blob/main/README.md)

## Highlights

- **30–99 display scale**
- **Two complementary scores**
- **Five skill groups**

## The problem

A global drill score compares young players with older athletes and obscures what to train next. Small cohorts and noisy repeat attempts make a raw rank unstable.

## Data & scope

The analysis used an internal session extract. The public repository contains an anonymized methodology writeup, not player-level data, executable scoring code, or a hosted dashboard.

## How it works

1. Normalize drill direction and skew before comparing results.
2. Fit relative scoring to age-and-gender cohorts and blend toward the wider population when a cohort is small.
3. Weight the three most recent attempts 0.5, 0.3 and 0.2.
4. Compute performance bands separately from reference quantile anchors, then rescale for display.

## Key technical decision

Keep relative standing separate from the performance band. A player’s band should not change simply because different peers enter the cohort; benchmark anchors need versioning to preserve that behavior.

## Results & interpretation

The framework explains one athlete through a peer-relative score and a separate performance band. Parent views emphasize progress and five skills; trainer views emphasize entry, comparison, and history. The repository documents the design and handoff.

## Limitations & next steps

No match-outcome labels were available to validate game-performance prediction. The proposed production handoff ports deterministic scoring to Elixir and checks it against Python fixtures; the public repository does not establish that this deployment occurred.

## Evidence & reproducibility

Public evidence is the anonymized README methodology; confidential player records and internal interfaces are excluded.

## Repository guide

- [`README.md`](README.md) — methodology and handoff.

## Run or inspect

Use the source links above. Confidential data and internal application code are not distributed; there is no public dashboard to run.

## Technology

Python · SQL · React

---

## Technical appendix

## Scoring

```
Session extract (SQL / warehouse)
        │
        ▼
Clean: drop or flag outliers, flip lower-is-better drills, log skewed drills
        │
        ├── RPS (the number on the screen, the SGI score)
        │     cohort median and cohort span
        │     same sigmoid as before
        │     blend toward the global distribution when n is small
        │     last 3 attempts weighted 0.5, 0.3, 0.2
        │
        └── APS (the band)
              piecewise map through the 10th, 25th, 50th, 75th, 90th
              if the cohort is thin: both genders in the age group, then the wider pool
        │
        ▼
Linear rescale onto 30–99
        │
        ▼
Band from APS: Foundation, Developing, Approaching, Strong, Elite
```

### Relative Performance Score

RPS answers “how does this player compare with similar players?”

The score is percentile-like, but a score of 75 does not establish that the athlete outperformed exactly 75% of peers. It is not an empirical rank. The raw attempt is standardized with that cohort’s median and span, then passed through the same sigmoid. When the cohort is too small to trust, cohort and global parameters are blended by sample size, so the score moves gradually instead of snapping to a fallback table.

Recency, for the three most recent attempts, oldest to newest in the weight list below:

| Attempt | Weight |
|---------|--------|
| Most recent | 0.50 |
| One back | 0.30 |
| Two back | 0.20 |

Fewer than three attempts: the same weights, renormalized.

### Absolute Performance Score

APS answers “how strong is this result, apart from who else trained this month?”

The raw result is mapped by piecewise interpolation through the 10th, 25th, 50th, 75th, and 90th percentiles of the reference distribution. Those anchors are the benchmark. If the cohort does not have enough mass, the benchmark widens in order: both genders in the age group, then the broader pool. Direction is fixed first, so a faster dash and a higher pass count both point up. Skewed drills are logged before that map.

### The 30–99 display scale

Both scores are linearly rescaled onto 30–99 after the statistical step. The product rule was explicit: scores under 30 read as a foundation level and were kept off the usual display so a young player is not handed a 12, and 99 leaves the best player room to move. Bands are cut on the APS value, not on RPS, so the label does not change just because a new player joined the cohort.

| Band | APS on the 30–99 scale |
|------|------------------------|
| Foundation | below 30 |
| Developing | 31–49 |
| Approaching | 50–69 |
| Strong | 70–89 |
| Elite | 90 and above |

What the screen shows: **RPS is the score. APS is the band.**

---

## What was tried and left out of the parent view

Players are specialized. A single overall number hides that. K-means on the five skill scores produced four readable profiles: playmaker (passing and vision), explosive athlete (agility), attacker (dribbling), and control (first touch and passing).

Those labels did not ship on the parent view. A type encourages a kid to specialize, which is the opposite of the training goal. The clustering writeup stayed with the company for trainer-side use.

---

## Product

Two interfaces on the same player record.

| View | Job |
|------|-----|
| Parent and trainee | One score, five skill bars, a band, and progress over time |
| Trainer | Fast score entry during a session, cohort comparison, assessment history |

The layout follows a FIFA-style card: the number first, then the skills, then the trend. The prototype is React and TypeScript. Scoring was developed and checked in Python. Production at Sogility is Elixir, so the handoff was to port the deterministic math into Elixir and validate it against the Python fixtures, not to stand up a separate Python service. Session rows in production are read through the existing database. The CSV was the analysis extract.

Python stayed the place for the statistical work: distribution checks, the sigmoid versus quantile comparison, the cohort-size blend, and the archetype exploration.

---

## Key Technical Decisions

1. **Keep the sigmoid. Change who it is fit to.** Parents already had a percentile-like number. Replacing the curve would have thrown away a model the company could explain. The bias was the global reference group.

2. **Do not let the band come from the rank.** RPS is supposed to move when the cohort moves. If the Elite label moved with it, a player could change tier without changing their drill results. Bands come from APS.

3. **Shrink small groups instead of dropping them.** A hard minimum sample size leaves new sites with no score. A sample-size blend keeps a score and tells the truth about how local it is.

4. **Weight the last three attempts.** Repeated measures on the same drill were too noisy to treat the latest row as ability.

5. **No player-type label on the parent screen.** The cluster structure is real. Publishing it fights the product.

---

## What this does not prove

There is no match outcome, coach grade, or competitive result in the extract. The scores are a measurement system. They are not a validated predictor of game performance. A later version can test them against those outcomes. Until then, the claim is interpretability and stability, not predictive accuracy.

The player file is not in this repository.

---

## Tech Stack

| Layer | Role |
|-------|------|
| SQL | Extract session rows and feed the scoring pipeline |
| Python | Distributions, cohort sigmoid, quantile APS, clustering experiments |
| TypeScript / React | Parent and trainer prototype |
| Elixir | Recommended production port of the same deterministic scoring |

