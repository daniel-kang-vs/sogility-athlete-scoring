# Athlete Scoring — Sogility

![Python](https://img.shields.io/badge/Python-analysis-blue?logo=python) ![SQL](https://img.shields.io/badge/SQL-extract%20%26%20pipeline-336791?logo=postgresql) ![React](https://img.shields.io/badge/React-parent%20%26%20trainer%20UI-149eca?logo=react)

> **Decision:** stop showing parents a raw drill number. Sogility’s old sigmoid turned every drill into one global percentile, and that percentile hid whether a player was actually good. The redesign keeps a statistical score, splits it into a peer score and a stable benchmark, and puts both on a **30–99** scale that parents and trainers can read.

Industry practicum with Sogility, a nationwide soccer training company, for the Purdue MS in Business Analytics and Information Management. Faculty advisor: Shoaib Amjad Khan. Team: Meghna Ashok Advani, Ashwath Raju Srikanth, Daniel Kang, Deniz Nemil, Mansi Anil Patil. May 2026.

Daniel’s work on the engagement was the scoring redesign and the parent-facing and trainer-facing interface.

---

## Project Highlights

- The old model was one sigmoid fit to **every player on a drill**. It flattened developing players and stretched small gaps at the top
- The replacement is two scores. **RPS** is the number on the screen: a cohort sigmoid, age group by gender, blended toward the global distribution when the cell is small. **APS** places the performance band from quantile anchors
- Last three attempts are weighted **0.5 / 0.3 / 0.2**, so one noisy rep does not become the player
- Time-based drills are flipped before scoring. Skewed drills are log-transformed. Everything then lands on **30–99**
- Parent view answers “how is my child doing?” Trainer view is built for fast entry, comparison, and history

---

## Problem Statement

A parent was handed a number like **61** with no peer, no level, and no next step. The drills underneath that number are not even on the same scale.

| Drill type | Example | Direction |
|------------|---------|-----------|
| Count | Pass Find, Receive 10 | Higher is better |
| Time | Freelap dash, Knockout | Lower is better |

The legacy score was a sigmoid. For a raw result \(x\), with the skill’s global median as Mid and its global 99th percentile as Span:

```
SSpan = log(0.99 / 0.01) = 4.5951
Z     = 4.5951 × (x − Mid) / |Mid − Span|
RPS   = 100 / (1 + e^(−Z))
```

That curve is smooth, and it is the wrong comparison. A younger player is scored against older players. The bottom of the curve compresses real gaps, so two developing players look the same. The top of the curve does the opposite. The shape of each drill disappears. And a percentile still does not answer “is this good?”

There is also no match result or external outcome in the data, so the score cannot be validated as a predictor of game performance. It has to be a measurement system people can explain.

---

## Dataset

The practicum used about **12,600** historical session rows and about **1,600** newer rows, covering about **630** players. The export this snapshot comes from has **13,779** rows, **649** players, and **31** drills, from July 2022 through December 2024.

The file has no header. Names in the real extract are players, including youth athletes, so they are not in this repository. The rows below use placeholders. Scores and timestamps are from the file.

| Player | Gender | Age code | Category | Assessment | Score | Timestamp |
|--------|--------|----------|----------|------------|------:|-----------|
| Player A | male | F | Circuit Training | Knockout | 35 | 2022-10-03 23:02 |
| Player A | male | E | Tech Touch | Receive 10 Air | 10 | 2022-10-04 16:57 |
| Player B | male | E | Circuit Training | Pass Find 1 | 19 | 2022-10-04 17:13 |
| Player B | male | E | Circuit Training | Knockout | 7 | 2022-10-04 17:15 |

| Column | Meaning |
|--------|---------|
| Age code | A–F. A is the smallest group (245 rows in this export). C and D are about 4,000 each |
| Category + assessment | The drill. Examples: Circuit Training × Knockout, Freelap × 10yd Dash, Vertical Jump × Vertical |
| Score | Raw result. Units differ by drill |
| Timestamp | When the attempt was recorded |

Gender in this export: 7,590 male, 6,189 female.

Drills are then mapped into five skills: **passing, vision, dribbling, agility, first touch**.

About 1–1.4% of rows are outliers, mostly on machine-recorded drills (a score of 410 on a drill that does not live there). Those points move a percentile. The recommendation was to flag them at entry, when the trainer can confirm the number, instead of silently deleting them later.

---

## Why the model changed

Three statistical problems, not a request for a more complicated model.

**1. The reference group was everyone.** Relative standing only means something inside a similar age and gender. The sigmoid formula stayed. Mid and Span moved from the global drill distribution to the cohort distribution.

**2. A percentile is not a level.** RPS moves when the cohort moves. Parents also need a label that stays put if the player’s raw result stays put. That label is APS, built from fixed quantile anchors rather than from today’s rank.

**3. One attempt is too noisy.** The same player on the same drill jumps around. The displayed score uses the last three attempts, not the last row.

Small cells are the practical constraint. Some drills have thousands of rows. Others have a few dozen. A raw rank inside a cell of ten players jumps when one athlete is added. The score has to shrink toward a wider distribution instead of pretending that cell is a population.

---

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

A 75 means better than about 75% of the age-and-gender cohort on that skill, before the display rescale. It is not an empirical rank. The raw attempt is standardized with that cohort’s median and span, then passed through the same sigmoid. When the cohort is too small to trust, cohort and global parameters are blended by sample size, so the score moves gradually instead of snapping to a fallback table.

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
