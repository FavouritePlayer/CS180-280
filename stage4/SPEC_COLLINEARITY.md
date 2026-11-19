# Project spec: collinearity across occluders (method candidate #2)

Status: **planning only — no code written yet.** This is a spec to implement
against, not a protocol. The actual `PROTOCOL_COLLINEARITY.md` (hyperparameters
and declared direction of effect, frozen before any real-data run) gets written
once the synthetic tests pass, matching the house convention from
`stage3/PROTOCOL_VP.md`.

## 1. Idea

A straight 3D edge — a baseboard, a counter edge, a door frame, a shelf line, a
window mullion, crown molding — projects to a straight 2D line under a pinhole
camera. If a foreground object (a chair, a person, a plant, clutter) occludes
the middle of that edge, the two visible pieces on either side of the occluder
must still lie on **exactly the same 2D line**, because projection doesn't bend
straight lines; occlusion just hides part of one. This is a weaker, purely
projective version of the same physical fact Sarkar et al. (CVPR 2024,
"Shadows Don't Lie and Lines Can't Bend!", arXiv:2311.17138) build a
line-segment classifier around — we're taking the "lines can't bend" half of
that claim and turning it into a cheap, local, calibration-free geometric
test rather than a learned classifier.

Hypothesis to test (declare the sign in the protocol, don't flip it after
seeing results, same as `PROTOCOL_VP.md`): real photos' occluded-line segment
pairs resume more precisely collinear (lower residual) than generated images'
do, because generators don't explicitly track hidden 3D geometry behind the
objects they paint in.

**Key property that makes this different from both existing cues:** it needs
no camera intrinsics, no depth, no normals. It's pure 2D image-plane geometry
on top of the line detector we already have. It can't share a failure mode
with Method 1 (depth-normal) or the VP work (Manhattan-world orthogonality via
GeoCalib's `K`) because it doesn't use either estimator — the only thing it
reuses is the LSD line-detection front end.

## 2. Pipeline, step by step

### Step 1 — Line detection (reuse, no new code)
Run `detect_lines` from `stage3/vp_geometry.py` unchanged (OpenCV LSD,
`cv2.LSD_REFINE_STD`, segments under 2% of the image diagonal dropped — same
settings already validated in stage 3). Output: a list of line segments, each
`((x1,y1), (x2,y2))`.

### Step 2 — Candidate occluded-pair search
For every pair of segments `(seg_i, seg_j)` in the image (or a filtered
subset — see Performance note below), test whether they are plausibly two
visible pieces of one line interrupted by an occluder:

1. **Direction agreement.** Compute each segment's unit direction vector.
   Angular difference between them must be below `angle_thresh_deg`
   (candidate: 3°, needs a sweep on synthetic data — see Section 4).
2. **Perpendicular offset agreement.** Fit the infinite line through
   `seg_i` (reuse `segments_to_lines`'s homogeneous-line construction).
   The perpendicular distance from `seg_j`'s midpoint to that line must be
   below `perp_thresh_frac * image_diagonal` (candidate: 0.5–1%).
3. **Non-overlap / ordering.** Project all four endpoints onto the shared
   direction axis. `seg_i` and `seg_j` must not overlap on that axis — one
   must sit entirely before the other. This is what distinguishes "two
   pieces split by an occluder" from "the LSD detector fragmented one
   continuous line into two abutting segments," which is a detector
   artifact, not a case we want to score.
4. **Gap bounds.** The gap between the near endpoints (along the shared
   axis) must be larger than `min_gap_frac * image_diagonal` (excludes
   detector-fragmentation noise, which produces tiny near-zero gaps) and
   smaller than `max_gap_frac * image_diagonal` (excludes coincidentally
   aligned, unrelated lines on opposite sides of the image — e.g. two
   different walls that happen to align in 2D by chance).

Any pair passing all four tests is a **candidate occluded-line pair**.

**v1 scope decision:** do not require that the gap region visually contain an
occluding object (no segmentation/object-detection dependency). Purely
geometric pairs that satisfy all four tests above count, whether or not there
is a real object sitting in the gap. This keeps the method classical and
$0-compute. Flag as a **v2 extension** (not in scope for the 3-methods-needed
goal): check gap-region content against the two line directions' context
(e.g. a Sobel-edge-density or color-histogram discontinuity in the gap) to
filter out pairs that are "coincidentally collinear but nothing occludes them"
— likely a small fraction given bound 4 above, probably not worth the added
complexity for a first pass.

### Step 3 — Per-pair residual
For each candidate pair, compute two residuals (keep them separate, don't
collapse to one number yet):
- `angle_residual_deg` — the angular difference from Step 2.1 (already
  computed as part of the candidate test, so this is free).
- `offset_residual_frac` — the perpendicular distance from Step 2.2,
  normalized by image diagonal.

### Step 4 — Per-image aggregation
Most images will have more than one candidate pair (a room has several
straight edges, several of which may be partially occluded). Aggregate:
- Primary score: **mean of the two normalized residuals, averaged over all
  candidate pairs** in the image (match the `score_image` pattern in
  `vp_geometry.py` — return a dict, not just a scalar).
- Also report **median** and **best-single-pair** (minimum residual) as
  secondary aggregates in the output, since a single very clean occlusion
  case might be more diagnostic than averaging over many noisy ones — decide
  which aggregate is primary in the protocol, before running on real data,
  not after.
- Report `num_candidate_pairs`, `num_lines` per image, same as `vp_geometry`
  reports `num_vps`/`num_lines` — needed for the same kind of confound check
  stage 3 needed (raw line/pair *count* might trivially separate classes via
  a sharpness artifact, independent of the geometric residual itself).

### Step 5 — Degenerate cases
Some images will have zero candidate pairs (no occlusion events over a
straight edge happened to be framed, or too few lines detected at all).
Report this explicitly as `"insufficient structure"` in the per-image JSONL
record — same convention as `vp_geometry.score_image` and the one COCO-matched
image that had zero VPs. Do not impute or silently drop. If this fraction
turns out to be large (see Section 5, coverage check), that's an important
finding to report, not a bug to hide.

## 3. Files (mirrors `stage3/` structure, new `stage4/` directory)

- `stage4/collinearity_geometry.py` — core pure functions, no I/O:
  - `find_collinear_pairs(segments, angle_thresh_deg, perp_thresh_frac, min_gap_frac, max_gap_frac) -> list[pair]`
  - `pair_residual(seg_i, seg_j) -> {"angle_residual_deg":..., "offset_residual_frac":...}`
  - `score_image(gray, angle_thresh_deg, perp_thresh_frac, min_gap_frac, max_gap_frac, min_length_frac=0.02) -> dict`
    returning `{"mean_score":..., "median_score":..., "best_score":..., "num_pairs":..., "num_lines":...}` or
    `{"status": "insufficient_structure", "num_lines":...}`.
  - Reuse `detect_lines` and the line-fitting helper from `stage3/vp_geometry.py`
    directly (`from stage3.vp_geometry import detect_lines, segments_to_lines`)
    rather than copy-pasting — keeps the two modules' line detection in sync if
    either is ever retuned.
- `stage4/test_collinearity_geometry.py` — synthetic unit tests, same spirit
  as `stage3/test_vp_geometry.py` (render actual pixels, run the real LSD
  detector, don't just test the math on hand-built segment lists):
  1. Two perfectly collinear rendered segments with a mid-sized gap → found as
     a candidate pair, residual ≈ 0.
  2. Two segments at a known angular offset (e.g. 10°) → found as a
     candidate only if above `angle_thresh_deg`'s test boundary; residual
     matches the known offset within detector noise.
  3. Two segments offset by a known perpendicular distance (parallel, not
     angled) → residual matches the known offset; confirms offset and angle
     residuals are measuring different things independently.
  4. Two clearly non-collinear segments (perpendicular, or far apart) → not
     returned as a candidate pair at all.
  5. Two abutting/overlapping segments on the same line (simulating detector
     fragmentation of one continuous line) → excluded by the non-overlap
     test, confirming it doesn't get double-counted as an "occluded" pair.
  6. Gap larger than `max_gap_frac` → excluded (two aligned-but-unrelated
     lines on opposite sides of a synthetic scene).
  7. End-to-end: render an actual line occluded by a drawn rectangle
     (`cv2.line` + `cv2.rectangle` painted over the middle), run
     `score_image` on the rendered pixels, confirm it recovers low residual
     and `num_pairs >= 1`.
- `stage4/run_collinearity.py` — batch runner. Unlike `stage3/run_vp.py`,
  **no `--calibration` argument is needed** — this method never touches `K`,
  depth, or normals, so it can run directly off raw images + a manifest, with
  no dependency on any other stage's cached output. Same resumable-JSONL
  append pattern as `stage2`/`stage3` for consistency (so the existing
  analysis scripts' conventions still fit), same `settings.json` dump of
  fixed hyperparameters.
- `stage4/requirements.txt` — subset of `stage3/requirements.txt`: numpy,
  opencv-contrib-python-headless, Pillow, pytest. (No scipy needed unless the
  aggregation ends up wanting it — note in the file if it's cut.)
- `stage4/PROTOCOL_COLLINEARITY.md` — to be written once tests pass, before
  any real-data run: final hyperparameter values, the declared aggregate
  (mean vs. median vs. best-pair) and declared direction of effect. Never
  edited after that point, same rule as `PROTOCOL_VP.md`.
- `stage4/README.md` — written after the pilot runs, same structure as
  `stage3/README.md` (idea, files, usage, status-with-results).

## 4. Hyperparameters to pin down before the protocol freeze

All of these need a quick sweep on synthetic test cases (Section 3's tests)
before picking final values — don't guess and declare blind:
- `angle_thresh_deg` — candidate: 3°. Too loose and unrelated lines get
  paired; too tight and real noisy detections (JPEG blur, slight lens
  distortion) get excluded as false negatives.
- `perp_thresh_frac` — candidate: 0.5–1% of image diagonal.
- `min_gap_frac` — candidate: 1–2% of image diagonal (excludes detector
  fragmentation of one continuous edge).
- `max_gap_frac` — candidate: 25–30% of image diagonal (excludes
  coincidental unrelated-line alignment across a whole scene). This one
  matters most for precision — needs the most scrutiny in the sweep.
- `min_length_frac` for the underlying LSD call — reuse stage 3's 2% value
  unless the coverage check (Section 5) suggests otherwise.
- Primary aggregate: mean vs. median vs. best-pair across an image's
  candidate pairs — pick one as primary in the protocol; report all three
  regardless.

## 5. Before committing to a full pilot: a coverage check

This method's coverage is **not guaranteed the way the VP method's was** —
every indoor photo has Manhattan-world structure somewhere, but not every
indoor photo happens to frame a straight edge that's actually occluded by
something in the foreground. Run the scorer (even with placeholder
hyperparameters) over 20–30 images from the existing Frontier or COCO-matched
pilot sample first and check:
- What fraction of images produce at least one candidate pair?
- What's the distribution of `num_candidate_pairs` per image?
- Spot-check a handful of the detected pairs visually (draw them on the
  image) to sanity-check that the geometric filter is actually catching real
  occlusion events and not noise.

If coverage turns out low (say, under 50% of images produce even one
candidate pair), that's a real risk to flag to the team before investing
further — it would mean this cue, unlike Method 1 and the VP work, can't
score a large fraction of the dataset at all, which may be reason enough to
reconsider against single-view-metrology or specular-highlight instead. This
check should happen early and cheaply (no real experiment needed yet, just
look at the numbers) before writing the protocol.

## 6. Analysis plan (once a protocol is frozen and a real run happens)

Match existing conventions exactly, reusing code rather than rewriting it:
- `stage2/detection_metrics.py`'s `auroc` / `bootstrap_auc` for the Frontier
  pilot (unmatched 50 real + 50 generated).
- `stage2/matched_metrics.py`'s `matched_summary` for the COCO-matched pilot
  (caption-group-preserving bootstrap, per-generator breakdown, paired
  contrast) — same as `stage3/analyze_coco_matched_vp.py` did for VP.
- **Confound check, mandatory, learned from stage 3:** report AUROC for raw
  `num_candidate_pairs` (and raw `num_lines`) alone, alongside the geometric
  residual's AUROC. Stage 3 found line-count alone beat the geometric score
  on the Frontier pilot (0.637 vs. 0.528) — don't skip this check here just
  because it didn't reproduce on the COCO-matched sample; it must be
  re-checked on each new sample this method is run on, not assumed resolved.
- Explicit missing-score handling for any image scored "insufficient
  structure," reported the same way the one COCO-matched VP image was
  (exclude its caption group from the matched analysis, keep it in pooled
  AUROC, name it explicitly in the report) — but given Section 5's concern,
  this fraction could be much larger here than it was for VP, so the report
  needs a clear statement of what fraction of the sample was actually usable.

## 7. Compute cost and timeline

- **Compute: $0.** Classical CV only, CPU-only, no GPU rental, no API calls,
  no new pretrained model. The only dependency beyond what's already in the
  repo is OpenCV, already in `stage3/requirements.txt`.
- **Rough timeline** (part-time student work, same order of effort as stage
  3's VP work):
  - ~1 day: `collinearity_geometry.py` + synthetic tests passing.
  - ~0.5 day: coverage check (Section 5) on existing pilot images.
  - ~0.5 day: `PROTOCOL_COLLINEARITY.md` frozen, `run_collinearity.py` written.
  - ~1 day: Frontier 100-image pilot run + `analyze_*` script + report.
  - ~1 day: COCO-matched 300-image pilot run + report, comparison against
    Method 1 and the VP work on the same sample (same table style as
    `results/stage3_coco_matched_vp/REPORT.md`'s "agreement" section).
  - Total: roughly 4 days of focused work, spread over about a week part-time.

## 8. Open questions for the team before starting

- Which aggregate (mean/median/best-pair) should be primary — needs a
  decision before the protocol freeze, not after seeing results.
- Is the v1 geometric-only candidate-pair filter (no occluder/object check)
  acceptable, or does the team want the v2 object-aware filter from Step 2
  in scope for this round too? (Recommendation: ship v1 first, revisit v2
  only if coverage or precision turns out to be a problem.)
- If the Section 5 coverage check comes back low, do we still proceed with
  this method, or pivot one of the 3 open slots to single-view-metrology or
  specular-highlight instead? Decide the threshold for "low" now rather than
  after seeing a disappointing number.
