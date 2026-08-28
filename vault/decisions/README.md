# Decisions

One numbered file per decision (`0001-...md`), sequential, no gaps. They are dated and
append-only: an ADR records what was decided *then*; when a decision is revisited, a new
ADR supersedes it and says so. The template and tone follow OpenCDaRR's
`vault/decisions/` — Context, Decision, Alternatives rejected, Consequences, Relations —
because this project is deliberately that project's BlueSky-engine sibling.

- [[0001-bluesky-fork-is-the-engine]] — the CDaRR BlueSky fork is a runtime dependency,
  wrapped at one boundary module.
- [[0002-event-based-broadcast-channel]] — jitter, latency, reception and surveillance
  range as one event-based channel; the two documented deviations from CDaRR.
- [[0003-declarative-experiments-opencdarr-style]] — Fixed/Sweep axes, closed vocabulary,
  YAML run files with a `sweep:` block, cards and CSV.
- [[0004-metric-seeding-and-crn]] — P(LoS) per encounter, raw-count uncertainty
  reporting (intervals removed by decision — see its update note), and the
  common-random-numbers seed tree.
- [[0005-aircraft-catalog-two-airframes]] — multirotor and fixed-wing as catalog values;
  the turn-limit policy lives here, the mechanism in the fork.
- [[0006-recovery-family-ftr-probftr]] — Past-CPA, FTR and Probabilistic FTR as a
  swappable component; the one-tick command lag the FTR criteria depend on, found by
  cross-validating against CDaRR itself.
- [[0007-blueskycdarr-parity-batch]] — VO, the noise-shape family, per-pair speed ranges,
  declared accuracy, and mixed pairs: every CDaRR experiment family as a declaration.
