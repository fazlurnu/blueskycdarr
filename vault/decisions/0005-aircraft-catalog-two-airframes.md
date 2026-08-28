# ADR 0005 — Aircraft performance is a two-entry catalog; policy here, mechanism in the fork

- Status: accepted
- Date: 2026-08-26
- Deciders: Fazlur Rahman

## Context

"Aircraft performance/type" is the categorical design variable: MixedVarLSENew sweeps
`kinematics in {multirotor, fixedwing}`, mirroring OpenCDaRR's registry (`M600` +
`multirotor`, `SMALL_FIXEDWING` + `fixedwing`). On the BlueSky side the ingredients are
scattered: the fork hard-codes the M600's turn-rate limits (15 deg/s, 10 deg/s²) inside
`Traffic.create`, speed envelopes live in OpenAP's `rotor/aircraft.json`, fixed-wing turn
authority comes from the autopilot's bank default — and BlueSky ships no small fixed-wing
UAV performance entry at all (its fixed-wing models are airliners, whose stall floors
around 60 m/s would fight a 15 m/s cruise inside the integrator).

## Decision

One frozen `AircraftModel` per airframe (`blueskycdarr/aircraft.py`): the BlueSky
carrier type, the resolver speed window `[v_min, v_max]`, and the turn authority. After
creation the engine *writes the catalog into the fork's per-aircraft arrays* — catalog as
policy, fork as mechanism — so airframe behaviour is data in this repo, not an edit to
BlueSky:

- **`multirotor`** — carrier `M600`, window [0, 18] m/s, `max_tr` 15 deg/s and
  `max_dtr2` 10 deg/s²: exactly the numbers CDaRR ran, now set explicitly rather than
  relying on the fork's `M600` string match.
- **`fixedwing`** — OpenCDaRR's `SMALL_FIXEDWING` numbers (stall 12, top 25 m/s, bank
  44 deg from Reyner & Liem, Drones 2026). Limiter off (`inf`), which selects BlueSky's
  own bank-angle turn path — the coordinated turn `g tan(phi) / V_TAS` *is* the
  fixed-wing model — with `ap.bankdef` set to 44 deg. The stall floor binds where it
  matters: MVP may not command below `v_min`, so a fixed-wing cannot resolve by stopping.
- **The carrier trick.** The fixed-wing flies under the OpenAP *rotor* id `Amzn`, the
  one shipped envelope (|v| <= 44 m/s, no positive stall floor) wide enough that
  BlueSky's integrator never interferes with the 12–25 m/s window. The fixed-wing
  *character* — turn dynamics and speed floor — comes entirely from this catalog's
  policy, not from the carrier's performance file.

## Alternatives rejected

- **Add a small fixed-wing entry to the fork's OpenAP data.** The honest long-term fix,
  but it moves the change surface into the engine dependency and stalls this package on
  a fork release. Deferred, noted as the successor if the fork grows such an entry.
- **Fly the fixed-wing as an OpenAP airliner.** The integrator enforces airliner stall
  speeds; a 15 m/s cruise is unflyable. Rejected on physics.
- **Turn-rate-limit the fixed-wing** (e.g. `max_tr = g tan(44 deg)/15 ~ 36 deg/s`).
  Loses the speed dependence that distinguishes a banked turn from a yaw-rate turn —
  the very contrast the categorical variable exists to expose. Rejected.
- **Free-form per-run performance numbers in config.** Two named airframes are the
  study; a performance editor is unrequested generality. Rejected.

## Consequences

**Good:** a new airframe is a new value plus tests; the categorical axis is
label-addressable from YAML, Python and the MixedVarLSENew space alike.
**Cost:** the `Amzn` carrier id is a misnomer readable in BlueSky logs; the catalog
comment and this ADR carry the explanation.
**Obligation:** catalog numbers cite their source in place (fork values, OpenCDaRR's
`performance.py`, the paper); a number without provenance does not land.

## Relations

- [[0001-bluesky-fork-is-the-engine]] — the limiter arrays this policy writes into.
- OpenCDaRR ADR 0012/0013 (their vault) — the multirotor/fixed-wing split this catalog
  mirrors, re-derived there, engine-delegated here.
