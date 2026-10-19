# Gate 4 — MSDL Learning Population-Ownership Validation
Date: 2026-10-18

## Purpose
Continue from the existing V10.2/Gate 1–3 codebase and current Codex conversation. Gate 1–3 are frozen.

Gate 4 has one purpose: verify that the learning path itself can use the same coarse-grained population-ownership principle already validated for forward/scaffold computation:

**persistent local ownership of state/weights → complete local computation → communication only at real dependency / commit boundaries.**

Do not run the final FULL yet.

## Non-negotiable scientific constraints
Do not change:
- P3-D / MSDL equations
- WQ, WK, WV, WO, W1, W2, WR learning rules
- reciprocal BR, BO, B2 rules
- E/M dynamics
- `N_SUB = 80`
- `FORCE = 27`
- learning rates or width normalization
- sparse masks
- fixed-point rounding / saturation semantics
- RNG order
- update / commit semantics
- canonical seeds
- final FULL configuration
- Gate 1–3 implementations or results

Retain the fixed ZERO_D bounds-safe implementation.

# Phase 1 — Build the real learning dependency DAG from source
Before writing the parallel implementation, inspect the current source and produce a dependency table for one learning event.

Include at minimum: WQ, WK, WV, WO, W1, W2, WR, BR, BO, B2.

For each group identify:
1. event-local inputs / drive values read;
2. E state read/write;
3. M state read/write;
4. weight state read/write;
5. accumulator state read/write;
6. RNG consumption, if any;
7. dependency on another group's new value from the same event;
8. weight/accumulator visibility before event completion;
9. exact commit boundary;
10. whether it may share a parallel wave with another group without changing serial semantics.

Do not assume all groups are independent.

Output:
- `GATE4_DEPENDENCY_DAG.md`
- `GATE4_DEPENDENCY_DAG.json`

The JSON must contain explicit parallel waves and reasons.

# Phase 2 — Measure the matched serial learning baseline
Instrument the existing serial learning kernel with Timer2 before choosing ownership.

Measure at least:
- total learning-stage cycles
- WQ/WK/WV/WO/W1/W2/WR cycles
- BR/BO/B2 cycles
- commit cycles
- event-boundary overhead

Use measured serial time to balance work across cores; do not balance by group count alone.

Output `GATE4_SERIAL_PROFILE.json`.

# Phase 3 — Implement coarse-grained resident learning ownership
Each worker core is resident from program start and permanently owns its assigned learning populations, including as applicable:
- E
- M
- weights
- accumulators
- scratch
- local counters

A worker receives the event-local inputs/drives required for its entire assigned wave, then independently performs the whole local learning block, including all `N_SUB=80` dynamics for its owned populations.

Forbidden:
- substep → sync
- neuron → sync
- small matrix op → sync
- group fragment → sync

Required pattern:

```text
wave begins
    |
    +-- Core A: complete all owned group dynamics/local update preparation
    +-- Core B: complete all owned group dynamics/local update preparation
    +-- Core C: complete all owned group dynamics/local update preparation
    |
wave barrier only if required by actual dependency
    |
deterministic commit / next dependency wave
```

There must be no synchronization inside an independent group's E/M evolution.

Communication is allowed only at true drive dependency boundaries, true wave-completion boundaries, commit / weight-visibility boundaries, and event completion where required.

## Core-count policy
Start with the minimum useful number of worker cores implied by the dependency DAG and measured workload. Prefer 2–4 owner cores unless the measured DAG clearly justifies another count. Do not create mostly-idle cores merely to claim parallelism.

Report both latency and allocated core-time.

# Phase 4 — Matched serial vs parallel correctness
The same binary must contain:
1. a matched single-core serial reference using the same current learning kernels and initial state;
2. the coarse-grained multi-core ownership implementation.

Both must consume identical initial weights, E/M states, accumulators, event descriptors, RNG stream/state, and drive values.

After each validation checkpoint compare for every affected learning group:
- E w/z state
- M w/z state
- decoded E output
- decoded M output
- weights
- accumulators
- sparse off-mask values
- spike/event counters
- commit counters
- RNG state
- reciprocal state
- any resident state learning may modify

Do not accept only final NLL/checksum agreement.

A mismatch must report first event index, group, field, element index, serial value, and parallel value.

# Commit-boundary coverage
Gate 4 must explicitly exercise the real commit path.

First inspect the source and determine the exact mapping between `FORCE=27`, event count, microsteps, and commit timing. Then construct workloads that demonstrably include:
1. immediately before the first commit;
2. the event/microstep crossing the first commit;
3. immediately after the first commit;
4. at least two additional commits.

Do not assume event 27 is necessarily the commit event unless the source confirms it. Record resolved commit indices in the result JSON.

# Bootstrap + continual coverage
Gate 4 must cover both:

## A. Bootstrap learning
Use a representative scaffold/readout learning path that exercises the actual bootstrap MSDL machinery.

## B. Continual learning
Use the real continual-learning event path, including reciprocal updates where applicable.

For both:
- matched serial and coarse parallel must be bit-exact;
- overflow = 0;
- protocol errors = 0;
- commit coverage must be explicit.

# Performance workloads
After correctness is established, measure at least:
- 64 learning events
- 256 learning events
- 1024 learning events

If commit semantics require another size to guarantee multiple commits, add it.

Each size:
- 5 independent board runs
- report all runs
- report median
- report dispersion
- use Timer2

Do not use a single run as the performance conclusion.

# Required timing / utilization counters
Report:
- matched serial total
- coarse parallel total
- speedup
- each owner core active cycles
- each owner core wait cycles
- barrier wait cycles
- dispatch count
- join/barrier count
- packets sent / received
- per-wave cycle totals
- per-group cycle totals if available
- commit cycles
- allocated application cores
- allocated core-time
- useful-active core-time

Keep wall-clock latency and core-time separate.

# Gate 4 acceptance
## Scientific / correctness PASS
Mandatory:
- dependency DAG documented from source
- matched serial and parallel use the same scientific kernel
- bootstrap learning parity PASS
- continual learning parity PASS
- pre-/cross-/post-commit states PASS
- multiple commits PASS
- all relevant E/M states bit-exact
- all weights and accumulators bit-exact
- RNG state bit-exact
- sparse-mask invariants PASS
- overflow = 0
- protocol errors = 0

## Performance status
Report separately:
- `PERFORMANCE_IMPROVED = true/false`
- speedup median for each workload
- core-time ratio

Correctness PASS does not require speedup. If scientifically exact but slower, retain the result and diagnose the measured bottleneck. Do not weaken correctness gates or alter MSDL to create acceleration.

# Required result files
Produce:
- `GATE4_DEPENDENCY_DAG.md`
- `GATE4_DEPENDENCY_DAG.json`
- `GATE4_SERIAL_PROFILE.json`
- `GATE4_RESULT.json`
- `REPORT.md`
- complete source diff / patch
- build/APLX hashes
- raw Timer2 results for all repetitions

Top-level `GATE4_RESULT.json` should include at least:

```json
{
  "PASS": true,
  "SCIENTIFIC_PARITY": true,
  "BOOTSTRAP_PASS": true,
  "CONTINUAL_PASS": true,
  "COMMIT_BOUNDARY_PASS": true,
  "OVERFLOW_COUNT": 0,
  "PROTOCOL_ERRORS": 0,
  "PERFORMANCE_IMPROVED": true,
  "serial_median_s": 0.0,
  "parallel_median_s": 0.0,
  "speedup": 0.0,
  "allocated_cores": 0,
  "core_time_ratio": 0.0
}
```

Extend with detailed per-group/per-wave and per-run measurements.

# Stop condition
After Gate 4 results are produced: **STOP. Do not run the final FULL automatically.**

Return:
1. dependency DAG;
2. source changes;
3. correctness result;
4. commit-boundary evidence;
5. Timer2 measurements;
6. speedup / core-time;
7. remaining bottleneck.

Only after Gate 4 is reviewed and accepted will the next step be the 5-run final integrated FULL.

# Frozen context
Use the existing `STAGE_SUMMARY_20261018.md` as the authoritative stage boundary.

Current accepted facts:
- Gate 1: matched coarse per-head QKV speedup ≈ 1.628×.
- Gate 2: complete attention-head/resident kernel speedup ≈ 1.999×.
- Gate 3: real 3000-context scaffold-bank pipeline speedup ≈ 2.023× including descriptor preparation.
- Gate 1–3 are frozen.
- No new coarse-learning FULL has been run.
- Final integrated FULL comes only after Gate 4 passes.
