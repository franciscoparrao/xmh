# XMH v1.5.0 — A check for the silent failure mode

Adds `validate_instrumentation()`, the check for the way this framework fails quietly. No
reported number changes; what changes is that a defect class which has twice produced a wrong
published figure can now be detected before trusting an attribution.

Supersedes v1.4.0 (10.5281/zenodo.23219527) as the archive of record.

## The failure mode

If an operator is registered but the evolution loop never applies it through its registered
handle, or no neutral exists for its type, then neutralizing it changes nothing, every
coalition containing it equals the coalition without it, and its Shapley value is exactly
zero. That zero is indistinguishable from a legitimate null player.

It is not hypothetical. It happened twice here:

- DE's greedy acceptance was inlined in the evolution loop rather than routed through the
  registered operator, which produced the 50/50/0 split reported before v1.1.0.
- PSO registered `velocity_update` and `position_update` while the neutral factory knew only
  `velocity` and `position`, so the factory returned nothing and the resulting NaN propagated
  into a coalition that never improved.

## Why the existing check does not catch it

The method already evaluates the empty coalition rather than assuming v(0) = 0. That check is
necessary but not sufficient: in the empty coalition every operator is neutralized, the trial
individual equals its target, nothing is accepted, and the improvement is zero whether or not
each operator was correctly routed.

We verified this by injecting the defect. With DE's greedy acceptance left unrouted,
v(0) = 0 still holds and selection's Shapley value comes out at exactly +0.0000.

## The check that does

`xmh/core/validation.py`. `validate_instrumentation()` runs each operator's neutralization
against the full coalition over a shared set of seeds and reports any operator whose
neutralization leaves every run identical. It passes for every operator of the six algorithms
instrumented here; re-injecting the DE defect makes it fail and name `selection_greedy`.

A failure is not proof of a bug — a genuine null player behaves identically — but it means
that operator's attribution is not interpretable until its registration has been inspected by
hand, which is exactly the distinction the zero hides.

An earlier version of this check counted invocations of the neutral operator. It was dropped:
the count depends on which internal path a given algorithm uses to apply its operators, and it
reported zero for DE, GA and PSO despite all three being correctly instrumented. Behavioural
difference is the property the method actually depends on and holds for any implementation.

## An API trap, documented

`disable_operator()` on its own does not survive `run()`, which re-initializes the operator
registry. Coalition evaluation patches the initializer so the neutralization is reapplied —
which is why the published results are correct — but a user calling the public method directly
and then running the algorithm silently gets the full algorithm back. This is now stated in
the README, in the module docstring, and in the manuscript.

## Tests

Two regression tests: the six instrumented algorithms must pass validation, and the injected
DE defect must be caught. Suite is now 45 passed, 1 skipped.
