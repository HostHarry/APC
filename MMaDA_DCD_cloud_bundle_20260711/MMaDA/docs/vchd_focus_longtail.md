# Persistent focus trajectory decoder (focus-dwell + focus-longtail)

## Motivation

The previous adaptive-temporal decoder used an exponential moving average
(EMA) over CD-APC distributions. That design did not degenerate cleanly when
visual grounding was unreliable: it used a different eligible set from the
purely visual anchor, kept an infinite geometric tail, and gave the initial
distribution an artificial weight. It also had to invent a separate fallback
whenever the previous stability gate returned an empty set.

The replacement drops both EMA and any explicit "top-\(V\) intersection
across frames" language. Instead it exposes a single, purely local eligibility
primitive per MASK position, the **focus-dwell counter**, and marginalizes
across time with a **shifted log-logistic survival kernel** whose strength is
gated by exposure, visual relevance, and current-vs-trajectory conflict.
When the kernel is silent (zero mix ceiling or unreliable visual channel),
the decoder is bit-exact identical to the focus-dwell baseline.

## Focus frame and dwell counter

Each committed iteration snapshots the top-\(V\) unresolved MASK positions
ranked by the current CD-APC confidence into a `FocusFrame`
(the *high-confidence focus region*). The focus-dwell profile keeps the last
`focus_dwell_depth` frames; the focus-long-tail profile keeps
`focus_longtail_history_upper_bound(kernel_scale, kernel_shape,
kernel_offset, focus_longtail_history_epsilon, focus_dwell_depth)` frames,
i.e., the smallest depth such that every retained log-logistic weight is at
least `focus_longtail_history_epsilon` (default \(10^{-4}\)). Older frames
are pruned because their kernel weight is numerically negligible; the ring
buffer therefore never grows with iteration count.

The eligibility primitive is defined **per position**. For iteration \(t\)
and position \(i\), the *focus-dwell counter* \(D_{i,t}\) is the length of the
longest contiguous suffix of ring-buffer frames whose focus region contains
\(i\), plus one for the current iteration.

The counter is the *only* thing the decoder uses to decide eligibility, and
the gate uses **warmup semantics**: position \(i\) is eligible when
\(D_{i,t}\ge 1 + M_t\), where \(M_t\) is the number of frames currently
stored in the dwell window (bounded by `focus_dwell_depth`). While the
buffer is filling (\(M_t < L\)) fewer consecutive appearances suffice, and
once \(M_t=L\) the gate reaches its steady-state value \(L+1\). This
preserves the invariant "candidate stays inside the model's high-confidence
focus region across every observed iteration" without ever forming,
enumerating, or storing an explicit set intersection.

## Long-tail marginalization

Let \(q_{i,t-\ell}\) be position \(i\)'s full-vocabulary CD-APC distribution
at lag \(\ell\). The shifted discrete log-logistic survival kernel is

\[
S(\ell;\lambda,\kappa,\delta)
=
\frac{1}{1+\left(\frac{\ell+\delta}{\lambda}\right)^\kappa}.
\]

Kernel activation is position-specific:

\[
z_{i,t} = z_{\max}
\left(1-e^{-E_{i,t}/\tau_E}\right)
\left(1-e^{-R_{i,t}/\tau_R}\right)
\left(1-e^{-D_{i,t}/\tau_C}\right),
\]

where \(z_{\max}\) is `focus_longtail_mix_ceiling`, \(E\) is new-context
exposure, \(R\) is visual relevance, and

\[
C_{i,t} =
\operatorname{TV}\left(
q_{i,t},
\frac{1}{L+1}\sum_{\ell=0}^{L}q_{i,t-\ell}
\right)
\]

is the current-vs-dwell trajectory conflict. The implementation clamps
\(z_{i,t}\) to \([0,1]\).

The unnormalized lag weight is

\[
\widetilde w_{i,t,\ell}
=
(1-z_{i,t})\,\mathbf 1[\ell\le L]
+z_{i,t}S(\ell;\lambda,\kappa,\delta),
\]

and the posterior is

\[
h_{i,t}
=
\frac{
\sum_{\ell=0}^{A_{i,t}}
\widetilde w_{i,t,\ell}q_{i,t-\ell}
}{
\sum_{\ell=0}^{A_{i,t}}\widetilde w_{i,t,\ell}
}.
\]

\(A_{i,t}\) is the length of the position's contiguous available trajectory.
The default \((\lambda,\kappa,\delta)=(3.2,8,1)\) keeps the first three
observations dominant, sharply reduces the fourth, and retains a strictly
positive power-law tail so old evidence never vanishes.

## Exact focus-dwell fallback

If \(z_{i,t}=0\), then

\[
\widetilde w_{i,t,\ell}=\mathbf 1[\ell\le L],
\qquad
h_{i,t}=\frac{1}{L+1}\sum_{\ell=0}^{L}q_{i,t-\ell}.
\]

This is the focus-dwell equal-weight marginalization over the
\((L+1)\)-frame window (three rounds for \(L=2\)). The decoder also calls
the same selector as the focus-dwell profile, so eligibility, dual gates,
empty-dwell fallback, and liveness fallback remain identical. Setting
`focus_longtail_mix_ceiling=0` therefore yields token-by-token identity to
the focus-dwell decoder; visual failure (\(R_{i,t}\to0\)) drives
\(z_{i,t}\to0\) automatically, so the fallback also triggers whenever the
visual channel is unreliable.

## Default profile

Use `--vchd-profile focus_longtail`. Relevant options are:

- `--vchd-focus-dwell-depth 2`
- `--vchd-focus-capacity 64`
- `--vchd-focus-longtail-kernel-scale 3.2`
- `--vchd-focus-longtail-kernel-shape 8`
- `--vchd-focus-longtail-kernel-offset 1`
- `--vchd-focus-longtail-mix-ceiling 1`
- `--vchd-focus-longtail-exposure-tau 0.1`
- `--vchd-focus-longtail-relevance-tau 0.01`
- `--vchd-focus-longtail-conflict-tau 0.002`
- `--vchd-focus-longtail-history-epsilon 1e-4`

The decoding report records eligibility, activation, long-tail mass, current
weight, effective dwell depth, token replacements, and empty-dwell fallback
counts under the `focus_longtail_*` / `focus_dwell_*` keys.
