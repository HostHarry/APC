# Exposure-gated log-logistic trajectory history

## Motivation

The previous adaptive-temporal decoder used an exponential moving average
(EMA) over CD-APC distributions. That design did not recover CCD when visual
adaptation was inactive: it used a different position set, an infinite EMA,
and different fallback behavior. It also gave the initialized distribution a
special weight.

The replacement keeps CCD's position-stability semantics and changes only the
temporal marginalization kernel. Visual information controls how much
long-tail history is used; it never adds another token-level visual score.

## Position eligibility

For iteration \(t\), let \(\mathcal{T}_t^V\) be the top-\(V\) unresolved MASK
positions ranked by the current CD-APC confidence. With the default
\(V=64\) and CCD history length \(L=2\), the eligible set is

\[
\mathcal{E}_t =
\mathcal{T}_t^V \cap \mathcal{T}_{t-1}^V \cap \mathcal{T}_{t-2}^V.
\]

Thus the mechanism retains CCD's per-MASK stability gate. Before two previous
snapshots exist, all available snapshots are used, exactly as in the existing
CCD implementation.

## Long-tail marginalization

For an eligible position \(i\), \(q_{i,t-\ell}\) is its full-vocabulary
CD-APC distribution at lag \(\ell\). The shifted discrete log-logistic
survival kernel is

\[
S(\ell;\lambda,\kappa,\delta)
=
\frac{1}{1+\left(\frac{\ell+\delta}{\lambda}\right)^\kappa}.
\]

The adaptive strength is position-specific:

\[
z_{i,t} = z_{\max}
\left(1-e^{-E_{i,t}/\tau_E}\right)
\left(1-e^{-R_{i,t}/\tau_R}\right)
\left(1-e^{-D_{i,t}/\tau_D}\right),
\]

where juxtaposition denotes multiplication. \(E\) is new-context
exposure, \(R\) is visual relevance, and

\[
D_{i,t} =
\operatorname{TV}\left(
q_{i,t},
\frac{1}{L+1}\sum_{\ell=0}^{L}q_{i,t-\ell}
\right)
\]

is current-versus-CCD trajectory conflict. The implementation clamps
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
positive power-law tail rather than truncating older observations.

## Exact CCD fallback

If \(z_{i,t}=0\), then

\[
\widetilde w_{i,t,\ell}=\mathbf 1[\ell\le L],
\qquad
h_{i,t}=\frac{1}{L+1}\sum_{\ell=0}^{L}q_{i,t-\ell}.
\]

This is exactly CCD's three-round equal-weight marginalization for \(L=2\).
The implementation also calls the same CCD selector, so position
intersection, dual gates, empty-intersection fallback, and liveness fallback
remain identical. Setting `adaptive_temporal_tail_mix_max=0` provides a
direct CCD-equivalence ablation.

## Default profile

Use `--vchd-profile adaptive_temporal`. Relevant options are:

- `--vchd-ccd-history-length 2`
- `--vchd-ccd-top-v-positions 64`
- `--vchd-adaptive-temporal-loglogistic-scale 3.2`
- `--vchd-adaptive-temporal-loglogistic-shape 8`
- `--vchd-adaptive-temporal-loglogistic-offset 1`
- `--vchd-adaptive-temporal-tail-mix-max 1`
- `--vchd-adaptive-temporal-exposure-scale 0.1`
- `--vchd-adaptive-temporal-relevance-scale 0.01`
- `--vchd-adaptive-temporal-conflict-scale 0.002`

The decoding report records eligibility, activation, long-tail mass, current
weight, effective history depth, token replacements, and CCD fallback counts.
