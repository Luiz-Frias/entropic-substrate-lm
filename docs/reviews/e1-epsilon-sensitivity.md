# ε-sensitivity map — E1 stage 1 (adversarial counterfactuals)

Derivation, not new measurement: every row applies the registered verdict
rule to the blind review's recomputed stage-1 CI **[0.00081, 0.00897]**
nats/byte (`docs/reviews/e1-review.md` §3; mean d = 0.00489, d = random −
fisher, positive favors the Fisher init). The direction clause ("CI entirely
above 0 → H1 supported") contains no ε, so the **SUPPORTED verdict is
identical in every row** — only the practical-relevance clause ("CI entirely
below +ε → no practically relevant improvement") moves.

| ε (nats/byte) | CI vs ε | practical-relevance reading |
|---|---|---|
| 0.1 | CI entirely below | no practically relevant improvement (emphatic) |
| 0.05 | CI entirely below | no practically relevant improvement (emphatic) |
| **0.01 (ratified)** | CI entirely below (0.00897 < 0.01) | **no practically relevant improvement** |
| 0.005 | CI straddles ε | clause does not fire — relevance UNRESOLVED |
| 0.001 | CI straddles ε (barely: 0.00081 < 0.001) | UNRESOLVED, leaning relevant (mean ≈ 5× ε) |
| 0.0001 | CI entirely above ε | effect exceeds the relevance bar decisively |
| none set | — | open interpretation: the CI is the complete evidence. Hostile reading: "significant but ≈0.6% of baseline — negligible." Sympathetic reading: "a free, zero-runtime-cost improvement with a confident sign." Neither is forced — which is precisely the failure mode a pre-declared ε exists to prevent. |

Structural facts the map makes visible:

- The flip zone for practical relevance is exactly the CI: any ε inside
  [0.00081, 0.00897] yields UNRESOLVED; below it, "relevant"; above it, "not
  relevant". The ratified ε = 0.01 sits 0.00103 above the CI's upper bound —
  within one SE (0.00117) of it, which is why the blind review flagged its
  post-run timing as MAJOR-1.
- No ε choice anywhere on the axis can strengthen the effect itself: mean d
  ≈ 0.6% of the ~0.77 baseline at this regime (85M params, enwik8, 20k
  steps), 5/5 pairs positive. ε only names the bar the effect must clear to
  be called *practically* meaningful.
- Adversarial bottom line: an operator free to pick ε post-hoc could have
  chosen the "relevant" reading (ε ≤ 0.0008) or the "not relevant" reading
  (ε ≥ 0.009). The ratified 0.01 produces the reading LESS favorable to the
  registered hypothesis's practical impact — the conservative direction —
  and the direction verdict never moves regardless.
