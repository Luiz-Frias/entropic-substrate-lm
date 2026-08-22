# Abstract

Paired sufficiency comparison of two embedding-parameter freezing variants
(A, B) against a shared trainable control (C), for a byte-level language
model trained on enwik8 (~85M parameters, 20,000 steps). Four seeds
(1-4); within a seed all three arms share the seed, the data order, and
every non-embedding initialization. Each variant freezes a registered subset
of the embedding parameters at their initialized values; WHICH subset each
of A and B freezes is sealed. C trains all parameters.

Primary metric: the mean of the final five held-out eval readings of
nats/byte, lower is better. The registered question is SUFFICIENCY — whether
freezing costs less than a pre-registered practical-relevance threshold —
not direction. This packet contains the pre-registration, the captured
per-run JSONL series, a self-contained validated reader, and a manifest of
file hashes and registration timestamps.
