# Abstract

Paired comparison of two embedding initializations (A, B) for a byte-level
language model trained on enwik8 (~85M parameters, 20,000 steps). Five seed
pairs (seeds 1-5); within a pair both arms share the seed, data order, and
all non-embedding initialization. Primary metric: final nats/byte on a held
out split, lower is better. The registered hypothesis is that arm A attains
lower nats/byte than arm B. This packet contains the pre-registration, the
captured per-run JSONL series, a self-contained validated reader, and a
manifest of file hashes and registration timestamps.
