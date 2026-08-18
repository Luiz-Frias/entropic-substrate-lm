# E1 review packet arm mapping — DISCLOSED

Unsealed 2026-08-18 after the blind review completed and was filed as
`e1-review.md` (per `docs/peer-review-protocol.md`, the mapping is disclosed
only after the review is filed). Sealed location during the review: the
operator-side memory directory, outside this repository.

Mapping: **A = e1-fisher, B = e1-random** (A = registered-favored arm by
protocol construction; the blind hides which initialization A is).

Scrub transform applied to every record: run -> arm-<X>; config.name ->
arm-<X>; config.model.embedding_init -> init-<X>; config.model.
fisher_init_path and config.model.anchor_mode dropped. reader.py =
telemetry.py with all docstrings/comments stripped (ast.unparse), the
package logging import shimmed, and the operator CLI removed.

| capture file | capture md5 | packet file | packet md5 | run |
|---|---|---|---|---|
| e1-fisher-1fca0b15-e1-fisher-s5.jsonl | 8c8e8cf48db0e939bd39ca9241d64831 | A-seed5.jsonl | 3df29981e57e947d02cefa3fb70b01e3 | run_id 1fca0b15ddb2467598252de1f28fec38 |
| e1-fisher-56d129a1-e1-fisher-s2.jsonl | e57e834fc356ef35b9618c15ae151af8 | A-seed2.jsonl | 4127a48c7f184cf245f6b16b285fceb1 | run_id 56d129a11a5646b683be8237e654dd7b |
| e1-fisher-6015d8d1-e1-fisher-s4.jsonl | 8269945108deacfaa8f0adcce83c61bf | A-seed4.jsonl | 308038a9a31c6a17c47692f0d599710e | run_id 6015d8d1b9ce4c2da3c793fda916fd12 |
| e1-fisher-7d84caf7-e1-fisher-s3.jsonl | ba7d61140cc2b7cc281aa881d2f67e35 | A-seed3.jsonl | a846d3eceace2bac3ce0d09aef75db72 | run_id 7d84caf73a8c4c88aa6b4d91db387932 |
| e1-fisher-bd0fb0d3-e1-fisher-s1.jsonl | f9e3498ab923b27ae233aa9e1b85c2bc | A-seed1.jsonl | 05bfd13e6504abfd55484b6daab61700 | run_id bd0fb0d341b94666bb51f38c6aabe57b |
| e1-random-2cc15d7b-e1-random-s3.jsonl | 5af2c2a8a21fb707d83ad406ec4ed44c | B-seed3.jsonl | d19202904c1256887d361c1bdc21888b | run_id 2cc15d7bf6c242829c3d7a2b9e032598 |
| e1-random-3629cca4-e1-random-s4.jsonl | f34cad979e70bfc08decf4bab5456411 | B-seed4.jsonl | c0c17ea0943944989ee0f9de461fab20 | run_id 3629cca4446c4697ac08af7c1818e44a |
| e1-random-5ec7c790-e1-random-s5.jsonl | 6cf44777014435a450b7d8d4e5a8368c | B-seed5.jsonl | a2a5a529059110a8e23c967261d5f771 | run_id 5ec7c79068344a3b832a3d32061b2a7b |
| e1-random-7bc2a00f-e1-random-s1.jsonl | 534ee3be5293f89b72f2fbd521e5a4a9 | B-seed1.jsonl | c8e6570c9990272c3959f4ca2fb91781 | run_id 7bc2a00fb2d54607af08e755c09edb13 |
| e1-random-d2cbbc78-e1-random-s2.jsonl | f8df7e6ae83b47ada84ad5146ede1aa3 | B-seed2.jsonl | 1fddf06adc61d80cefe0086db0aab514 | run_id d2cbbc7858e1431d961e19dd471274e7 |
