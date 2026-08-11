Parameter budgeting notes (see BUILD_SPEC "Experiment ladder").

Baseline block cost at d_model=768: attn 4*d^2 = 2.36M, MLP 2*4*d^2 = 4.72M
=> 7.08M / layer.
  e0-byte / e1-*: 12 layers = 84.9M + 0.20M byte embedding  ~= 85.1M
  e0-bpe        :  6 layers = 42.5M + 38.6M GPT-2 embedding ~= 81.1M (tied)

nGPT block adds the SwiGLU gate: attn 4*d^2 = 2.36M, MLP (2*4*d^2 + 4*d^2)
= 7.08M => 9.44M / layer.
  e2 / e3       :  9 layers = 84.9M + 0.20M byte embedding  ~= 85.1M
So depth differs between the baseline and hyperspherical arms *on purpose*:
the ladder is parameter-matched, not depth-matched.
