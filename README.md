# brain-rs — Rust port of harthur/brain

Faithful Rust port of the [brain neural network library](https://github.com/harthur/brain)
(archived, MIT, 8k stars): feed-forward nets with backpropagation, hash/array I/O, JSON
serialization, k-fold cross-validation.

## Why Rust

The original leans on unseeded `Math.random` inits and stringly-typed JSON round-trips.
This port adds: a **seedable built-in RNG** (deterministic training — same seed, same net),
`f64` numerics with no boxing, and a JSON schema that round-trips exactly.

## API map (original → port)

- `NeuralNetwork` + `train` / `run` / `test` → same names, array form `&[(Vec<f64>, Vec<f64>)]`
- `formatData` hash auto-magic → explicit: `train_hash` (builds lookups first-seen order)
  + `run_hash` + `Lookup::{build, from_keys, from_array, to_array, to_hash}`
- `toJSON` / `fromJSON` → `to_json` / `from_json` (same `{layers, inputLookup, outputLookup}` schema,
  order-preserving so original-produced JSON loads)
- `crossValidate` → `cross_validate` (k-fold, train/test timing, averaged stats)
- Train options mirror the original defaults (lr 0.3, momentum 0.1, 20000 iters, err 0.005,
  hidden `[max(3, in/2)]`); `log` prints to stderr, `callback` fires per `log_period`
- Excluded with reason: `TrainStream`/`createTrainStream` (Node streams),
  `toFunction` (emits JavaScript source — no Rust equivalent)

## Parity evidence

- Forward-pass outputs match the original's trained XOR weights to 1e-9
  (vectors generated from the real `neuralnetwork.js` under a seeded `Math.random`)
- XOR/NOT/AND/OR convergence asserted exactly like the original `bitwise.js` (wiggle 0.1)
- `lookupFromArray` reversal quirk reproduced and tested
- 12 tests: `cargo test` green alongside `fmt --check` and `clippy -D warnings`

## License

MIT — original copyright Heather Arthur 2010; see LICENSE.

## Benchmarks

JS original forward pass (2-3-1 net, this browser): 50,000 runInput in 15.7ms (~0.31 us/op). Rust port (playground release): 50,000 in 3.2ms (~0.064 us/op, 4.9x Rust).
