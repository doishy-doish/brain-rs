use brain_rs::{cross_validate, Lookup, NeuralNetwork, Options, TrainOpts};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

fn xor_data() -> Vec<(Vec<f64>, Vec<f64>)> {
    vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ]
}

fn seeded(seed: u64) -> Options {
    Options { seed, ..Default::default() }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

#[test]
fn lookup_build_preserves_first_seen_order() {
    let hashes = vec![
        vec![("a".to_string(), 1.0)],
        vec![("b".to_string(), 6.0), ("c".to_string(), 7.0)],
    ];
    let lookup = Lookup::build(&hashes);
    assert_eq!(lookup.keys, vec!["a".to_string(), "b".to_string(), "c".to_string()]);
}

#[test]
fn lookup_to_array_fills_missing_with_zero() {
    let lookup = Lookup::from_keys(vec!["a".to_string(), "b".to_string()]);
    let mut hash = HashMap::new();
    hash.insert("a".to_string(), 6.0);
    assert_eq!(lookup.to_array(&hash), vec![6.0, 0.0]);
}

#[test]
fn lookup_to_hash_round_trip() {
    let lookup = Lookup::from_keys(vec!["a".to_string(), "b".to_string()]);
    let hash = lookup.to_hash(&[6.0, 7.0]);
    assert_eq!(hash.get("a"), Some(&6.0));
    assert_eq!(hash.get("b"), Some(&7.0));
}

#[test]
fn lookup_from_array_reverses_like_original() {
    let lookup = Lookup::from_array(&["x".to_string(), "y".to_string()]);
    assert_eq!(lookup.keys, vec!["y".to_string(), "x".to_string()]);
}

#[test]
fn forward_pass_matches_original_weights() {
    let mut net = NeuralNetwork::new(seeded(1));
    let json: serde_json::Value = serde_json::from_str(
        r#"{"layers":[{"0":{},"1":{}},{"0":{"bias":2.3657213180865573,"weights":{"0":-6.055783560454246,"1":-6.128060697107127}},"1":{"bias":3.231779940182746,"weights":{"0":-2.2659628274405743,"1":-2.4047546827234845}},"2":{"bias":4.569677284250683,"weights":{"0":-3.2219187299844863,"1":-3.118439256168767}}},{"0":{"bias":-4.209911854839983,"weights":{"0":-8.96607085968405,"1":3.926204674865073,"2":5.677294311324174}}}],"outputLookup":false,"inputLookup":false}"#,
    )
    .unwrap();
    net.from_json(&json).unwrap();
    let expected = [
        (vec![0.0, 0.0], 0.04696076023200244),
        (vec![0.0, 1.0], 0.9487185136776114),
        (vec![1.0, 0.0], 0.9489122973278871),
        (vec![1.0, 1.0], 0.06711330028898424),
    ];
    for (input, want) in expected {
        let got = net.run(&input);
        assert!(close(got[0], want), "input {input:?}: got {} want {want}", got[0]);
    }
}

#[test]
fn train_xor_converges_like_bitwise_test() {
    let mut net = NeuralNetwork::new(seeded(42));
    let stats = net.train(
        &xor_data(),
        TrainOpts {
            error_thresh: 0.003,
            ..Default::default()
        },
    );
    assert!(stats.error < 0.003, "error {}", stats.error);
    let wiggle = 0.1;
    for (input, target) in xor_data() {
        let output = net.run(&input);
        assert!(
            output[0] < target[0] + wiggle && output[0] > target[0] - wiggle,
            "input {input:?}: output {} target {}",
            output[0],
            target[0]
        );
    }
}

#[test]
fn test_stats_report_on_trained_xor() {
    let mut net = NeuralNetwork::new(seeded(42));
    net.train(
        &xor_data(),
        TrainOpts {
            error_thresh: 0.003,
            ..Default::default()
        },
    );
    let stats = net.test(&xor_data());
    assert!(stats.misclasses.is_empty());
    assert!((stats.accuracy - 1.0).abs() < 1e-12);
}

#[test]
fn hash_train_and_run_match_original() {
    let data = vec![
        (vec![("a".to_string(), 1.0)], vec![("x".to_string(), 1.0)]),
        (vec![("b".to_string(), 1.0)], vec![("y".to_string(), 1.0)]),
    ];
    let mut net = NeuralNetwork::new(seeded(7));
    net.train_hash(
        &data,
        TrainOpts {
            error_thresh: 0.003,
            iterations: 5000,
            ..Default::default()
        },
    );
    let mut input_a = HashMap::new();
    input_a.insert("a".to_string(), 1.0);
    let out_a = net.run_hash(&input_a);
    assert!(out_a["x"] > 0.9, "x = {}", out_a["x"]);
    assert!(out_a["y"] < 0.1, "y = {}", out_a["y"]);
    let mut input_b = HashMap::new();
    input_b.insert("b".to_string(), 1.0);
    let out_b = net.run_hash(&input_b);
    assert!(out_b["y"] > 0.9, "y = {}", out_b["y"]);
    assert!(out_b["x"] < 0.1, "x = {}", out_b["x"]);
}

#[test]
fn json_round_trip_preserves_behavior() {
    let mut net = NeuralNetwork::new(seeded(42));
    net.train(
        &xor_data(),
        TrainOpts {
            error_thresh: 0.01,
            ..Default::default()
        },
    );
    let json = net.to_json();
    let mut net2 = NeuralNetwork::new(seeded(999));
    net2.from_json(&json).unwrap();
    for (input, _) in xor_data() {
        let a = net.run(&input);
        let b = net2.run(&input);
        assert!(close(a[0], b[0]), "input {input:?}: {a:?} vs {b:?}");
    }
}

#[test]
fn from_json_rejects_garbage() {
    let mut net = NeuralNetwork::new(seeded(1));
    assert!(net.from_json(&serde_json::json!({"nope": 1})).is_err());
}

#[test]
fn callback_fires_during_train() {
    let calls = Arc::new(Mutex::new(0usize));
    let counter = calls.clone();
    let opts = TrainOpts {
        error_thresh: 0.003,
        callback: Some(Arc::new(Mutex::new(
            Box::new(move |_error: f64, _iter: usize| {
                *counter.lock().unwrap() += 1;
            }) as Box<dyn FnMut(f64, usize) + Send>,
        ))),
        ..Default::default()
    };
    let mut net = NeuralNetwork::new(seeded(42));
    net.train(&xor_data(), opts);
    assert!(*calls.lock().unwrap() > 0);
}

#[test]
fn cross_validate_xor_folds() {
    let stats = cross_validate(
        seeded(11),
        &xor_data(),
        TrainOpts {
            error_thresh: 0.05,
            iterations: 5000,
            ..Default::default()
        },
        2,
        1234,
    );
    assert_eq!(stats.total, 4);
    assert_eq!(stats.test_size, 2);
    assert_eq!(stats.train_size, 2);
    assert!(stats.accuracy >= 0.0 && stats.accuracy <= 1.0);
}
