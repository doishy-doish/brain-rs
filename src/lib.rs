use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn zeros(size: usize) -> Vec<f64> {
    vec![0.0; size]
}

struct Rng(u64);

impl Rng {
    fn next_unit(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let bits = x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        (bits as f64) / ((1u64 << 53) as f64)
    }

    fn weight(&mut self) -> f64 {
        self.next_unit() * 0.4 - 0.2
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_unit() * n as f64) as usize
    }
}

fn time_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E37_79B9_7F4A_7C15)
}

#[derive(Clone, Debug)]
pub struct Lookup {
    pub keys: Vec<String>,
    index: HashMap<String, usize>,
}

impl Lookup {
    pub fn from_keys(keys: Vec<String>) -> Self {
        let mut index = HashMap::new();
        for (i, k) in keys.iter().enumerate() {
            index.entry(k.clone()).or_insert(i);
        }
        Lookup { keys, index }
    }

    pub fn build(hashes: &[Vec<(String, f64)>]) -> Self {
        let mut keys = Vec::new();
        let mut seen = HashSet::new();
        for h in hashes {
            for (k, _) in h.iter() {
                if seen.insert(k.clone()) {
                    keys.push(k.clone());
                }
            }
        }
        Lookup::from_keys(keys)
    }

    pub fn from_array(values: &[String]) -> Self {
        let mut keys: Vec<String> = values.to_vec();
        keys.reverse();
        Lookup::from_keys(keys)
    }

    pub fn to_array(&self, hash: &HashMap<String, f64>) -> Vec<f64> {
        self.keys
            .iter()
            .map(|k| hash.get(k).copied().unwrap_or(0.0))
            .collect()
    }

    pub fn to_hash(&self, array: &[f64]) -> HashMap<String, f64> {
        self.keys
            .iter()
            .enumerate()
            .map(|(i, k)| (k.clone(), array[i]))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

#[derive(Clone)]
pub struct Options {
    pub learning_rate: f64,
    pub momentum: f64,
    pub binary_thresh: f64,
    pub hidden_sizes: Option<Vec<usize>>,
    pub seed: u64,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            learning_rate: 0.3,
            momentum: 0.1,
            binary_thresh: 0.5,
            hidden_sizes: None,
            seed: time_seed(),
        }
    }
}

pub type ProgressCallback = Arc<Mutex<Box<dyn FnMut(f64, usize) + Send>>>;

#[derive(Clone, Default)]
pub struct TrainOpts {
    pub iterations: usize,
    pub error_thresh: f64,
    pub learning_rate: Option<f64>,
    pub log: bool,
    pub log_period: usize,
    pub callback: Option<ProgressCallback>,
}

impl TrainOpts {
    pub fn with_iterations(mut self, n: usize) -> Self {
        self.iterations = n;
        self
    }

    pub fn with_error_thresh(mut self, e: f64) -> Self {
        self.error_thresh = e;
        self
    }
}

#[derive(Clone, Debug)]
pub struct TrainStats {
    pub error: f64,
    pub iterations: usize,
}

#[derive(Clone, Debug, Default)]
pub struct TestStats {
    pub error: f64,
    pub misclasses: Vec<usize>,
    pub true_pos: usize,
    pub true_neg: usize,
    pub false_pos: usize,
    pub false_neg: usize,
    pub total: usize,
    pub precision: f64,
    pub recall: f64,
    pub accuracy: f64,
}

pub struct NeuralNetwork {
    pub learning_rate: f64,
    pub momentum: f64,
    pub binary_thresh: f64,
    pub hidden_sizes: Option<Vec<usize>>,
    pub input_lookup: Option<Lookup>,
    pub output_lookup: Option<Lookup>,
    sizes: Vec<usize>,
    output_layer: usize,
    biases: Vec<Vec<f64>>,
    weights: Vec<Vec<Vec<f64>>>,
    outputs: Vec<Vec<f64>>,
    deltas: Vec<Vec<f64>>,
    changes: Vec<Vec<Vec<f64>>>,
    errors: Vec<Vec<f64>>,
    rng: Rng,
}

impl NeuralNetwork {
    pub fn new(options: Options) -> Self {
        NeuralNetwork {
            learning_rate: options.learning_rate,
            momentum: options.momentum,
            binary_thresh: options.binary_thresh,
            hidden_sizes: options.hidden_sizes,
            input_lookup: None,
            output_lookup: None,
            sizes: Vec::new(),
            output_layer: 0,
            biases: Vec::new(),
            weights: Vec::new(),
            outputs: Vec::new(),
            deltas: Vec::new(),
            changes: Vec::new(),
            errors: Vec::new(),
            rng: Rng(options.seed),
        }
    }

    pub fn initialize(&mut self, sizes: &[usize]) {
        self.sizes = sizes.to_vec();
        self.output_layer = sizes.len() - 1;
        self.biases = Vec::new();
        self.weights = Vec::new();
        self.outputs = Vec::new();
        self.deltas = Vec::new();
        self.changes = Vec::new();
        self.errors = Vec::new();
        for (layer, &size) in sizes.iter().enumerate() {
            self.deltas.push(zeros(size));
            self.errors.push(zeros(size));
            self.outputs.push(zeros(size));
            if layer > 0 {
                self.biases.push(Vec::new());
                self.weights.push(Vec::new());
                self.changes.push(Vec::new());
                let prev = sizes[layer - 1];
                for _ in 0..size {
                    self.biases[layer].push(self.rng.weight());
                    self.weights[layer].push((0..prev).map(|_| self.rng.weight()).collect());
                    self.changes[layer].push(zeros(prev));
                }
            } else {
                self.biases.push(Vec::new());
                self.weights.push(Vec::new());
                self.changes.push(Vec::new());
            }
        }
    }

    pub fn run_input(&mut self, input: &[f64]) -> Vec<f64> {
        self.outputs[0] = input.to_vec();
        for layer in 1..=self.output_layer {
            for node in 0..self.sizes[layer] {
                let mut sum = self.biases[layer][node];
                for (k, w) in self.weights[layer][node].iter().enumerate() {
                    sum += w * self.outputs[layer - 1][k];
                }
                self.outputs[layer][node] = sigmoid(sum);
            }
        }
        self.outputs[self.output_layer].clone()
    }

    pub fn run(&mut self, input: &[f64]) -> Vec<f64> {
        self.run_input(input)
    }

    pub fn run_hash(&mut self, input: &HashMap<String, f64>) -> HashMap<String, f64> {
        let lookup = self.input_lookup.clone().unwrap_or_else(|| Lookup::from_keys(vec![]));
        let array = lookup.to_array(input);
        let output = self.run_input(&array);
        match &self.output_lookup {
            Some(lookup) => lookup.to_hash(&output),
            None => HashMap::new(),
        }
    }

    pub fn train(&mut self, data: &[(Vec<f64>, Vec<f64>)], options: TrainOpts) -> TrainStats {
        let iterations = if options.iterations == 0 { 20000 } else { options.iterations };
        let error_thresh = if options.error_thresh == 0.0 { 0.005 } else { options.error_thresh };
        let log_period = if options.log_period == 0 { 10 } else { options.log_period };
        let learning_rate = options.learning_rate.unwrap_or(self.learning_rate);
        if learning_rate == 0.0 {
            self.learning_rate = 0.3;
        }
        let input_size = data[0].0.len();
        let output_size = data[0].1.len();
        let hidden = self.hidden_sizes.clone().unwrap_or_else(|| {
            vec![std::cmp::max(3, input_size / 2)]
        });
        let mut sizes = vec![input_size];
        sizes.extend(hidden);
        sizes.push(output_size);
        self.initialize(&sizes);
        let mut callback = options.callback;
        let mut error = 1.0;
        let mut i = 0;
        while i < iterations && error > error_thresh {
            let mut sum = 0.0;
            for (input, target) in data {
                sum += self.train_pattern(input, target, learning_rate);
            }
            error = sum / data.len() as f64;
            if options.log && i % log_period == 0 {
                eprintln!("iterations: {i} training error: {error}");
            }
            if let Some(cb) = callback.as_ref() {
                if i % log_period == 0 {
                    cb.lock().unwrap()(error, i);
                }
            }
            i += 1;
        }
        TrainStats { error, iterations: i }
    }

    pub fn train_hash(
        &mut self,
        data: &[(Vec<(String, f64)>, Vec<(String, f64)>)],
        options: TrainOpts,
    ) -> TrainStats {
        if self.input_lookup.is_none() {
            let inputs: Vec<Vec<(String, f64)>> = data.iter().map(|d| d.0.clone()).collect();
            self.input_lookup = Some(Lookup::build(&inputs));
        }
        if self.output_lookup.is_none() {
            let outputs: Vec<Vec<(String, f64)>> = data.iter().map(|d| d.1.clone()).collect();
            self.output_lookup = Some(Lookup::build(&outputs));
        }
        let arrays: Vec<(Vec<f64>, Vec<f64>)> = data
            .iter()
            .map(|d| {
                let in_map: HashMap<String, f64> = d.0.iter().cloned().collect();
                let out_map: HashMap<String, f64> = d.1.iter().cloned().collect();
                let input = self.input_lookup.as_ref().unwrap().to_array(&in_map);
                let output = self.output_lookup.as_ref().unwrap().to_array(&out_map);
                (input, output)
            })
            .collect();
        self.train(&arrays, options)
    }

    pub fn train_pattern(&mut self, input: &[f64], target: &[f64], learning_rate: f64) -> f64 {
        self.run_input(input);
        self.calculate_deltas(target);
        self.adjust_weights(learning_rate);
        mse(&self.errors[self.output_layer])
    }

    fn calculate_deltas(&mut self, target: &[f64]) {
        for layer in (0..=self.output_layer).rev() {
            for node in 0..self.sizes[layer] {
                let output = self.outputs[layer][node];
                let mut error = 0.0;
                if layer == self.output_layer {
                    error = target[node] - output;
                } else {
                    for k in 0..self.deltas[layer + 1].len() {
                        error += self.deltas[layer + 1][k] * self.weights[layer + 1][k][node];
                    }
                }
                self.errors[layer][node] = error;
                self.deltas[layer][node] = error * output * (1.0 - output);
            }
        }
    }

    fn adjust_weights(&mut self, learning_rate: f64) {
        for layer in 1..=self.output_layer {
            for node in 0..self.sizes[layer] {
                let delta = self.deltas[layer][node];
                for k in 0..self.outputs[layer - 1].len() {
                    let incoming = self.outputs[layer - 1][k];
                    let change = learning_rate * delta * incoming
                        + self.momentum * self.changes[layer][node][k];
                    self.changes[layer][node][k] = change;
                    self.weights[layer][node][k] += change;
                }
                self.biases[layer][node] += learning_rate * delta;
            }
        }
    }

    pub fn test(&mut self, data: &[(Vec<f64>, Vec<f64>)]) -> TestStats {
        let binary = data[0].1.len() == 1;
        let mut stats = TestStats {
            total: data.len(),
            ..Default::default()
        };
        let mut sum = 0.0;
        for (idx, (input, target)) in data.iter().enumerate() {
            let output = self.run_input(input);
            let actual;
            let expected;
            if binary {
                actual = if output[0] > self.binary_thresh { 1 } else { 0 };
                expected = target[0] as usize;
            } else {
                actual = argmax(&output);
                expected = argmax(target);
            }
            if actual != expected {
                stats.misclasses.push(idx);
            }
            if binary {
                match (actual, expected) {
                    (0, 0) => stats.true_neg += 1,
                    (1, 1) => stats.true_pos += 1,
                    (0, 1) => stats.false_neg += 1,
                    _ => stats.false_pos += 1,
                }
            }
            let errs: Vec<f64> = output.iter().zip(target.iter()).map(|(o, t)| t - o).collect();
            sum += mse(&errs);
        }
        stats.error = sum / data.len() as f64;
        if binary {
            stats.precision = stats.true_pos as f64 / (stats.true_pos + stats.false_pos) as f64;
            stats.recall = stats.true_pos as f64 / (stats.true_pos + stats.false_neg) as f64;
            stats.accuracy =
                (stats.true_neg + stats.true_pos) as f64 / data.len() as f64;
        }
        stats
    }

    fn node_name(&self, layer: usize, j: usize) -> String {
        if layer == 0 {
            if let Some(lookup) = &self.input_lookup {
                return lookup.keys[j].clone();
            }
        }
        if layer == self.output_layer {
            if let Some(lookup) = &self.output_lookup {
                return lookup.keys[j].clone();
            }
        }
        j.to_string()
    }

    fn prev_names(&self, layer: usize) -> Vec<String> {
        if layer == 1 {
            if let Some(lookup) = &self.input_lookup {
                return lookup.keys.clone();
            }
        }
        (0..self.sizes[layer - 1]).map(|k| k.to_string()).collect()
    }

    pub fn to_json(&self) -> Value {
        let mut layers = Vec::new();
        for (li, size) in self.sizes.iter().enumerate() {
            let mut layer = Map::new();
            for j in 0..*size {
                let mut node = Map::new();
                if li > 0 {
                    node.insert("bias".to_string(), Value::from(self.biases[li][j]));
                    let mut w = Map::new();
                    for (k, name) in self.prev_names(li).iter().enumerate() {
                        w.insert(name.clone(), Value::from(self.weights[li][j][k]));
                    }
                    node.insert("weights".to_string(), Value::Object(w));
                }
                layer.insert(self.node_name(li, j), Value::Object(node));
            }
            layers.push(Value::Object(layer));
        }
        let mut root = Map::new();
        root.insert("layers".to_string(), Value::Array(layers));
        root.insert(
            "outputLookup".to_string(),
            Value::Bool(self.output_lookup.is_some()),
        );
        root.insert(
            "inputLookup".to_string(),
            Value::Bool(self.input_lookup.is_some()),
        );
        Value::Object(root)
    }

    pub fn from_json(&mut self, value: &Value) -> Result<(), String> {
        let layers = value
            .get("layers")
            .and_then(|l| l.as_array())
            .ok_or_else(|| "missing layers array".to_string())?;
        let input_flag = value.get("inputLookup").and_then(|v| v.as_bool()).unwrap_or(false);
        let output_flag = value
            .get("outputLookup")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let size = layers.len();
        self.output_layer = size - 1;
        self.sizes = vec![0; size];
        self.weights = vec![Vec::new(); size];
        self.biases = vec![Vec::new(); size];
        self.outputs = vec![Vec::new(); size];
        self.deltas = vec![Vec::new(); size];
        self.changes = vec![Vec::new(); size];
        self.errors = vec![Vec::new(); size];
        for (i, layer) in layers.iter().enumerate() {
            let obj = layer.as_object().ok_or_else(|| "layer not an object".to_string())?;
            let names: Vec<String> = obj.keys().cloned().collect();
            let has_zero = obj.contains_key("0");
            if i == 0 && (!has_zero || input_flag) {
                self.input_lookup = Some(Lookup::from_keys(names.clone()));
            } else if i == self.output_layer && (!has_zero || output_flag) {
                self.output_lookup = Some(Lookup::from_keys(names.clone()));
            }
            self.sizes[i] = names.len();
            let prev: Vec<String> = if i == 1 {
                if let Some(lookup) = &self.input_lookup {
                    lookup.keys.clone()
                } else {
                    (0..self.sizes[0]).map(|k| k.to_string()).collect()
                }
            } else if i > 1 {
                (0..self.sizes[i - 1]).map(|k| k.to_string()).collect()
            } else {
                Vec::new()
            };
            for node in names.iter() {
                let entry = obj.get(node).ok_or_else(|| "missing node".to_string())?;
                if i == 0 {
                    continue;
                }
                let bias = entry
                    .get("bias")
                    .and_then(|b| b.as_f64())
                    .ok_or_else(|| "missing bias".to_string())?;
                self.biases[i].push(bias);
                let wobj = entry
                    .get("weights")
                    .and_then(|w| w.as_object())
                    .ok_or_else(|| "missing weights".to_string())?;
                let mut row = Vec::new();
                for name in prev.iter() {
                    let w = wobj
                        .get(name)
                        .and_then(|v| v.as_f64())
                        .ok_or_else(|| format!("missing weight {name}"))?;
                    row.push(w);
                }
                self.weights[i].push(row);
            }
            let n = self.sizes[i];
            self.outputs[i] = zeros(n);
            self.deltas[i] = zeros(n);
            self.errors[i] = zeros(n);
            if i > 0 {
                self.changes[i] = vec![zeros(prev.len()); n];
            }
        }
        Ok(())
    }
}

fn argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in values.iter().enumerate() {
        if v > values[best] {
            best = i;
        }
    }
    best
}

fn mse(errors: &[f64]) -> f64 {
    errors.iter().map(|e| e * e).sum::<f64>() / errors.len() as f64
}

#[derive(Clone, Debug, Default)]
pub struct CvAvgs {
    pub error: f64,
    pub train_time_ms: u128,
    pub test_time_ms: u128,
    pub iterations: f64,
    pub train_error: f64,
}

#[derive(Clone, Debug, Default)]
pub struct CvStats {
    pub avgs: CvAvgs,
    pub true_pos: usize,
    pub true_neg: usize,
    pub false_pos: usize,
    pub false_neg: usize,
    pub total: usize,
    pub precision: f64,
    pub recall: f64,
    pub accuracy: f64,
    pub test_size: usize,
    pub train_size: usize,
    pub misclass_total: usize,
}

pub fn cross_validate(
    options: Options,
    data: &[(Vec<f64>, Vec<f64>)],
    train_options: TrainOpts,
    k: usize,
    seed: u64,
) -> CvStats {
    let k = if k == 0 { 4 } else { k };
    let mut order: Vec<usize> = (0..data.len()).collect();
    let mut rng = Rng(seed);
    for i in (1..order.len()).rev() {
        let j = rng.below(i + 1);
        order.swap(i, j);
    }
    let size = std::cmp::max(1, data.len() / k);
    let mut avgs = CvAvgs::default();
    let mut stats = CvStats::default();
    for fold in 0..k {
        let mut train_set = Vec::new();
        let mut test_set = Vec::new();
        for (pos, &idx) in order.iter().enumerate() {
            if pos / size == fold && pos < fold * size + size {
                test_set.push(data[idx].clone());
            } else {
                train_set.push(data[idx].clone());
            }
        }
        let mut net = NeuralNetwork::new(options.clone());
        let begin_train = Instant::now();
        let train_stats = net.train(&train_set, train_options.clone());
        let begin_test = Instant::now();
        let test_stats = net.test(&test_set);
        let end_test = Instant::now();
        avgs.error += test_stats.error;
        avgs.train_time_ms += begin_test.duration_since(begin_train).as_millis();
        avgs.test_time_ms += end_test.duration_since(begin_test).as_millis();
        avgs.iterations += train_stats.iterations as f64;
        avgs.train_error += train_stats.error;
        stats.true_pos += test_stats.true_pos;
        stats.true_neg += test_stats.true_neg;
        stats.false_pos += test_stats.false_pos;
        stats.false_neg += test_stats.false_neg;
        stats.total += test_stats.total;
        stats.misclass_total += test_stats.misclasses.len();
    }
    let kf = k as f64;
    avgs.error /= kf;
    avgs.train_time_ms /= k as u128;
    avgs.test_time_ms /= k as u128;
    avgs.iterations /= kf;
    avgs.train_error /= kf;
    stats.avgs = avgs;
    if stats.true_pos + stats.false_pos > 0 {
        stats.precision = stats.true_pos as f64 / (stats.true_pos + stats.false_pos) as f64;
    }
    if stats.true_pos + stats.false_neg > 0 {
        stats.recall = stats.true_pos as f64 / (stats.true_pos + stats.false_neg) as f64;
    }
    if stats.total > 0 {
        stats.accuracy = (stats.true_neg + stats.true_pos) as f64 / stats.total as f64;
    }
    stats.test_size = size;
    stats.train_size = data.len() - size;
    stats
}
