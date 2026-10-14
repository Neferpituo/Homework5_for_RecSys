/**
 * week5/script.js — Matrix Factorization with TensorFlow.js (HW5).
 *
 * This is a plain (classic) script, NOT an ES module. `index.html` loads the
 * TensorFlow.js runtime from a CDN before this file, exposing the global `tf`:
 *
 *   <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"></script>
 *   <script src="data.js"></script>
 *   <script src="script.js"></script>
 *
 * `week5/data.js` (loaded before this file) exposes the MovieLens 100k data
 * layer as globals: `movies`, `ratings`, `numUsers`, `numMovies`,
 * `ratingMatrix`, `SPLIT_EXCLUSIONS`, `loadData()` and `splitByTimestamp()`.
 * This file reads those globals, renders a dataset / split summary, and wires
 * the page controls. The factorization itself is built on `tf.Variable`s and
 * tensor ops so it can run on the WebGL (GPU) backend.
 *
 * BACKEND: `initTf()` prefers WebGL, falls back to CPU, and writes the chosen
 * backend into `#tf-status`. If the CDN did not load (`typeof tf ===
 * 'undefined'`) the page shows a clear message and stops instead of throwing.
 *
 * WHAT YOU MUST IMPLEMENT (`TODO(hw5)` — each stub throws until you write it):
 *   1. `predictRating`        — scalar gather + dot + globalMean, clamp [1, 5].
 *   2. `initializeFactors`    — two `tf.Variable`s + the global mean.
 *   3. `trainMF`              — batched tensor training loop with L2 / Adam.
 *   4. `selectBestCheckpoint` — earliest entry with the smallest val RMSE.
 *   5. `recommendTopK`        — `matMul` + `tf.topk`, exclude train/val items.
 *   6. `recallAtK`            — Recall@K against relevant (>= 4.0) test items.
 *
 * PROVIDED for you (scaffolding): `initTf`, the shared metric helpers
 * (`computeRMSEFor`, `computeBaselineRMSE`), formatting helpers, every DOM
 * renderer, the hyper-parameter reader, the training pipeline wrapper, and the
 * automated test harness. Leave the provided code as-is and implement only the
 * six stubs.
 *
 * The harness reports unimplemented stubs as PENDING rather than FAIL, so the
 * page stays usable before and after the assignment is done.
 *
 * @module week5/script
 */

// ---------------------------------------------------------------------------
// Typedefs
// ---------------------------------------------------------------------------

/**
 * @typedef {Object} Movie
 * @property {number} id
 * @property {string} title
 * @property {string[]} genres
 */

/**
 * @typedef {Object} Rating
 * @property {number} userId
 * @property {number} movieId
 * @property {number} rating
 * @property {number} timestamp
 */

/**
 * @typedef {Object} Model
 * @property {tf.Variable} userFactors Dense factor matrix `[numUsers + 1, F]`,
 *   row index = raw `userId`.
 * @property {tf.Variable} itemFactors Dense factor matrix `[maxMovieId + 1, F]`,
 *   row index = raw `movieId`.
 * @property {number} globalMean       Mean rating of the training set.
 */

/**
 * @typedef {Object} HistoryRow
 * @property {number} epoch
 * @property {number} trainRMSE
 * @property {number} valRMSE
 * @property {number} testRMSE  `NaN` when no test split was supplied.
 */

/**
 * @typedef {Object} TrainOptions
 * @property {number} epochs    Number of full passes over the training set.
 * @property {number} lr        Learning rate.
 * @property {number} reg       L2 regularization strength.
 * @property {number} batchSize Mini-batch size.
 * @property {string} [engine]  `"tfjs"` (default) or `"js"` (pure-JS SGD).
 */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/**
 * Number of latent factors used by the default pipeline.
 *
 * WHY 16 DIMENSIONS? The factors form a low-rank approximation of the sparse
 * user x movie rating matrix. Each user and each movie becomes a 16-number
 * vector, and the predicted rating is their dot product. A small width (16 vs.
 * the 943 x 1682 raw matrix) is a capacity/overfitting trade-off: rich enough to
 * capture taste patterns (genre affinity, era, popularity) but small enough that
 * the model cannot simply memorise every training rating. It is the width the
 * assignment fixes.
 */
const NUM_FACTORS = 16;

/** Engine id: TensorFlow.js tensor path (WebGL/GPU when available). */
const ENGINE_TFJS = "tfjs";

/** Engine id: hand-written pure-JavaScript SGD path (no TF.js dependency). */
const ENGINE_JS = "js";

/** Engine selected in the UI; drives which init/train pair the pipeline calls. */
let activeEngine = ENGINE_TFJS;

/** Training split (80% of each user's ratings, oldest first). */
let trainSet = [];

/** Validation split (10% of each user's ratings). */
let valSet = [];

/** Test split (10% of each user's ratings, newest). */
let testSet = [];

/**
 * The fitted factorization, or `null` before `initializeFactors` runs.
 *
 * @type {Model|null}
 */
let model = null;

/**
 * Metric history produced by `trainMF`.
 *
 * @type {HistoryRow[]}
 */
let trainingHistory = [];

/**
 * The history entry returned by `selectBestCheckpoint`, or `null`.
 *
 * @type {HistoryRow|null}
 */
let bestCheckpoint = null;

/** Name of the TensorFlow.js backend selected by `initTf` (`"webgl"` / `"cpu"`). */
let tfBackend = null;

// ---------------------------------------------------------------------------
// TensorFlow.js bootstrap (provided)
// ---------------------------------------------------------------------------

/**
 * Select the fastest available TensorFlow.js backend and report it.
 *
 * Tries WebGL (GPU) first and falls back to the pure-JS CPU backend. Writes
 * `backend: <name>` into `#tf-status` when that element exists. If the CDN
 * failed, `tf` is undefined: this renders a readable message and returns
 * `null` instead of letting the page throw an uncaught `ReferenceError`.
 *
 * @returns {Promise<string|null>} the selected backend, or `null` when TF.js is missing.
 */
async function initTf() {
  const statusEl = document.getElementById("tf-status");
  if (typeof tf === "undefined") {
    const message =
      "TensorFlow.js failed to load. Check your network connection (the CDN script) and reload the page.";
    if (statusEl) {
      statusEl.textContent = message;
      statusEl.className = "error";
    } else if (document.body) {
      const banner = document.createElement("p");
      banner.className = "error";
      banner.textContent = message;
      document.body.insertBefore(banner, document.body.firstChild);
    }
    return null;
  }

  try {
    await tf.setBackend("webgl");
  } catch (error) {
    await tf.setBackend("cpu");
  }
  await tf.ready();

  tfBackend = tf.getBackend();
  if (statusEl) statusEl.textContent = `TensorFlow.js backend: ${tfBackend}`;
  return tfBackend;
}

/**
 * True when `candidate` is a TF.js factor model (dense `tf.Variable` rows).
 *
 * Used to route `computeRMSEFor` through the batched tensor path and to keep
 * the fixture-based tests independent of how the model was produced.
 *
 * @param {*} candidate
 * @returns {boolean}
 */
function isTfModel(candidate) {
  return (
    !!candidate &&
    !!candidate.userFactors &&
    typeof candidate.userFactors.dataSync === "function" &&
    !!candidate.itemFactors &&
    typeof candidate.itemFactors.dataSync === "function"
  );
}

// ---------------------------------------------------------------------------
// Formatting helpers (provided)
// ---------------------------------------------------------------------------

/**
 * Format a number to a fixed number of decimals. Non-finite and missing values
 * render as an em dash so the tables never show `NaN` or `undefined`.
 *
 * @param {number|null|undefined} x
 * @param {number} [d=4] decimal places
 * @returns {string}
 */
function fmt(x, d = 4) {
  if (x === null || x === undefined) return "\u2014";
  const n = Number(x);
  if (!Number.isFinite(n)) return "\u2014";
  return n.toFixed(d);
}

/**
 * Look up a movie title by id, with a readable fallback for unknown ids.
 *
 * @param {number|string} id
 * @returns {string}
 */
function movieTitle(id) {
  const numericId = Number(id);
  const movie = movies.find((m) => m.id === numericId);
  return movie ? movie.title : `Movie #${id}`;
}

/**
 * Escape text before inserting it into HTML.
 *
 * @param {string} text
 * @returns {string}
 */
function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/**
 * Clamp a number into the inclusive range `[lo, hi]`.
 *
 * @param {number} value
 * @param {number} lo
 * @param {number} hi
 * @returns {number}
 */
function clamp(value, lo, hi) {
  return Math.min(hi, Math.max(lo, value));
}

// ---------------------------------------------------------------------------
// Shared metric helpers (provided)
// ---------------------------------------------------------------------------

/**
 * Root-mean-square error of the current model over a split.
 *
 * Provided so that `trainMF` and the test harness share one implementation.
 *
 * When `model` holds TF.js variables this builds the index tensors ONCE, uses
 * `tf.gather` + element-wise multiply to score every rating in a single batch,
 * and clamps to `[1, 5]` — the same contract as `predictRating`, but fast
 * enough to call every epoch. Out-of-range ids fall back to `globalMean`, and
 * the whole computation runs inside `tf.tidy` so the intermediate tensors are
 * released automatically.
 *
 * When no TF.js model is available it falls back to the scalar
 * `predictRating` loop. That path returns `NaN` instead of throwing while
 * `predictRating` is still a `TODO(hw5)` stub, so the helper never crashes the
 * page or the harness.
 *
 * @param {Rating[]} split
 * @returns {number} `0` for an empty split; `NaN` if the scalar path throws.
 */
function computeRMSEFor(split) {
  if (!split || split.length === 0) return 0;

  if (isTfModel(model)) {
    return tf.tidy(() => {
      const n = split.length;
      const uRows = model.userFactors.shape[0];
      const iRows = model.itemFactors.shape[0];
      const userIds = new Int32Array(n);
      const movieIds = new Int32Array(n);
      const actual = new Float32Array(n);
      const valid = new Float32Array(n);

      for (let k = 0; k < n; k++) {
        const r = split[k];
        userIds[k] = r.userId;
        movieIds[k] = r.movieId;
        actual[k] = r.rating;
        const uOk = r.userId >= 0 && r.userId < uRows;
        const iOk = r.movieId >= 0 && r.movieId < iRows;
        valid[k] = uOk && iOk ? 1 : 0;
      }

      const uIdx = tf.clipByValue(tf.tensor1d(userIds, "int32"), 0, Math.max(0, uRows - 1));
      const iIdx = tf.clipByValue(tf.tensor1d(movieIds, "int32"), 0, Math.max(0, iRows - 1));
      const u = tf.gather(model.userFactors, uIdx);
      const i = tf.gather(model.itemFactors, iIdx);
      const dot = tf.sum(tf.mul(u, i), 1);
      const mask = tf.tensor1d(valid, "float32");
      const pred = tf.clipByValue(dot.mul(mask).add(model.globalMean), 1, 5);
      const diff = pred.sub(tf.tensor1d(actual, "float32"));
      return Math.sqrt(tf.mean(diff.square()).dataSync()[0]);
    });
  }

  let sum = 0;
  try {
    for (const r of split) {
      const diff = predictRating(r.userId, r.movieId) - r.rating;
      sum += diff * diff;
    }
  } catch (error) {
    return Number.NaN;
  }
  return Math.sqrt(sum / split.length);
}

/**
 * Baseline RMSE: predict the training-set mean rating for every pair and
 * measure the error on `split`. This is the number your factorization must beat.
 *
 * @param {Rating[]} split
 * @returns {number} `0` for an empty split (never divides by zero).
 */
function computeBaselineRMSE(split) {
  if (!split || split.length === 0) return 0;
  const mean = trainSet.length
    ? trainSet.reduce((acc, r) => acc + r.rating, 0) / trainSet.length
    : 0;
  let sum = 0;
  for (const r of split) {
    const diff = mean - r.rating;
    sum += diff * diff;
  }
  return Math.sqrt(sum / split.length);
}

// ---------------------------------------------------------------------------
// Student stubs (TODO(hw5))
//
// Each stub below throws until you implement it. Fill in the body, keep the
// signature, and update the JSDoc contract notes if your design differs.
// ---------------------------------------------------------------------------

/**
 * Predict the rating a user would give a movie (scalar).
 *
 * TODO(hw5): implement the prediction with TensorFlow.js. The model stores the
 * factors as two dense `tf.Variable`s:
 *   - `model.userFactors` shape `[numUsers + 1, F]`, row index = `userId`
 *   - `model.itemFactors` shape `[maxMovieId + 1, F]`, row index = `movieId`
 *
 * Suggested TF.js path (wrap it in `tf.tidy` and pull out a JS number):
 * ```js
 * const f = model.userFactors.shape[1];
 * return tf.tidy(() => {
 *   const u = model.userFactors.slice([userId, 0], [1, f]); // [1, F]
 *   const i = model.itemFactors.slice([movieId, 0], [1, f]); // [1, F]
 *   const dot = u.mul(i).sum().dataSync()[0];                // scalar
 *   return clamp(model.globalMean + dot, 1, 5);
 * });
 * ```
 * `dataSync()` copies the value to the CPU as a plain number, which survives
 * `tf.tidy` (the tensors do not). Keep this function scalar: `computeRMSEFor`
 * uses its own batched tensor path, and the harness calls it once per pair.
 *
 * Contract:
 *  - Return `model.globalMean` when the user or movie row is out of range
 *    (an id that has no allocated factor row). Never throw for an unknown id.
 *  - Return `0` when the model is not initialized (`model === null`) so callers
 *    can guard cheaply.
 *  - The result is clamped to `[1, 5]`.
 *
 * @param {number} userId
 * @param {number} movieId
 * @returns {number} predicted rating in `[1, 5]`
 */
function predictRating(userId, movieId) {
  if (!model) return 0;

  // Pure-JS model: plain typed arrays, a straightforward dot product.
  if (model.kind === ENGINE_JS) {
    if (
      userId < 0 || userId >= model.numUserRows ||
      movieId < 0 || movieId >= model.numItemRows
    ) {
      return clamp(model.globalMean, 1, 5);
    }
    const F = model.numFactors;
    const pu = userId * F;
    const qi = movieId * F;
    let dot = 0;
    for (let f = 0; f < F; f++) {
      dot += model.userFactors[pu + f] * model.itemFactors[qi + f];
    }
    return clamp(model.globalMean + dot, 1, 5);
  }

  if (!isTfModel(model)) return 0;

  const F = model.userFactors.shape[1];
  const uRows = model.userFactors.shape[0];
  const iRows = model.itemFactors.shape[0];
  if (
    userId < 0 || userId >= uRows ||
    movieId < 0 || movieId >= iRows
  ) {
    return clamp(model.globalMean, 1, 5);
  }

  return tf.tidy(() => {
    const u = model.userFactors.slice([userId, 0], [1, F]); // [1, F]
    const i = model.itemFactors.slice([movieId, 0], [1, F]); // [1, F]
    const dot = u.mul(i).sum().dataSync()[0]; // scalar on the CPU
    return clamp(model.globalMean + dot, 1, 5);
  });
}

/**
 * Allocate the latent factor variables and the global mean.
 *
 * TODO(hw5): compute `globalMean` as the mean rating over `train`
 * (`0` for an empty training set), then create the two dense variables:
 * ```js
 * const maxMovieId = movies.reduce((m, x) => Math.max(m, x.id), 0);
 * model = {
 *   userFactors: tf.variable(tf.randomNormal([numUsers + 1, numFactors], 0, 0.05)),
 *   itemFactors: tf.variable(tf.randomNormal([maxMovieId + 1, numFactors], 0, 0.05)),
 *   globalMean,
 * };
 * ```
 * `tf.randomNormal(shape, mean, std)` draws small values — use a small std
 * (e.g. `0.05`) so training starts near the global mean. `tf.variable` wraps a
 * tensor so the optimizer can update it in place. Store the result on the
 * module-level `model` object and return it.
 *
 * Contract:
 *  - `numFactors` defaults to the module-level `NUM_FACTORS`.
 *  - If a previous `model` exists, dispose its variables (`tf.dispose`) before
 *    replacing it so repeated "Train" clicks do not leak GPU memory.
 *  - Return the `model` object.
 *
 * @param {Rating[]} train
 * @param {number} [numFactors=NUM_FACTORS]
 * @returns {Model}
 */
function initializeFactors(train, numFactors = NUM_FACTORS) {
  // Free any previous TF.js variables so repeated "Train" clicks do not leak
  // GPU memory (tf.Variables live outside the JS heap).
  if (model && isTfModel(model)) {
    model.userFactors.dispose();
    model.itemFactors.dispose();
  }
  model = null;

  const globalMean =
    train && train.length
      ? train.reduce((acc, r) => acc + r.rating, 0) / train.length
      : 0;

  const maxMovieId = movies.reduce((m, x) => Math.max(m, x.id), 0);

  // Dense rows indexed by RAW ids: row 0 is unused padding, row `userId` /
  // `movieId` holds that id's 16 factors. Small random init (std 0.05) starts
  // every prediction close to the global mean.
  model = {
    userFactors: tf.variable(tf.randomNormal([numUsers + 1, numFactors], 0, 0.05)),
    itemFactors: tf.variable(tf.randomNormal([maxMovieId + 1, numFactors], 0, 0.05)),
    globalMean,
  };
  return model;
}

/**
 * Pure-JS twin of `initializeFactors`: same dense, raw-id-indexed layout but the
 * factors are flat `Float32Array`s (row `id` occupies `id*F .. id*F+F-1`), so the
 * SGD engine needs no TensorFlow.js at all.
 *
 * @param {Rating[]} train
 * @param {number} [numFactors=NUM_FACTORS]
 * @returns {Model}
 */
function initializeFactorsJS(train, numFactors = NUM_FACTORS) {
  if (model && isTfModel(model)) {
    model.userFactors.dispose();
    model.itemFactors.dispose();
  }
  model = null;

  const globalMean =
    train && train.length
      ? train.reduce((acc, r) => acc + r.rating, 0) / train.length
      : 0;

  const maxMovieId = movies.reduce((m, x) => Math.max(m, x.id), 0);
  const numUserRows = numUsers + 1;
  const numItemRows = maxMovieId + 1;

  const userFactors = new Float32Array(numUserRows * numFactors);
  const itemFactors = new Float32Array(numItemRows * numFactors);
  for (let i = 0; i < userFactors.length; i++) {
    userFactors[i] = (Math.random() * 2 - 1) * 0.05;
  }
  for (let i = 0; i < itemFactors.length; i++) {
    itemFactors[i] = (Math.random() * 2 - 1) * 0.05;
  }

  model = {
    kind: ENGINE_JS,
    userFactors,
    itemFactors,
    numFactors,
    numUserRows,
    numItemRows,
    globalMean,
  };
  return model;
}

/**
 * Train the factorization with batched gradient descent.
 *
 * TODO(hw5): run a mini-batch loop over `train` using the TF.js optimizer.
 * Score each batch with tensor ops (never one `tf` call per rating) and let the
 * optimizer update both variables in place:
 * ```js
 * const optimizer = tf.train.adam(lr);
 * for (let epoch = 1; epoch <= epochs; epoch++) {
 *   for (const batch of batches) {
 *     optimizer.minimize(() => tf.tidy(() => {
 *       const u = tf.gather(model.userFactors, batch.userIdx);
 *       const i = tf.gather(model.itemFactors, batch.itemIdx);
 *       const pred = tf.sum(u.mul(i), 1).add(model.globalMean);
 *       const mse = tf.losses.meanSquaredError(batch.targets, pred);
 *       const l2 = model.userFactors.square().sum()
 *         .add(model.itemFactors.square().sum()).mul(reg);
 *       return mse.add(l2);
 *     }), false, [model.userFactors, model.itemFactors]).dispose();
 *   }
 *   trainingHistory.push({
 *     epoch,
 *     trainRMSE: computeRMSEFor(train),
 *     valRMSE: computeRMSEFor(val),
 *   });
 * }
 * return trainingHistory;
 * ```
 * Key TF.js APIs to use: `tf.train.adam(lr)` (or `tf.train.sgd`),
 * `optimizer.minimize(fn, returnCost, varList)`, `tf.gather`,
 * `tf.losses.meanSquaredError`, and `tf.tidy` for per-batch intermediates.
 * `minimize` returns a scalar loss tensor — dispose it (or use `tf.tidy`) so
 * the loop does not accumulate tensors.
 *
 * Contract:
 *  - `opts = { epochs, lr, reg, batchSize }`. Use mini-batches of `batchSize`
 *    (or full-batch when `batchSize >= train.length`).
 *  - Update `model` in place. Do NOT reallocate it.
 *  - Append one `HistoryRow` per epoch to the module-level `trainingHistory`
 *    and return it: `{ epoch, trainRMSE: computeRMSEFor(train),
 *    valRMSE: computeRMSEFor(val) }`. `epoch` is 1-based.
 *  - Shuffling the batch order each epoch is allowed and usually helps.
 *  - Guard `train.length === 0` by returning an empty history.
 *
 * @param {Rating[]} train
 * @param {Rating[]} val
 * @param {TrainOptions} opts
 * @returns {HistoryRow[]}
 */
function trainMF(train, val, opts, test = []) {
  trainingHistory = [];
  if (!train || train.length === 0 || !isTfModel(model)) return trainingHistory;

  const epochs = Math.max(1, Math.round(opts.epochs));
  const lr = opts.lr;
  const reg = opts.reg;
  const batchSize = Math.max(1, Math.round(opts.batchSize));

  const optimizer = tf.train.adam(lr);

  const n = train.length;
  const uAll = new Int32Array(n);
  const iAll = new Int32Array(n);
  const tAll = new Float32Array(n);
  for (let k = 0; k < n; k++) {
    uAll[k] = train[k].userId;
    iAll[k] = train[k].movieId;
    tAll[k] = train[k].rating;
  }
  const order = new Int32Array(n);
  for (let k = 0; k < n; k++) order[k] = k;

  for (let epoch = 1; epoch <= epochs; epoch++) {
    // Shuffle the visit order each epoch (helps SGD escape valleys).
    for (let k = n - 1; k > 0; k--) {
      const j = Math.floor(Math.random() * (k + 1));
      const tmp = order[k];
      order[k] = order[j];
      order[j] = tmp;
    }

    const bsz = batchSize >= n ? n : batchSize;
    for (let start = 0; start < n; start += bsz) {
      const end = Math.min(start + bsz, n);
      const count = end - start;
      const uIdxArr = new Int32Array(count);
      const iIdxArr = new Int32Array(count);
      const tArr = new Float32Array(count);
      for (let b = 0; b < count; b++) {
        const src = order[start + b];
        uIdxArr[b] = uAll[src];
        iIdxArr[b] = iAll[src];
        tArr[b] = tAll[src];
      }

      const uIdx = tf.tensor1d(uIdxArr, "int32");
      const iIdx = tf.tensor1d(iIdxArr, "int32");
      const targets = tf.tensor1d(tArr, "float32");

      // One tensor op per mini-batch; the optimizer updates both tf.Variables.
      // WHY L2 REGULARIZATION? The unregularized loss can drive factor norms up
      // to fit the sparse training rows, which fits noise and worsens val/test
      // RMSE. The `reg` penalty on the squared factor norms shrinks the vectors,
      // trading a little training fit for generalisation.
      optimizer.minimize(
        () =>
          tf.tidy(() => {
            const u = tf.gather(model.userFactors, uIdx);
            const i = tf.gather(model.itemFactors, iIdx);
            const pred = tf.sum(u.mul(i), 1).add(model.globalMean);
            const mse = tf.losses.meanSquaredError(targets, pred);
            const l2 = model.userFactors
              .square()
              .sum()
              .add(model.itemFactors.square().sum())
              .mul(reg);
            return mse.add(l2);
          }),
        false,
        [model.userFactors, model.itemFactors],
      );

      uIdx.dispose();
      iIdx.dispose();
      targets.dispose();
    }

    trainingHistory.push({
      epoch,
      trainRMSE: computeRMSEFor(train),
      valRMSE: computeRMSEFor(val),
      testRMSE: test && test.length ? computeRMSEFor(test) : Number.NaN,
    });
  }

  return trainingHistory;
}

/**
 * Pure-JS mini-batch SGD twin of `trainMF`.
 *
 * For each mini-batch it accumulates the squared-error gradient plus the L2
 * penalty term (WHY L2: see the note in `trainMF`), averages over the batch, and
 * applies one SGD step to the user and movie factor rows. Both gradients are
 * accumulated BEFORE any update so they use the same pre-step factors.
 *
 * @param {Rating[]} train
 * @param {Rating[]} val
 * @param {TrainOptions} opts
 * @param {Rating[]} [test]
 * @returns {HistoryRow[]}
 */
function trainMFJS(train, val, opts, test = []) {
  trainingHistory = [];
  if (!train || train.length === 0 || !model || model.kind !== ENGINE_JS) {
    return trainingHistory;
  }

  const epochs = Math.max(1, Math.round(opts.epochs));
  const lr = opts.lr;
  const reg = opts.reg;
  const batchSize = Math.max(1, Math.round(opts.batchSize));

  const F = model.numFactors;
  const P = model.userFactors;
  const Q = model.itemFactors;
  const mu = model.globalMean;
  const n = train.length;

  const order = new Int32Array(n);
  for (let k = 0; k < n; k++) order[k] = k;

  const applyBatch = (batch) => {
    const gP = new Map();
    const gQ = new Map();
    const gradFor = (map, id) => {
      let g = map.get(id);
      if (!g) {
        g = new Float32Array(F);
        map.set(id, g);
      }
      return g;
    };

    for (const r of batch) {
      const u = r.userId;
      const i = r.movieId;
      if (u < 0 || u >= model.numUserRows || i < 0 || i >= model.numItemRows) {
        continue;
      }
      const pu = u * F;
      const qi = i * F;
      let dot = 0;
      for (let f = 0; f < F; f++) dot += P[pu + f] * Q[qi + f];
      const err = mu + dot - r.rating; // residual r_hat - r
      const gu = gradFor(gP, u);
      const gi = gradFor(gQ, i);
      for (let f = 0; f < F; f++) {
        gu[f] += err * Q[qi + f] + reg * P[pu + f];
        gi[f] += err * P[pu + f] + reg * Q[qi + f];
      }
    }

    // Apply the SUMMED batch gradient (not the batch mean): each rating already
    // contributed its own error and L2 term, so a batch of B behaves like B
    // sequential SGD steps with the same learning rate.
    for (const [u, g] of gP) {
      const pu = u * F;
      for (let f = 0; f < F; f++) P[pu + f] -= lr * g[f];
    }
    for (const [i, g] of gQ) {
      const qi = i * F;
      for (let f = 0; f < F; f++) Q[qi + f] -= lr * g[f];
    }
  };

  for (let epoch = 1; epoch <= epochs; epoch++) {
    for (let k = n - 1; k > 0; k--) {
      const j = Math.floor(Math.random() * (k + 1));
      const tmp = order[k];
      order[k] = order[j];
      order[j] = tmp;
    }

    const bsz = batchSize >= n ? n : batchSize;
    for (let start = 0; start < n; start += bsz) {
      const end = Math.min(start + bsz, n);
      const batch = [];
      for (let s = start; s < end; s++) batch.push(train[order[s]]);
      applyBatch(batch);
    }

    trainingHistory.push({
      epoch,
      trainRMSE: computeRMSEFor(train),
      valRMSE: computeRMSEFor(val),
      testRMSE: test && test.length ? computeRMSEFor(test) : Number.NaN,
    });
  }

  return trainingHistory;
}

/**
 * Pick the best epoch from a training history.
 *
 * TODO(hw5): return the entry with the smallest `valRMSE`. On a tie return the
 * entry with the EARLIEST `epoch` (do not simply take the last minimum). This
 * one is plain JavaScript — no TF.js needed.
 *
 * Contract:
 *  - Return `null` for an empty or missing history.
 *  - Do not mutate the input array.
 *
 * @param {HistoryRow[]} history
 * @returns {HistoryRow|null}
 */
function selectBestCheckpoint(history) {
  if (!history || history.length === 0) return null;
  let best = history[0];
  for (let i = 1; i < history.length; i++) {
    // Strict `<` keeps the EARLIEST epoch on a tie (do not take the last min).
    if (history[i].valRMSE < best.valRMSE) best = history[i];
  }
  return best;
}

/**
 * Recommend the top-K movies for a user.
 *
 * TODO(hw5): score every candidate movie and return the K highest-scoring
 * movies the user has NOT already rated in `trainSet` or `valSet`. Test-set
 * ratings are allowed as candidates. **Prefer the tensor path**:
 * ```js
 * const userRow = model.userFactors.slice([userId, 0], [1, F]); // [1, F]
 * const scores = model.itemFactors.matMul(userRow, false, true)  // [rows, 1]
 *   .squeeze().add(model.globalMean);                            // [rows]
 * // mask excluded ids to a large negative value, then:
 * const { values, indices } = tf.topk(maskedScores, Math.min(K, rows));
 * ```
 * Because `itemFactors` is indexed by raw `movieId`, the `tf.topk` indices ARE
 * the movie ids — no reverse mapping needed. Excluded rows must be masked (for
 * example multiply by a 0/1 mask and add `(1 - mask) * -1e9`); re-apply the
 * same mask when mapping results back so masked ids are dropped.
 *
 * Contract:
 *  - Return `[{ movieId, title, score }, ...]` sorted by descending `score`,
 *    truncated to `K` (fewer if not enough candidates).
 *  - Exclude any movie id that is absent from the global `movies` list, and any
 *    id the user rated in `trainSet` or `valSet`.
 *  - Return `[]` when the model is not initialized, when `K <= 0`, or when the
 *    user has no eligible candidates — never throw.
 *  - `title` comes from `movieTitle(movieId)`.
 *
 * @param {number} userId
 * @param {number} [K=10]
 * @returns {Array<{movieId: number, title: string, score: number}>}
 */
function recommendTopK(userId, K = 10) {
  if (!model || K <= 0) return [];

  if (model.kind === ENGINE_JS) return recommendTopKJS(userId, K);
  if (!isTfModel(model)) return [];
  if (userId < 0 || userId >= model.userFactors.shape[0]) return [];

  const F = model.userFactors.shape[1];
  const rows = model.itemFactors.shape[0];

  // Ids to hide: everything the user rated in train or validation. Test-rated
  // items stay eligible (they are the "future" we are trying to predict).
  const excluded = new Set();
  for (const r of trainSet) if (r.userId === userId) excluded.add(r.movieId);
  for (const r of valSet) if (r.userId === userId) excluded.add(r.movieId);

  // Keep only ids that exist in the parsed `movies` list and are not excluded.
  const movieIds = new Set(movies.map((m) => m.id));
  const maskArr = new Float32Array(rows);
  for (let id = 0; id < rows; id++) {
    maskArr[id] = movieIds.has(id) && !excluded.has(id) ? 1 : 0;
  }

  return tf.tidy(() => {
    const userRow = model.userFactors.slice([userId, 0], [1, F]); // [1, F]
    const scores = model.itemFactors
      .matMul(userRow, false, true) // [rows, 1]
      .squeeze() // [rows]
      .add(model.globalMean); // predicted rating per movie id

    const mask = tf.tensor1d(maskArr);
    // Masked rows get a huge negative score so they never enter the top-K.
    const masked = scores.mul(mask).add(mask.neg().add(1).mul(-1e9));

    const k = Math.min(K, rows);
    const { values, indices } = tf.topk(masked, k);
    const vals = values.dataSync();
    const idx = indices.dataSync();

    const out = [];
    for (let t = 0; t < idx.length && out.length < K; t++) {
      const id = idx[t];
      if (!maskArr[id]) continue; // drop masked / unknown / already-rated ids
      out.push({ movieId: id, title: movieTitle(id), score: vals[t] });
    }
    return out;
  });
}

/**
 * Pure-JS twin of `recommendTopK`: score every parsed movie with a dot product
 * and sort in JS. Same exclusion rule (train or validation ratings hidden).
 *
 * @param {number} userId
 * @param {number} K
 * @returns {Array<{movieId: number, title: string, score: number}>}
 */
function recommendTopKJS(userId, K) {
  if (userId < 0 || userId >= model.numUserRows) return [];
  const F = model.numFactors;
  const pu = userId * F;

  const excluded = new Set();
  for (const r of trainSet) if (r.userId === userId) excluded.add(r.movieId);
  for (const r of valSet) if (r.userId === userId) excluded.add(r.movieId);

  const scored = [];
  for (const m of movies) {
    if (excluded.has(m.id)) continue;
    if (m.id < 0 || m.id >= model.numItemRows) continue;
    const qi = m.id * F;
    let dot = 0;
    for (let f = 0; f < F; f++) {
      dot += model.userFactors[pu + f] * model.itemFactors[qi + f];
    }
    scored.push({ movieId: m.id, title: m.title, score: model.globalMean + dot });
  }

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, K);
}

/**
 * Compute Recall@K for the current model.
 *
 * TODO(hw5): retrieval evaluation built on `recommendTopK`.
 *   - relevant = test ratings with `rating >= 4.0`.
 *   - Candidate items = the set of movie ids that appear anywhere in `trainSet`
 *     (a global candidate pool, NOT just the user's own training items).
 *   - For every user with at least one relevant test movie among the
 *     candidates, take `recommendTopK(userId, K)` and count how many of that
 *     user's relevant candidate movies appear in it.
 *
 * Contract:
 *  - Return
 *    `{ overallRecall, perUser, excluded }` where:
 *      `overallRecall` = mean over reported users of `hit / relevant`
 *        (`0` when there are no reported users — never divide by zero);
 *      `perUser` = `[{ userId, relevant, hit }, ...]` (reported users only);
 *      `excluded` = `{ unseenIds, usersWithoutRelevant }`.
 *  - `unseenIds` = distinct relevant test movie ids that are NOT present in
 *    `trainSet` (they can never be recommended from a training-only pool).
 *  - `usersWithoutRelevant` = distinct users that appear in `testSet` but have
 *    no relevant test movie among the candidates.
 *  - Report exclusions explicitly so the recall number is interpretable.
 *
 * @param {number} [K=10]
 * @returns {{overallRecall: number, perUser: Array<{userId: number, relevant: number, hit: number}>, excluded: {unseenIds: number[], usersWithoutRelevant: number[]}}}
 */
function recallAtK(K = 10) {
  // WHAT IS Recall@K? For each user we ask: of the movies they will actually
  // like in the future (relevant test items, rating >= 4.0), how many appear in
  // the Top-K list we would show them? recall_u = hits / relevant. The reported
  // number is the mean over evaluable users. It measures retrieval quality, not
  // rating accuracy (that is what RMSE measures).
  const relevant = (r) => r.rating >= 4.0;

  // Candidate pool = every movie that appears anywhere in the training split.
  // A training-only recommender has no factor signal for anything else.
  const candidateMovies = new Set();
  for (const r of trainSet) candidateMovies.add(r.movieId);

  // Group each user's relevant test movies.
  const relevantByUser = new Map();
  for (const r of testSet) {
    if (!relevant(r)) continue;
    if (!relevantByUser.has(r.userId)) relevantByUser.set(r.userId, new Set());
    relevantByUser.get(r.userId).add(r.movieId);
  }

  // Exclusions: relevant test movies missing from the candidate pool can never
  // be recommended from a training-only pool, so they are reported separately.
  const unseenIds = new Set();
  for (const ids of relevantByUser.values()) {
    for (const id of ids) if (!candidateMovies.has(id)) unseenIds.add(id);
  }

  const perUser = [];
  const usersWithoutRelevant = new Set();

  for (const [userId, ids] of relevantByUser) {
    const candidateRelevant = new Set();
    for (const id of ids) if (candidateMovies.has(id)) candidateRelevant.add(id);

    if (candidateRelevant.size === 0) {
      usersWithoutRelevant.add(userId);
      continue; // no eligible relevant movie -> user is not evaluable
    }

    const recs = recommendTopK(userId, K);
    const recIds = new Set(recs.map((r) => r.movieId));
    let hit = 0;
    for (const id of candidateRelevant) if (recIds.has(id)) hit++;
    perUser.push({ userId, relevant: candidateRelevant.size, hit });
  }

  // Users who appear in the test split but have no relevant test movie at all.
  const testUsers = new Set(testSet.map((r) => r.userId));
  for (const u of testUsers) {
    if (!relevantByUser.has(u)) usersWithoutRelevant.add(u);
  }

  const overallRecall = perUser.length
    ? perUser.reduce((acc, x) => acc + x.hit / x.relevant, 0) / perUser.length
    : 0;

  return {
    overallRecall,
    perUser,
    excluded: {
      unseenIds: Array.from(unseenIds).sort((a, b) => a - b),
      usersWithoutRelevant: Array.from(usersWithoutRelevant).sort((a, b) => a - b),
    },
  };
}

// ---------------------------------------------------------------------------
// UI rendering (provided)
// ---------------------------------------------------------------------------

/**
 * Render the dataset shape summary.
 *
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderDatasetSummary(container) {
  const target = container || document.getElementById("dataset-summary-body");
  if (!target) return;
  const maxMovieId = movies.reduce((max, m) => Math.max(max, m.id), 0);
  const cells = ratingStats.cells || numUsers * movies.length;
  const observed = ratingStats.observed || ratings.length;
  const sparsity = cells > 0 ? 1 - observed / cells : 0;
  target.innerHTML = `
    <dl class="summary-list">
      <dt>Movies</dt><dd>${movies.length.toLocaleString("en-US")}</dd>
      <dt>Ratings</dt><dd>${ratings.length.toLocaleString("en-US")}</dd>
      <dt>Users</dt><dd>${numUsers.toLocaleString("en-US")}</dd>
      <dt>Rating matrix</dt><dd>${(numUsers + 1).toLocaleString("en-US")} &times; ${(maxMovieId + 1).toLocaleString("en-US")}</dd>
      <dt>Observed cells</dt><dd>${observed.toLocaleString("en-US")} of ${cells.toLocaleString("en-US")}</dd>
      <dt>Sparsity</dt><dd>${fmt(sparsity * 100, 2)}%</dd>
    </dl>
    <p class="hint">Sparsity is the share of user&ndash;movie cells with no observed rating. Missing ratings are treated as unobserved, never as zeros.</p>`;
}

/**
 * Render the chronological 80/10/10 split summary.
 *
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderSplitSummary(container) {
  const target = container || document.getElementById("split-summary-body");
  if (!target) return;
  const total = trainSet.length + valSet.length + testSet.length;
  const pct = (n) => (total ? ((n / total) * 100).toFixed(1) : "0.0");
  const exclusions = typeof SPLIT_EXCLUSIONS === "undefined" ? [] : SPLIT_EXCLUSIONS;
  const pool =
    typeof TRAIN_POOL_EXCLUSIONS === "undefined"
      ? { moviesNotInTrain: [], usersNotInTrain: [] }
      : TRAIN_POOL_EXCLUSIONS;
  target.innerHTML = `
    <table class="data-table">
      <thead><tr><th scope="col">Split</th><th scope="col">Ratings</th><th scope="col">Share</th></tr></thead>
      <tbody>
        <tr><td>Train</td><td class="num">${trainSet.length}</td><td class="num">${pct(trainSet.length)}%</td></tr>
        <tr><td>Validation</td><td class="num">${valSet.length}</td><td class="num">${pct(valSet.length)}%</td></tr>
        <tr><td>Test</td><td class="num">${testSet.length}</td><td class="num">${pct(testSet.length)}%</td></tr>
      </tbody>
    </table>
    <p class="hint">Each user's ratings are sorted by timestamp and split 80/10/10.</p>
    <p class="hint">${exclusions.length} user(s) skipped while splitting (fewer than 5 ratings); ${pool.moviesNotInTrain.length} movie(s) and ${pool.usersNotInTrain.length} user(s) never appear in train and are excluded from the recommendation candidate pool.</p>
    <p class="hint">Baseline (train-mean) test RMSE ${fmt(computeBaselineRMSE(testSet))}.</p>`;
}

/**
 * Render the training history as a plain table (no SVG, no chart library).
 *
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderTrainingCurve(container) {
  const target = container || document.getElementById("training-curve");
  if (!target) return;
  if (!trainingHistory || trainingHistory.length === 0) {
    target.innerHTML =
      '<p class="empty-state">No training history yet. Implement the <code>TODO(hw5)</code> functions and press &ldquo;Train&rdquo;.</p>';
    return;
  }
  let bestIndex = 0;
  for (let i = 1; i < trainingHistory.length; i++) {
    if (trainingHistory[i].valRMSE < trainingHistory[bestIndex].valRMSE) bestIndex = i;
  }
  const rows = trainingHistory
    .map(
      (row, i) => `
      <tr${i === bestIndex ? ' class="best-epoch"' : ""}>
        <td class="num">${row.epoch}</td>
        <td class="num">${fmt(row.trainRMSE)}</td>
        <td class="num">${fmt(row.valRMSE)}</td>
        <td class="num">${fmt(row.testRMSE)}</td>
      </tr>`,
    )
    .join("");
  target.innerHTML = `
    ${trainingChartSvg(trainingHistory)}
    <p class="hint">Learning curves (higher RMSE = worse): the three lines track train, validation and test
      error per epoch. The highlighted table row is the checkpoint kept by the lowest validation RMSE.
      If validation/test RMSE starts rising while train RMSE keeps falling, the model is overfitting.</p>
    <table class="data-table">
      <thead><tr><th scope="col">Epoch</th><th scope="col">Train RMSE</th><th scope="col">Val RMSE</th><th scope="col">Test RMSE</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

/**
 * Build a small, dependency-free SVG line chart of the train/validation/test
 * RMSE curves. Colors are set via inline attributes so no stylesheet change is
 * needed (the provided `style.css` stays untouched).
 *
 * @param {HistoryRow[]} history
 * @returns {string} an `<svg>` markup string
 */
function trainingChartSvg(history) {
  const W = 680;
  const H = 260;
  const padL = 52;
  const padR = 16;
  const padT = 16;
  const padB = 34;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const series = [
    { key: "trainRMSE", label: "Train", color: "#0f6e6e" },
    { key: "valRMSE", label: "Validation", color: "#b4530a" },
    { key: "testRMSE", label: "Test", color: "#3557a0" },
  ];

  let lo = Infinity;
  let hi = -Infinity;
  for (const row of history) {
    for (const s of series) {
      const v = row[s.key];
      if (Number.isFinite(v)) {
        lo = Math.min(lo, v);
        hi = Math.max(hi, v);
      }
    }
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
    lo = 0;
    hi = 1;
  }
  if (hi - lo < 1e-6) hi = lo + 1;
  const pad = (hi - lo) * 0.08;
  lo -= pad;
  hi += pad;

  const n = history.length;
  const xAt = (i) => padL + (n === 1 ? plotW / 2 : (i / (n - 1)) * plotW);
  const yAt = (v) => padT + (1 - (v - lo) / (hi - lo)) * plotH;

  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Training, validation and test RMSE per epoch" style="width:100%;height:auto;max-width:${W}px;display:block;font-family:system-ui,sans-serif">`;

  // Y gridlines + tick labels (5 steps).
  const ticks = 4;
  for (let t = 0; t <= ticks; t++) {
    const v = lo + ((hi - lo) * t) / ticks;
    const y = yAt(v);
    svg += `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${padL + plotW}" y2="${y.toFixed(1)}" stroke="#e2e6ec" stroke-width="1"/>`;
    svg += `<text x="${padL - 6}" y="${(y + 4).toFixed(1)}" text-anchor="end" font-size="11" fill="#5b6472">${v.toFixed(2)}</text>`;
  }
  // Axes.
  svg += `<line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + plotH}" stroke="#9aa3b0" stroke-width="1"/>`;
  svg += `<line x1="${padL}" y1="${padT + plotH}" x2="${padL + plotW}" y2="${padT + plotH}" stroke="#9aa3b0" stroke-width="1"/>`;

  // X labels: first, last and a few in between.
  const labelEvery = Math.max(1, Math.ceil(n / 8));
  for (let i = 0; i < n; i++) {
    if (i !== 0 && i !== n - 1 && i % labelEvery !== 0) continue;
    svg += `<text x="${xAt(i).toFixed(1)}" y="${padT + plotH + 16}" text-anchor="middle" font-size="11" fill="#5b6472">${history[i].epoch}</text>`;
  }

  // One polyline per series, skipping non-finite points.
  for (const s of series) {
    const pts = [];
    for (let i = 0; i < n; i++) {
      const v = history[i][s.key];
      if (!Number.isFinite(v)) continue;
      pts.push(`${xAt(i).toFixed(1)},${yAt(v).toFixed(1)}`);
    }
    if (pts.length === 0) continue;
    svg += `<polyline fill="none" stroke="${s.color}" stroke-width="2" points="${pts.join(" ")}"/>`;
    if (pts.length === 1) {
      const [cx, cy] = pts[0].split(",");
      svg += `<circle cx="${cx}" cy="${cy}" r="3" fill="${s.color}"/>`;
    }
  }

  // Legend.
  let lx = padL;
  for (const s of series) {
    svg += `<line x1="${lx}" y1="12" x2="${lx + 18}" y2="12" stroke="${s.color}" stroke-width="3"/>`;
    svg += `<text x="${lx + 23}" y="16" font-size="12" fill="#1c2128">${s.label}</text>`;
    lx += 30 + s.label.length * 7;
  }

  svg += "</svg>";
  return svg;
}

/**
 * Render the top-K recommendations for a user.
 *
 * @param {number} userId
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderTopK(userId, container) {
  const target = container || document.getElementById("recommendations");
  if (!target) return;
  if (!model) {
    target.innerHTML = '<p class="empty-state">Train the model before requesting recommendations.</p>';
    return;
  }
  try {
    const recs = recommendTopK(Number(userId), 10);
    if (!recs || recs.length === 0) {
      target.innerHTML = '<p class="empty-state">No eligible recommendations for this user.</p>';
      return;
    }
    target.innerHTML =
      '<ol class="recommendation-list">' +
      recs
        .map(
          (r) =>
            `<li><span class="rec-title">${escapeHtml(r.title)}</span> <span class="rec-score">${fmt(r.score)}</span></li>`,
        )
        .join("") +
      "</ol>";
  } catch (error) {
    target.innerHTML = `<p class="empty-state">Recommendations unavailable — implement the <code>TODO(hw5)</code> functions. <code>${escapeHtml(error.message)}</code></p>`;
  }
}

/**
 * Render the Recall@K report, including explicit exclusions.
 *
 * @param {number} [K=10]
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderRecall(K = 10, container) {
  const target = container || document.getElementById("recall-results");
  if (!target) return;
  try {
    const result = recallAtK(K);
    const rows = result.perUser
      .map(
        (u) => `
        <tr>
          <td>${u.userId}</td>
          <td class="num">${u.relevant}</td>
          <td class="num">${u.hit}</td>
          <td class="num">${u.relevant ? fmt(u.hit / u.relevant) : "\u2014"}</td>
        </tr>`,
      )
      .join("");
    target.innerHTML = `
      <p><strong>Recall@${K}:</strong> <span class="num">${fmt(result.overallRecall)}</span></p>
      <table class="data-table">
        <thead><tr><th scope="col">User</th><th scope="col">Relevant</th><th scope="col">Hits</th><th scope="col">Recall</th></tr></thead>
        <tbody>${rows || '<tr><td colspan="4">No evaluable users.</td></tr>'}</tbody>
      </table>
      <p class="hint">Excluded: ${result.excluded.unseenIds.length} unseen relevant test item(s); ${result.excluded.usersWithoutRelevant.length} user(s) without a relevant candidate.</p>`;
  } catch (error) {
    target.innerHTML = `<p class="empty-state">Recall unavailable — implement the <code>TODO(hw5)</code> functions. <code>${escapeHtml(error.message)}</code></p>`;
  }
}

/**
 * Render the headline metrics: baseline vs. matrix-factorization test RMSE and
 * Recall@10.
 *
 * RMSE (root-mean-square error) is the square root of the average squared
 * difference between predicted and actual ratings; lower is better and it is in
 * the same units as a star rating. The baseline predicts the training mean for
 * every pair, so it is the number the factorization must beat. Recall@10 is a
 * retrieval metric (share of a user's relevant test movies that make the Top-10).
 *
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderMetrics(container) {
  const target = container || document.getElementById("metrics-body");
  if (!target) return;

  const baseline = computeBaselineRMSE(testSet);
  const mf =
    bestCheckpoint && Number.isFinite(bestCheckpoint.testRMSE)
      ? bestCheckpoint.testRMSE
      : null;
  const improvement =
    mf !== null && baseline > 0 ? ((baseline - mf) / baseline) * 100 : null;

  let recall = null;
  if (model) {
    try {
      recall = recallAtK(10).overallRecall;
    } catch (error) {
      recall = null;
    }
  }

  if (!model) {
    target.innerHTML =
      '<p class="empty-state">Train the model to compute the baseline and matrix-factorization RMSE.</p>';
    return;
  }

  target.innerHTML = `
    <table class="data-table">
      <thead><tr><th scope="col">Metric</th><th scope="col">Value</th></tr></thead>
      <tbody>
        <tr><td>Baseline test RMSE (train mean)</td><td class="num">${fmt(baseline)}</td></tr>
        <tr><td>Matrix-factorization test RMSE (best checkpoint)</td><td class="num">${mf === null ? "\u2014" : fmt(mf)}</td></tr>
        <tr><td>Improvement over baseline</td><td class="num">${improvement === null ? "\u2014" : fmt(improvement, 2) + "%"}</td></tr>
        <tr><td>Best epoch (by validation RMSE)</td><td class="num">${bestCheckpoint ? bestCheckpoint.epoch : "\u2014"}</td></tr>
        <tr><td>Recall@10</td><td class="num">${recall === null ? "\u2014" : fmt(recall)}</td></tr>
      </tbody>
    </table>`;
}

/**
 * Look up a movie record (id, title, genres) by id.
 *
 * @param {number} id
 * @returns {Movie|undefined}
 */
function movieById(id) {
  return movies.find((m) => m.id === id);
}

/**
 * Render the analytics block: for two test users, compare their Top-10 with the
 * genres/titles they already rated, to judge whether the recommender helps them
 * DISCOVER movies or mostly REPEATS their known taste.
 *
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderAnalytics(container) {
  const target = container || document.getElementById("analytics-body");
  if (!target) return;

  if (!model || !bestCheckpoint) {
    target.innerHTML =
      '<p class="empty-state">Train the model to compare each user\'s Top-10 with their past ratings.</p>';
    return;
  }

  // Two test users that also have training history (so a Top-10 is meaningful).
  const testUsers = Array.from(new Set(testSet.map((r) => r.userId)));
  const withHistory = testUsers.filter((u) => trainSet.some((r) => r.userId === u));
  const chosen = withHistory.sort((a, b) => a - b).slice(0, 2);

  if (chosen.length === 0) {
    target.innerHTML = '<p class="empty-state">No test users with training history.</p>';
    return;
  }

  const sections = chosen.map((userId) => {
    const recs = recommendTopK(userId, 10);
    const recIds = new Set(recs.map((r) => r.movieId));

    // Past preferences: movies the user liked (rating >= 4) in train or val.
    const liked = [...trainSet, ...valSet].filter(
      (r) => r.userId === userId && r.rating >= 4.0,
    );
    const likedGenres = new Map();
    for (const r of liked) {
      const m = movieById(r.movieId);
      if (!m) continue;
      for (const g of m.genres) likedGenres.set(g, (likedGenres.get(g) || 0) + 1);
    }
    const topGenres = Array.from(likedGenres.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([g, n]) => `${escapeHtml(g)} (${n})`)
      .join(", ") || "no liked movies with genres";

    // Overlap of the Top-10 with known taste: hits a liked genre (repeat) or not (discover).
    let repeats = 0;
    let discoveries = 0;
    const recItems = recs.map((r) => {
      const m = movieById(r.movieId);
      const genres = m ? m.genres : [];
      const overlaps = genres.filter((g) => likedGenres.has(g));
      if (overlaps.length > 0) repeats++;
      else discoveries++;
      return { rec: r, genres, overlaps };
    });

    const removed = liked.filter((r) => recIds.has(r.movieId)).length; // should be 0

    const rows = recItems
      .map(
        ({ rec, genres, overlaps }) => `
        <tr>
          <td>${escapeHtml(rec.title)}</td>
          <td class="num">${fmt(rec.score)}</td>
          <td>${genres.map(escapeHtml).join(", ") || "\u2014"}</td>
          <td>${overlaps.length ? "repeat" : "new"}</td>
        </tr>`,
      )
      .join("");

    return `
      <h3>User ${userId}</h3>
      <p class="hint">Top liked genres (rating &ge; 4 in train/val): ${topGenres}. Top-10 overlap with already-liked movies: ${removed}.</p>
      <table class="data-table">
        <thead><tr><th scope="col">Recommended title</th><th scope="col">Score</th><th scope="col">Genres</th><th scope="col">Tie to taste</th></tr></thead>
        <tbody>${rows || '<tr><td colspan="4">No eligible recommendations.</td></tr>'}</tbody>
      </table>
      <p class="hint">${repeats} of ${recs.length} recommendations share a liked genre (repeat); ${discoveries} introduce genres the user has not liked yet (discovery).</p>`;
  });

  target.innerHTML = sections.join("");
}

/**
 * Read the training hyper-parameters from the DOM, falling back to defaults when
 * the inputs are absent or invalid.
 *
 * @returns {TrainOptions}
 */
function readHyperparameters() {
  const readNumber = (id, fallback) => {
    const el = document.getElementById(id);
    if (!el) return fallback;
    const value = Number(el.value);
    return Number.isFinite(value) ? value : fallback;
  };
  const engineEl = document.getElementById("engine");
  return {
    epochs: Math.max(1, Math.round(readNumber("epochs", 20))),
    lr: readNumber("lr", 0.01),
    reg: readNumber("reg", 0.05),
    batchSize: Math.max(1, Math.round(readNumber("batch-size", 64))),
    engine: engineEl && engineEl.value === ENGINE_JS ? ENGINE_JS : ENGINE_TFJS,
  };
}

/**
 * Populate the user `<select>` from the training split.
 *
 * @returns {void}
 */
function populateUserSelect() {
  const select = document.getElementById("user-select");
  if (!select) return;
  const ids = Array.from(new Set(trainSet.map((r) => r.userId))).sort((a, b) => a - b);
  select.innerHTML = ids.length
    ? ids.map((id) => `<option value="${id}">User ${id}</option>`).join("")
    : '<option value="">No users</option>';
}

/**
 * Run the full training pipeline: initialize, train, select the best checkpoint,
 * and refresh the training curve.
 *
 * @returns {void}
 */
function runPipeline() {
  const status = document.getElementById("status");
  if (!movies.length || trainSet.length === 0) {
    if (status) status.textContent = "Data not loaded yet — reload the page.";
    return;
  }
  const opts = readHyperparameters();
  activeEngine = opts.engine;
  // Never call the TF.js path when the library is absent (offline CDN): fall
  // back to the pure-JS engine instead of throwing.
  if (activeEngine === ENGINE_TFJS && typeof tf === "undefined") {
    activeEngine = ENGINE_JS;
  }
  try {
    if (status) status.textContent = "Training\u2026";
    // Dispatch to the selected engine: same model contract, different math path.
    if (activeEngine === ENGINE_JS) {
      initializeFactorsJS(trainSet, NUM_FACTORS);
      trainingHistory = trainMFJS(trainSet, valSet, opts, testSet);
    } else {
      initializeFactors(trainSet, NUM_FACTORS);
      trainingHistory = trainMF(trainSet, valSet, opts, testSet);
    }
    bestCheckpoint = selectBestCheckpoint(trainingHistory);
    renderTrainingCurve();
    renderMetrics();
    renderAnalytics();
    if (status) {
      const engineName = activeEngine === ENGINE_JS ? "pure JS SGD" : tfBackend || "tf";
      status.textContent = bestCheckpoint
        ? `Trained ${opts.epochs} epoch(s) with ${engineName} — best val RMSE ${fmt(bestCheckpoint.valRMSE)} at epoch ${bestCheckpoint.epoch}; test RMSE ${fmt(bestCheckpoint.testRMSE)}.`
        : `Trained ${opts.epochs} epoch(s) — no history recorded.`;
    }
  } catch (error) {
    const message = String(error && error.message ? error.message : error);
    if (status) status.textContent = message;
    const target = document.getElementById("training-curve");
    if (target) {
      target.innerHTML = `<p class="empty-state">Training failed — implement the <code>TODO(hw5)</code> functions, then press &ldquo;Train&rdquo;. <code>${escapeHtml(message)}</code></p>`;
    }
  }
}

/**
 * Wire the controls once the data has been loaded. Rendering is defensive so an
 * unimplemented stub can never take the whole page down at load time.
 *
 * @returns {void}
 */
function init() {
  populateUserSelect();
  renderDatasetSummary();
  renderSplitSummary();
  renderTrainingCurve();

  const testButton = document.getElementById("run-tests");
  if (testButton) testButton.addEventListener("click", () => runTests());

  const trainButton = document.getElementById("train");
  if (trainButton) trainButton.addEventListener("click", runPipeline);

  const recommendButton = document.getElementById("recommend-btn");
  if (recommendButton) {
    recommendButton.addEventListener("click", () => {
      const select = document.getElementById("user-select");
      renderTopK(select ? select.value : 1);
    });
  }

  const recallButton = document.getElementById("recall-btn");
  if (recallButton) recallButton.addEventListener("click", () => renderRecall(10));

  const status = document.getElementById("status");
  if (status) {
    const backendName = tfBackend || "pure JS (SGD)";
    status.textContent = `Dataset ready: ${movies.length.toLocaleString("en-US")} movies, ${trainSet.length.toLocaleString("en-US")} train ratings; engine ${backendName}. Press \u201cTrain\u201d.`;
  }
}

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------

/** Marker carried by every unimplemented stub error message. */
const TODO_MARKER = "TODO(hw5)";

/** A five-rating toy training set used by several checks. */
const TOY_RATINGS = [
  { userId: 1, movieId: 1, rating: 5, timestamp: 1 },
  { userId: 1, movieId: 2, rating: 4, timestamp: 2 },
  { userId: 1, movieId: 3, rating: 3, timestamp: 3 },
  { userId: 2, movieId: 1, rating: 2, timestamp: 1 },
  { userId: 2, movieId: 2, rating: 5, timestamp: 2 },
];

/** Movies matching `TOY_RATINGS`. */
const TOY_MOVIES = [
  { id: 1, title: "Toy One", genres: [] },
  { id: 2, title: "Toy Two", genres: [] },
  { id: 3, title: "Toy Three", genres: [] },
];

/** A tiny train/val/test fixture for the recommendation checks. */
const REC_FIXTURE = {
  movies: [
    { id: 1, title: "Alpha", genres: [] },
    { id: 2, title: "Bravo", genres: [] },
    { id: 3, title: "Charlie", genres: [] },
    { id: 4, title: "Delta", genres: [] },
    { id: 5, title: "Echo", genres: [] },
  ],
  train: [{ userId: 1, movieId: 1, rating: 5, timestamp: 1 }],
  val: [{ userId: 1, movieId: 2, rating: 4, timestamp: 2 }],
  test: [{ userId: 1, movieId: 3, rating: 5, timestamp: 3 }],
};

/**
 * Build a TF.js factor model from sparse score maps.
 *
 * `userScores[id]` / `itemScores[id]` may be a number (written into factor 0,
 * so the dot product is just the product) or an array (written across factors).
 * Rows are dense and indexed by raw id, exactly like the real model, so the
 * recommendation / recall fixtures exercise the same tensor path as training.
 *
 * @param {Object<number, number|number[]>} userScores
 * @param {Object<number, number|number[]>} itemScores
 * @param {number} [globalMean=0]
 * @returns {Model}
 */
function fixtureModel(userScores, itemScores, globalMean = 0) {
  const F = NUM_FACTORS;
  const userIds = Object.keys(userScores).map(Number);
  const itemIds = Object.keys(itemScores).map(Number);

  // A number goes into factor 0 (so the dot product is the product); an array
  // is copied across the first factors.
  const rowsFor = (value) => {
    if (Array.isArray(value) || ArrayBuffer.isView(value)) return Array.from(value);
    const row = new Array(F).fill(0);
    row[0] = Number(value);
    return row;
  };

  const maxUser = Math.max(numUsers, ...userIds);
  const maxMovieFromMovies = movies.reduce((m, x) => Math.max(m, x.id), 0);
  const maxItem = Math.max(maxMovieFromMovies, ...itemIds);

  const uData = new Float32Array((maxUser + 1) * F);
  for (const id of userIds) {
    const row = rowsFor(userScores[id]);
    for (let f = 0; f < Math.min(F, row.length); f++) uData[id * F + f] = row[f];
  }
  const iData = new Float32Array((maxItem + 1) * F);
  for (const id of itemIds) {
    const row = rowsFor(itemScores[id]);
    for (let f = 0; f < Math.min(F, row.length); f++) iData[id * F + f] = row[f];
  }

  // When TensorFlow.js is unavailable (offline CDN), build the equivalent
  // pure-JS model so the prediction/recommendation/recall fixtures still run.
  if (typeof tf === "undefined") {
    return {
      kind: ENGINE_JS,
      userFactors: uData,
      itemFactors: iData,
      numFactors: F,
      numUserRows: maxUser + 1,
      numItemRows: maxItem + 1,
      globalMean,
    };
  }

  return {
    userFactors: tf.variable(tf.tensor2d(uData, [maxUser + 1, F])),
    itemFactors: tf.variable(tf.tensor2d(iData, [maxItem + 1, F])),
    globalMean,
  };
}

/**
 * A recall fixture with 12 movies, one user whose single relevant test movie
 * (movie 2) is placed at a chosen rank by the hand-made model.
 *
 * @param {number} targetRank 1 to put movie 2 first, 11 to put it last.
 * @returns {{movies: Movie[], train: Rating[], val: Rating[], test: Rating[], model: Model}}
 */
function recallFixture(targetRank) {
  const movies = [];
  for (let id = 1; id <= 12; id++) {
    movies.push({ id, title: `Movie ${id}`, genres: [] });
  }
  const train = [
    { userId: 1, movieId: 1, rating: 5, timestamp: 1 },
    { userId: 2, movieId: 2, rating: 5, timestamp: 1 },
    { userId: 2, movieId: 3, rating: 4, timestamp: 2 },
  ];
  const val = [];
  const test = [{ userId: 1, movieId: 2, rating: 5, timestamp: 2 }];
  const itemScores = { 1: 3.0 };
  if (targetRank === 1) {
    itemScores[2] = 5.0;
    for (let id = 3; id <= 12; id++) itemScores[id] = 1.0 + 0.01 * id;
  } else {
    itemScores[2] = 1.0;
    for (let id = 3; id <= 12; id++) itemScores[id] = 1.1 + 0.01 * id;
  }
  return { movies, train, val, test, model: fixtureModel({ 1: 1 }, itemScores) };
}

/**
 * Run `fn` with the module-level data globals temporarily replaced by a fixture,
 * then restore them. Keeps the checks independent of the loaded dataset.
 *
 * @param {{movies?: Movie[], train?: Rating[], val?: Rating[], test?: Rating[], model?: Model|null, history?: HistoryRow[]}} fixture
 * @param {() => any} fn
 * @returns {any}
 */
function withFixture(fixture, fn) {
  const saved = { movies, trainSet, valSet, testSet, model, trainingHistory };
  try {
    movies = fixture.movies !== undefined ? fixture.movies : movies;
    trainSet = fixture.train || [];
    valSet = fixture.val || [];
    testSet = fixture.test || [];
    model = fixture.model !== undefined ? fixture.model : null;
    trainingHistory = fixture.history || [];
    return fn();
  } finally {
    movies = saved.movies;
    trainSet = saved.trainSet;
    valSet = saved.valSet;
    testSet = saved.testSet;
    model = saved.model;
    trainingHistory = saved.trainingHistory;
  }
}

/**
 * Run the automated self-checks. Checks that call an unimplemented
 * `TODO(hw5)` stub are reported as PENDING rather than FAIL, so the harness is
 * useful before and after the assignment is implemented.
 *
 * @param {HTMLElement|null} [logElement] element that receives the text log
 * @returns {{passed: number, failed: number, pending: number, checks: Array<Object>}}
 */
function runTests(logElement) {
  const checks = [];
  const close = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;

  const pass = (name, detail) => checks.push({ name, status: "PASS", detail });
  const fail = (name, detail) => checks.push({ name, status: "FAIL", detail });
  const pending = (name, detail) => checks.push({ name, status: "PENDING", detail });

  const check = (name, fn) => {
    try {
      const outcome = fn();
      if (outcome && outcome.pending) pending(name, outcome.pending);
      else pass(name, outcome === undefined ? "" : String(outcome));
    } catch (error) {
      if (String(error && error.message).includes(TODO_MARKER)) {
        pending(name, "TODO(hw5): not implemented yet.");
      } else {
        fail(name, String(error && error.message ? error.message : error));
      }
    }
  };

  // 1. Shared helper: baseline RMSE on a toy split.
  check("computeBaselineRMSE: toy split matches the hand-computed value", () =>
    withFixture(
      {
        train: [
          { userId: 1, movieId: 1, rating: 2, timestamp: 1 },
          { userId: 1, movieId: 2, rating: 4, timestamp: 2 },
        ],
      },
      () => {
        const rmse = computeBaselineRMSE(trainSet);
        if (!close(rmse, 1)) throw new Error(`rmse = ${rmse}, expected 1`);
        return `rmse=${fmt(rmse)}`;
      },
    ),
  );

  // 2. Shared helper: zero-denominator guard.
  check("computeBaselineRMSE: empty split returns 0 (no division by zero)", () =>
    withFixture({ train: [{ userId: 1, movieId: 1, rating: 3, timestamp: 1 }] }, () => {
      if (computeBaselineRMSE([]) !== 0) throw new Error("expected 0");
      return "guarded";
    }),
  );

  // 3. Formatting helper.
  check("fmt: formats numbers and guards non-finite input", () => {
    if (fmt(1.23456) !== "1.2346") throw new Error(`fmt(1.23456) = ${fmt(1.23456)}`);
    if (fmt(2, 2) !== "2.00") throw new Error(`fmt(2, 2) = ${fmt(2, 2)}`);
    if (fmt(Number.NaN) !== "\u2014") throw new Error("NaN must render as an em dash");
    return "ok";
  });

  // 4. Formatting helper.
  check("movieTitle: resolves ids and falls back for unknown ids", () =>
    withFixture({ movies: [{ id: 7, title: "Seven Samurai", genres: [] }] }, () => {
      if (movieTitle(7) !== "Seven Samurai") throw new Error("known id failed");
      if (movieTitle(999) !== "Movie #999") throw new Error(`fallback = ${movieTitle(999)}`);
      return "ok";
    }),
  );

  // 5. Student stub: prediction and clamp (TF.js gather + dot).
  check("predictRating: clamps to [1, 5] and falls back to the global mean", () =>
    withFixture(
      {
        movies: TOY_MOVIES,
        train: TOY_RATINGS,
        model: fixtureModel({ 1: [1, 2] }, { 3: [3, 4] }, 3),
      },
      () => {
        const pred = predictRating(1, 3);
        if (pred !== 5) throw new Error(`predictRating = ${pred}, expected 5 (3 + 11 clamped)`);
        const fallback = predictRating(100000, 3);
        if (fallback !== 3) throw new Error(`out-of-range user must return globalMean, got ${fallback}`);
        return `clamped=${pred}, fallback=${fallback}`;
      },
    ),
  );

  // 6. Student stub: TF.js variable allocation on the toy 5-rating set.
  //    Requires TensorFlow.js; reported PENDING when the CDN is unavailable.
  check("initializeFactors: allocates tf.Variable factors and the global mean", () => {
    if (typeof tf === "undefined") return { pending: "TensorFlow.js unavailable" };
    return withFixture({ movies: TOY_MOVIES, train: TOY_RATINGS, model: null }, () => {
      const created = initializeFactors(TOY_RATINGS, 4);
      if (!created || !model) throw new Error("model was not assigned");
      if (!close(model.globalMean, 3.8)) {
        throw new Error(`globalMean = ${model.globalMean}, expected 3.8`);
      }
      if (!isTfModel(model)) throw new Error("userFactors/itemFactors are not tf tensors");
      if (model.userFactors.shape[1] !== 4 || model.itemFactors.shape[1] !== 4) {
        throw new Error(
          `factor width = ${model.userFactors.shape[1]} / ${model.itemFactors.shape[1]}, expected 4`,
        );
      }
      if (model.itemFactors.shape[0] !== 4) {
        throw new Error(`item rows = ${model.itemFactors.shape[0]}, expected 4 (max movie id 3 + 1)`);
      }
      if (model.userFactors.shape[0] < 3) {
        throw new Error(`user rows = ${model.userFactors.shape[0]}, expected at least 3`);
      }
      const row = model.userFactors.slice([1, 0], [1, 4]).dataSync();
      if (!Array.from(row).every(Number.isFinite)) {
        throw new Error("user factor row is not finite");
      }
      return `user [${model.userFactors.shape}], item [${model.itemFactors.shape}]`;
    });
  });

  // 7. Student stub: batched training loop.
  //    Requires TensorFlow.js; reported PENDING when the CDN is unavailable.
  check("trainMF: returns one finite history row per epoch", () => {
    if (typeof tf === "undefined") return { pending: "TensorFlow.js unavailable" };
    return withFixture({ movies: TOY_MOVIES, train: TOY_RATINGS, val: TOY_RATINGS, model: null }, () => {
      initializeFactors(TOY_RATINGS, 4);
      const history = trainMF(TOY_RATINGS, TOY_RATINGS, {
        epochs: 3,
        lr: 0.02,
        reg: 0.05,
        batchSize: 2,
      });
      if (!Array.isArray(history) || history.length !== 3) {
        throw new Error(`history length = ${history && history.length}, expected 3`);
      }
      history.forEach((row, i) => {
        if (row.epoch !== i + 1 && row.epoch !== i) {
          throw new Error(`row ${i} epoch = ${row.epoch}`);
        }
        if (!Number.isFinite(row.trainRMSE) || !Number.isFinite(row.valRMSE)) {
          throw new Error(`row ${i} RMSE is not finite`);
        }
      });
      return history.map((r) => `e${r.epoch}:${fmt(r.trainRMSE)}`).join(" ");
    });
  });

  // 8. Student stub: checkpoint selection and tie-break.
  check("selectBestCheckpoint: smallest valRMSE wins, ties keep the earliest epoch", () => {
    const history = [
      { epoch: 2, trainRMSE: 0.9, valRMSE: 0.4 },
      { epoch: 1, trainRMSE: 1.0, valRMSE: 0.3 },
      { epoch: 3, trainRMSE: 0.8, valRMSE: 0.3 },
    ];
    const best = selectBestCheckpoint(history);
    if (!best || best.epoch !== 1) {
      throw new Error(`best epoch = ${best && best.epoch}, expected 1 (tie -> earliest)`);
    }
    return `best epoch ${best.epoch}`;
  });

  // 9. Student stub: recommendation filtering (tensor matMul + topk).
  check("recommendTopK: excludes train+val items but allows test-rated items", () =>
    withFixture(
      {
        movies: REC_FIXTURE.movies,
        train: REC_FIXTURE.train,
        val: REC_FIXTURE.val,
        test: REC_FIXTURE.test,
        model: fixtureModel({ 1: 1 }, { 1: 5, 2: 4, 3: 3, 4: 2, 5: 1 }),
      },
      () => {
        const recs = recommendTopK(1, 10);
        const ids = recs.map((r) => r.movieId);
        if (ids.includes(1)) throw new Error("movie 1 is in train and must be excluded");
        if (ids.includes(2)) throw new Error("movie 2 is in val and must be excluded");
        if (!ids.includes(3)) throw new Error("movie 3 is test-rated and must be eligible");
        for (let i = 1; i < recs.length; i++) {
          if (recs[i - 1].score < recs[i].score) {
            throw new Error("results are not sorted by descending score");
          }
        }
        return ids.join(", ");
      },
    ),
  );

  // 10. Student stub: recall at rank 1 vs rank 11.
  check("recallAtK: relevant movie at rank 1 scores 1.0; at rank 11 scores 0.0", () => {
    const rankOne = withFixture({ ...recallFixture(1), history: [] }, () => recallAtK(10));
    if (!close(rankOne.overallRecall, 1)) {
      throw new Error(`rank-1 recall = ${rankOne.overallRecall}, expected 1`);
    }
    const rankEleven = withFixture({ ...recallFixture(11), history: [] }, () => recallAtK(10));
    if (!close(rankEleven.overallRecall, 0)) {
      throw new Error(`rank-11 recall = ${rankEleven.overallRecall}, expected 0`);
    }
    return `rank1=${fmt(rankOne.overallRecall)} rank11=${fmt(rankEleven.overallRecall)}`;
  });

  // 11. Student stub: empty-result guard.
  check("recallAtK: empty test set returns 0 without dividing by zero", () =>
    withFixture(
      {
        movies: REC_FIXTURE.movies,
        train: REC_FIXTURE.train,
        val: [],
        test: [],
        model: fixtureModel({ 1: 1 }, { 1: 1, 2: 1, 3: 1, 4: 1, 5: 1 }),
      },
      () => {
        const result = recallAtK(10);
        if (result.overallRecall !== 0) {
          throw new Error(`expected 0, got ${result.overallRecall}`);
        }
        if (!Array.isArray(result.perUser) || result.perUser.length !== 0) {
          throw new Error("perUser must be empty");
        }
        if (!result.excluded) throw new Error("excluded report missing");
        return "guarded";
      },
    ),
  );

  const passed = checks.filter((c) => c.status === "PASS").length;
  const failed = checks.filter((c) => c.status === "FAIL").length;
  const pendingCount = checks.filter((c) => c.status === "PENDING").length;

  const lines = [
    `HW5 self-checks — pass ${passed}, fail ${failed}, pending ${pendingCount} (of ${checks.length})`,
    "",
    ...checks.map((c) => `${c.status.padEnd(7)} ${c.name}${c.detail ? ` \u2014 ${c.detail}` : ""}`),
  ];
  const text = lines.join("\n");

  const target = logElement || document.getElementById("testLog");
  if (target) target.textContent = text;

  return { passed, failed, pending: pendingCount, checks };
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

/**
 * Select the TF.js backend, load the MovieLens data, split it chronologically,
 * and wire the page. `data.js` must be loaded before this script.
 *
 * TensorFlow.js is OPTIONAL: when the CDN is unavailable (offline, blocked) the
 * page still works by switching to the pure-JS SGD engine, so the dataset and
 * the self-checks stay usable instead of failing to load.
 *
 * @returns {Promise<void>}
 */
window.onload = async function () {
  const status = document.getElementById("status");
  const backend = await initTf(); // null when the tf CDN did not load

  if (!backend) {
    // No TensorFlow.js: force the pure-JS engine and disable the TF option so
    // pressing "Train" cannot throw on a missing `tf`.
    activeEngine = ENGINE_JS;
    const engineEl = document.getElementById("engine");
    if (engineEl) {
      engineEl.value = ENGINE_JS;
      const tfOption = engineEl.querySelector('option[value="tfjs"]');
      if (tfOption) tfOption.disabled = true;
    }
  }

  try {
    if (status) status.textContent = "Loading data\u2026";
    await loadData();
    const split = splitByTimestamp();
    trainSet = split.train;
    valSet = split.val;
    testSet = split.test;
    init();
    if (!backend && status) {
      status.textContent += " TensorFlow.js unavailable — using the Pure JS (SGD) engine.";
    }
  } catch (error) {
    console.error("Initialization failed:", error);
    if (status) status.textContent = `Error: ${error.message}`;
  }
};
