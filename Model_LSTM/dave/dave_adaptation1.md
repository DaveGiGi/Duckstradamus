# Change Summary — Jacques' LSTM → Dave's Price Model

**Goal:** 24h-ahead electricity price forecast (node OTA), beat the naive baseline.

---

## Phase 1 — Preprocessing

- **Naive baseline** — changed `shift(1)` → `shift(24)`; kept as a separate Series, not a feature. _Day-ahead baseline = same hour yesterday; the bar to beat, not a predictor._
- **Weather `shift(-24)`** — pulls tomorrow's weather onto today's row. _We know tomorrow's weather (forecast) but not its prices; only weather gets shifted._
- **Dropped last 24 rows** — _weather shift empties the tail; delete it._
- **Day-of-week cyclic (sin/cos)** — _weekday vs weekend signal; holiday column already covers weekends, so this is minor._
- **Rolling price features** — 24h avg of target + 24h avg of all 9 nodes. _Backward-only window = no leakage; captures recent price level._
- **Cyclic wind direction (sin/cos)** — _degrees are circular, 359°≈1°; raw degree columns dropped._

## Phase 2 — Data Preparation

- **Expanding-window CV** — replaced fixed overlapping folds; 18 folds, train grows from 3yr +2mo each, test next 2mo. _Walk-forward = always predict future from past; growing data finally lets it learn the yearly cycle._
- **No leakage from windowing** — _prices/production/demand safe because the window ends a full day before predicted hours; only weather needed shifting._
- **`INPUT_LENGTH` 168 → 336** — _2-week window so it sees two weekly cycles (best practice, kept non-negotiable)._
- **`STRIDE` → 12 (or 24)** — _one sample every 12/24h; cut RAM/time; 24 = midnight-aligned day-ahead, the real task._
- **float32 + vectorised `get_X_y`** — _`sliding_window_view` builds samples ~100× faster, half the RAM; fixed laptop crash._

## Phase 3 — Architecture

- **Stacked 2-layer LSTM (64→32)** — _depth helps sequence modelling._
- **Huber loss** (was MSE) — _robust to price spikes._

## Phase 4 — Training

- **EarlyStopping patience 3 → 8** — _let it converge._
- **CV loop** — trains all 18 folds, stores `fold_results`, prints mean±std. _Each fold scored on its own test window._

---

## Left as-is (dead but harmless)

- **`train_test_split` cell** — unused now (expanding folds do the split). Left in place.
- **Single-model cells** (`model = init_model(...)` + summary, single-fold eval) — leftovers from the old single-run flow. Left in place; the CV loop handles everything.

## Next session — Phase 5

- Compare each fold's LSTM vs naive `shift(24)` on the **same** test window — the real scorecard.
- Runtime note: full CV on CPU is slow (~hours at `INPUT_LENGTH=336`). Time one fold first; if needed, the no-quality-loss levers are bigger `STEP` (fewer folds) or larger `STRIDE`.
