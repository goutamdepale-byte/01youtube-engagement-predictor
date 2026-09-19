# Real-Time YouTube Engagement Prediction — Master Project Roadmap (Updated)

**Project**: Real-time social media analyst (YouTube first, extensible to other platforms)
**Core ML task**: Predict a video's engagement rate at ~24 hours post-publish, using features from its first 6 hours
**Architecture theme**: Zero-cost cloud automation (MongoDB Atlas + AWS Lambda + EventBridge)

---

## PART A — COMPLETED WORK (submitted / working)

### Day 1: Data Cleaning & EDA
- Loaded 80,081-row raw dataset (10-min snapshots, Aug 8–21, 2026)
- Fixed Excel corruption issue (`#NAME?` video_id values) — dropped 11 affected videos
- Handled missing data: forward/backward-filled static fields per video, forward-filled counters, recomputed derived ratios
- Parsed ISO 8601 `duration` into seconds
- EDA visuals: engagement distribution, category averages, correlation/leakage heatmap, sample growth curves
- **Output**: `cleaned_data.csv` — 41,487 rows, 162 usable videos

### Day 2: Feature Engineering & Target Construction
- Defined early window (≤6h) and target window (18–30h, closest to 24h)
- Built early-signal features: category, duration, upload hour/day/weekend, early views/likes/comments/rates
- **Deliberately excluded title-text features** — justified by small sample size (162 videos) risking overfitting
- Built target: `target_engagement_rate` from each video's closest snapshot to 24h
- Built-in leakage check (confirmed no raw current-time counters leaked into features)
- **Output**: `training_table.csv` — 162 rows × 17 columns

### Day 3: Modeling
- Random Forest Regressor, evaluated via 5-fold cross-validation
- **Results**: R² = 0.835 (±0.135), RMSE = 1.15, MAE = 0.64
- Feature importance: `early_engagement_rate` (49%) and `early_like_rate` (36%) dominate — model mainly captures "engagement momentum"
- **Output**: `engagement_model.pkl`, `feature_columns.pkl`

### Day 4: Real-Time Prediction Pipeline
- `youtube_api.py` — secure API key handling (`.env`), URL parsing (all formats), live data fetching
- `predict_live.py` — converts live data to model's feature format, handles unseen categories gracefully, flags out-of-window predictions with honest warnings
- Tested successfully on real live videos (fresh + old, known + unknown category)

### Day 5: Streamlit Dashboard
- URL input → live prediction with current stats, badge (Low/Med/High from real data percentiles), current-vs-predicted chart, feature importance chart, warnings
- **Verified working** end-to-end with real videos

---

## PART B — CLOUD AUTOMATION (in progress — status updated)

### ✅ Phase 1: MongoDB Atlas Setup — COMPLETE
- Free-tier M0 cluster created (permanent free, 512MB — not a trial), region ap-south-1
- Database `youtube_engagement`, collections: `videos_master`, `snapshots`
- Network access configured (`0.0.0.0/0` — required since Lambda has no fixed IP)
- **TTL index** created on `snapshots.snapshot_time` — auto-expires documents after 60 days
- Performance indexes: unique index on `videos_master.video_id`, index on `videos_master.published_at`, index on `snapshots.video_id`
- Connection verified via `mongo_setup.py` — ping, write/read test, cleanup all passed

### ✅ Phase 2: Convert Collection Scripts to MongoDB — COMPLETE
- `discover_videos.py` and `poll_snapshots.py` converted to read/write MongoDB instead of parquet
- Upsert-based dedup confirmed working (re-running discovery never creates duplicate `video_id` entries)
- Age-based active-video filtering confirmed working (videos >32h correctly excluded from polling)
- **Corrected settings** (validated through design review):
  - `FRESHNESS_HOURS = 3` — discovery must stay well under the 6h early-window cutoff, or discovered videos can never yield valid early features
  - `DISCOVERY_INTERVAL_HOURS = 3` — run discovery repeatedly to accumulate more videos/category over time, instead of widening the freshness window
  - `MAX_VIDEOS_PER_CATEGORY = 40`, `STOP_POLLING_AFTER_HOURS = 32`
- Locally tested end-to-end against real Atlas cluster before AWS deployment — confirmed working via Windows Task Scheduler (temporary bridge step) before full AWS migration

### ✅ Phase 4: AWS Lambda + EventBridge Automation — COMPLETE (fully live)

**Two Lambda functions deployed and running:**

| Function | Runtime | Handler | Schedule | Status |
|---|---|---|---|---|
| `youtube-discover-videos` | Python 3.12 | `lambda_discover.lambda_handler` | `rate(3 hours)` via EventBridge rule `discover-every-3-hours` | Live |
| `youtube-poll-snapshots` | Python 3.12 | `lambda_poll.lambda_handler` | `rate(10 minutes)` via EventBridge rule `poll-every-10-minutes` | Live |

**Key implementation decisions:**
- Rewrote collection logic **without pandas** (`lambda_discover.py`, `lambda_poll.py`) to keep deployment packages small enough for direct upload — avoids needing S3, keeping the zero-cost design intact
- Dependencies installed with Linux-targeting flags on Windows:
  ```
  pip install --target ./lambda_package_X --platform manylinux2014_x86_64 --implementation cp --python-version 3.12 --only-binary=:all: --upgrade pymongo google-api-python-client google-auth-httplib2 google-auth-oauthlib python-dotenv
  ```
- Deployed via AWS CLI (`aws lambda update-function-code`) rather than console upload, since package size (~29.7MB) triggered browser upload warnings
- API key + MongoDB URI stored as **Lambda environment variables** (not Secrets Manager — avoids a recurring cost)
- Triggers created via Lambda's own "Add trigger" → "EventBridge (CloudWatch Events)" → "Schedule expression (Legacy)" path (simpler than the standalone EventBridge console)

**Real issues hit and resolved during deployment (useful for report/viva):**
1. `ModuleNotFoundError: No module named 'lambda_function'` — Handler was left at AWS's default name instead of being set to the actual filename (`lambda_discover.lambda_handler`)
2. `No module named 'lambda_discover'` — zip was created by compressing the *folder* itself instead of its *contents*, nesting the code one level too deep. Fixed by zipping from *inside* the folder
3. `cannot import name 'exceptions' from 'cryptography.hazmat.bindings._rust'` — classic Windows-vs-Linux binary mismatch; `pip install` had downloaded Windows-compiled native dependencies. Fixed with `--platform manylinux2014_x86_64 --only-binary=:all:`
4. Same cryptography error recurred on the **poll** function specifically because the old (broken, Windows-compiled) package folder wasn't deleted before reinstalling — fixed by fully removing the folder (`rmdir /s /q`) before a clean reinstall
5. Python 3.14 (Lambda's initial default) had incomplete Linux wheel availability for native-dependency packages — downgraded Lambda runtime to **Python 3.12** for full compatibility
6. `MONGODB_URI environment variable not set` — environment variables were set on the wrong function / not saved — fixed by re-adding them under the correct function's Configuration tab

**Verification**: confirmed via CloudWatch Logs and direct MongoDB Atlas inspection that both functions run automatically and insert real data with zero manual triggering.

**Local Task Scheduler tasks disabled** once AWS Lambda was confirmed working, to avoid duplicate polling and doubled API quota usage.

**Zero-cost safety net**: AWS Budgets "zero spend budget" configured, alerting by email if account spend ever exceeds $0.01.

### Phase 3: Category-Balanced Data Collection — IN PROGRESS (background, automatic)
- Running continuously via the Phase 4 Lambda automation — **no manual action required**
- Snapshot volume scales with active video count: ~1 snapshot per active video per 10-minute cycle (verified: 3 active videos → 3 snapshots/cycle in early testing; expected to grow to an estimated 50–100 steady-state active videos as discovery accumulates more over the following days)
- Estimated steady-state storage: ~86–173MB (well within MongoDB's 512MB free limit), self-bounded by the 60-day TTL on `snapshots`
- **Target**: let this run for a minimum of 2–3 days (ideally longer) to build meaningful coverage across all ~14 assignable categories for the configured region (India), especially low-volume categories (Sports, News & Politics, Education, Science & Technology)
- **Can run in parallel with Phase 6** (dashboard upgrades) — no blocking dependency

### Action Item Added: Auto-Deletion for `videos_master`
- Currently, **only `snapshots` has a TTL index** (60-day auto-expiry) — `videos_master` has no automatic cleanup, so video metadata records accumulate indefinitely
- At current scale this is not urgent (metadata records are tiny — a few MB even after months), but for full lifecycle consistency with the "sliding window" retraining design, add a matching TTL (or a scheduled cleanup Lambda) on `videos_master`:
  - **Option A (simple)**: Add a TTL index on a `discovered_at` field with a longer retention window (e.g., 90 days) — keeps master records around longer than raw snapshots, useful for historical reference
  - **Option B (more control)**: A small scheduled Lambda (e.g., weekly) that deletes `videos_master` documents older than N days AND have no remaining snapshots — avoids deleting a video record while it might still be mid-target-window
  - **Recommended**: Option A for simplicity, given the low storage cost either way
- **Status**: not yet implemented — add during Phase 7 (Automation Layer) alongside the retraining Lambda, since both involve scheduled maintenance jobs

### Phase 5: Retrain Model on Expanded Data (~1–2 days, blocked on Phase 3 completing)
- Pull `videos_master` + `snapshots` from MongoDB into pandas
- Re-run Day 1→2→3 pipeline on the larger, more category-diverse dataset
- Compare new vs old model performance — document changes honestly
- Save versioned model, log metrics to a new `model_log` collection

### Phase 6: Dashboard Upgrades (~2–3 days — can start now, in parallel with Phase 3)
- **Virality badge**: percentile-based scoring derived from existing model output vs training distribution (no new model needed)
- **Gauge/speedometer chart** for virality score
- **Trend Insights panel**: category-wise engagement trends over the collection period, best upload times — descriptive analytics from historical snapshot data
- **Radar chart**: this video vs its category's average stats
- UI/UX polish: consistent dark theme, spacing, icons
- Dashboard reads model via the `current_model` pointer (see Phase 7) — no manual file swaps ever again

### Phase 7: Full Automation Layer (~1–2 days)
- Weekly **Retrain Lambda**: pulls recent MongoDB data → cleans → engineers features → trains → evaluates
- **Champion/challenger safety check**: new model only replaces the live one if it performs equal or better
- `current_model` pointer document in MongoDB — dashboard always reads whichever version is marked active, updated automatically by the Retrain Lambda
- All retrain runs (including failures) logged to `model_log`
- CloudWatch Alarms + SNS email alerts if Poll/Discover/Retrain Lambdas fail repeatedly
- **Add `videos_master` auto-deletion here** (see Action Item above) — bundle as a third scheduled maintenance Lambda, or fold into the weekly retrain job

### Phase 8: Report & Documentation Update (~2 days — update continuously)
- Updated methodology: full dataset size, category coverage achieved
- New section: **"Model Lifecycle & Concept Drift"** — sliding-window retraining rationale
- Before/after model comparison table (6 categories → full coverage)
- Updated architecture diagram: YouTube API → Lambda (Discover + Poll + Retrain) → MongoDB (with TTL auto-cleanup) → Model → Dashboard
- **New: "AWS Deployment Lessons Learned" section** — the 6 real issues resolved during Lambda packaging (platform mismatch, handler naming, zip structure, Python version compatibility) make strong, specific engineering-maturity talking points for a viva
- Screenshots of upgraded dashboard (virality badge, trend panel, radar chart)
- Zero-cost architecture explanation, including the Budgets safety net

---

## Automation Summary — What Runs on Its Own (Current State)

| Job | Frequency | Trigger | Status | Manual work required |
|---|---|---|---|---|
| Discover videos | Every 3 hours | EventBridge rule `discover-every-3-hours` → `youtube-discover-videos` | Live | None |
| Poll snapshots | Every 10 minutes | EventBridge rule `poll-every-10-minutes` → `youtube-poll-snapshots` | Live | None |
| Snapshot auto-deletion | Continuous | MongoDB TTL index (60 days) | Live | None |
| Video metadata auto-deletion | — | — | Not yet implemented | Planned for Phase 7 |
| Retrain model | Weekly | EventBridge → Retrain Lambda (planned) | Not yet built | Planned for Phase 7 |
| Dashboard | On-demand | User pastes URL | Live (current model) | None |
| Failure alerts | On failure | CloudWatch → SNS (planned) | Not yet built | Planned for Phase 7 |
| Zero-cost safety net | Continuous | AWS Budgets zero-spend alert | Live | Check email occasionally |

---

## Cost Summary (Target: $0 — currently holding at $0)

| Service | Tier used | Cost |
|---|---|---|
| MongoDB Atlas | M0 (permanent free) | $0 |
| AWS Lambda | Always Free (1M requests/month) | $0 |
| AWS EventBridge | Free at this scheduling volume | $0 |
| CloudWatch Logs | Always Free (5GB/month) | $0 |
| ~~AWS Secrets Manager~~ | Avoided — not free | N/A |
| ~~AWS S3~~ | Avoided — free tier expires after 12 months | N/A |

**Safety net**: $0.01 AWS Budgets zero-spend alert — **configured and active**.

---

## Updated Suggested Execution Order (from current point)

1. ~~Phase 1 (MongoDB setup)~~ — done
2. ~~Phase 2 (script conversion)~~ — done
3. ~~Phase 4 (AWS Lambda deployment)~~ — done, fully live
4. **Phase 3 (let collection run 2–3+ more days)** — in progress, no action needed
5. **Phase 6 (dashboard upgrades)** — start now, in parallel with step 4
6. Phase 5 (retrain on expanded data) — once Phase 3 has accumulated enough category coverage
7. Phase 7 (automation/champion-challenger layer + `videos_master` auto-deletion)
8. Phase 8 (report/documentation) — updated continuously, finalized last
