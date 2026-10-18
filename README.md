# PayeeShield

**Stop the payment before the money's gone.**

PayeeShield scores every UPI payment in real time, before the payer taps confirm. It looks at the **receiving** side: is this payee behaving like a mule account? It returns a risk score from 0 to 100, a decision (ALLOW, WARN or BLOCK) and plain-language reasons the payer's app can show.

Built for IBM Z Datathon 2026, theme "Real-Time AI for Critical Decisions" (AI for Good), by Team ACHILLES.

---

## The problem

Digital payment fraud accounts for about 56.5% of all reported bank frauds in RBI's FY25 annual report, and 51% of UPI fraud victims never report it (LocalCircles, 2025). Once money lands in a mule account, it is usually moved on within minutes, so recovery after the fact rarely works.

Existing defences mostly watch the **payer** (unusual device, unusual amount) or flag mule accounts in batches after the fact. NPCI shows real-time AI warnings, and RBI's MuleHunter.AI identifies mule accounts. What is missing at the moment of payment is a single check on the account **receiving** the money, made before the payer confirms.

## The ₹1 story

A scammer gave our class tutor a phone number to pay. Our tutor sent ₹1 on UPI as a test and it **succeeded**. When our tutor called the number, it was **out of service**.

That is possible because UPI checks the SIM only once, at registration (SMS device binding). After that, the number can be dead on the telecom network and the UPI ID keeps receiving money. NPCI delinks only numbers inactive for 12 months or more, and telecoms recycle numbers after 90 days, so scam numbers that are still in use slip through.

PayeeShield scores that exact payment at **BLOCK** with these reasons:

- Payee's mobile number is unreachable on the telecom network
- 40 different people paid this payee in 24h
- Money leaves payee account within 2 minutes
- Payee UPI ID created 3 days ago

## How it works

Each payment is described by 10 features (`src/features.py`). The model is trained on them and a transparent rule layer turns them into reasons.

| Signal | Feature | Why it matters | Reason shown to the payer |
|---|---|---|---|
| Payee SIM reachable | `payee_sim_reachable` | Mule numbers are often switched off or dead after setup | Payee's mobile number is unreachable on the telecom network |
| Payee SIM age | `payee_sim_age_days` | Fresh SIMs are cheap to burn | Payee SIM only N days old |
| SIM swaps | `payee_sim_changes_90d` | Swaps are how old accounts get hijacked | Payee SIM changed N times in 90 days |
| UPI ID age | `payee_vpa_age_days` | Mule VPAs are new | Payee UPI ID created N days ago |
| Unique senders in 24h | `payee_unique_senders_24h` | Many strangers paying a new or dead-SIM account is a collection pattern | N different people paid this payee in 24h |
| Outflow speed | `payee_outflow_minutes` | Mules move money out within minutes | Money leaves payee account within N minutes |
| First-time payee | `first_time_payee` | Most fraud goes to someone the payer has never paid | First payment to this payee |
| Payer on a call | `payer_on_call` | Victims are kept on a call during "digital arrest" scams | You are on a call while paying a new payee — common in 'digital arrest' scams |
| Amount | `amount` | Large first payments carry more risk | Large amount (₹N) to a new payee |
| Hour | `hour` | Context only | (none) |

Decision thresholds: **WARN at 40**, **BLOCK at 75**. At most four reasons are shown, most severe first, and only for WARN or BLOCK. An ALLOW shows "No risk signals found".

Unique senders alone is not treated as suspicious: a busy tea stall gets dozens of payers. The rule fires only when the payee is also new or unreachable.

## Architecture

```
                   telecom status feed            UPI history feed
                 (SIM reachable, SIM age,   (VPA age, unique senders 24h,
                     SIM swaps 90d)              outflow speed, payer-payee history)
                           │                               │
                           ▼                               ▼
 ┌────────────┐   ┌──────────────┐   ┌──────────────────────────────────────┐
 │ Payer app  │──▶│  UPI switch  │──▶│  PayeeShield scorer                  │
 │ (PSP)      │   │              │   │  on IBM Z / LinuxONE                 │
 └────────────┘   └──────────────┘   │  features → model → score 0-100      │
       ▲                             │  + reason rules                      │
       │                             └──────────────────────────────────────┘
       │                                              │
       └──────── ALLOW / WARN / BLOCK + reasons ──────┘
                 (shown before the payer confirms)
```

## Repo structure

```
payeeshield/
├── src/
│   ├── features.py      shared contract: FEATURES, WARN_AT, BLOCK_AT, paths
│   ├── data_gen.py      synthetic UPI transactions (legit segments + 4 fraud scenarios)
│   ├── train.py         trains the model, writes models/model.joblib and metrics.json
│   ├── scorer.py        real-time scorer: probability → risk score, decision, reasons
│   ├── api.py           FastAPI service: POST /score, POST /score/batch, GET /health
│   ├── demo_story.py    scores the ₹1 story and other demo cases in the terminal
│   └── stream_sim.py    replays a stream through the scorer, reports latency and catch rate
├── models/              trained model and metrics
├── data/                generated dataset
├── tests/               pytest suite
├── dashboard/
│   └── index.html       offline live demo, no backend needed
├── docs/
│   └── project_description.md
└── requirements.txt
```

Fraud scenarios in the synthetic data: `dead_sim_mule`, `digital_arrest`, `fake_refund_qr`, `sim_swap_mule`. Legit segments: everyday person-to-person payments, busy merchants, and new shops (which have young UPI IDs and many payers on purpose, to keep the problem hard).

## Quickstart

```bash
pip install -r requirements.txt
python3 src/data_gen.py          # generate synthetic transactions
python3 src/train.py             # train and evaluate, writes models/
python3 src/demo_story.py        # score the ₹1 story and other cases
python3 src/stream_sim.py        # simulated live stream on unseen payments: latency and catch rate
uvicorn src.api:app --port 8000  # start the scoring API
open dashboard/index.html        # live demo in the browser, works offline (Linux: xdg-open)
```

Run the tests with `python3 -m pytest -q`. `docs/stream_summary.json` was produced with `python3 src/stream_sim.py -n 3000 --quiet --json docs/stream_summary.json`.

## API example

```bash
curl -s -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
        "txn_id": "DEMO-1",
        "amount": 1,
        "hour": 22,
        "payee_sim_reachable": 0,
        "payee_sim_age_days": 6,
        "payee_sim_changes_90d": 2,
        "payee_vpa_age_days": 3,
        "payee_unique_senders_24h": 40,
        "payee_outflow_minutes": 2,
        "first_time_payee": 1,
        "payer_on_call": 0
      }'
```

Sample response (score and latency vary with the trained model and hardware):

```json
{
  "txn_id": "DEMO-1",
  "risk_score": 89,
  "decision": "BLOCK",
  "reasons": [
    "Payee's mobile number is unreachable on the telecom network",
    "40 different people paid this payee in 24h",
    "Money leaves payee account within 2 minutes",
    "Payee UPI ID created 3 days ago"
  ],
  "latency_ms": 4.314,
  "model_version": "model-6f7c1e97"
}
```

`POST /score/batch` takes a JSON array of the same objects. `GET /health` reports whether the model is loaded.

## Results

Model metrics are from a held-out 20% test split (10,000 payments, 3.1% fraud) of the synthetic data. Latency and throughput are from `stream_sim.py` replaying 3,000 fresh payments generated with a different seed, on a standard CPU.

| Metric | Value |
|---|---|
| ROC-AUC | 0.964 (payer-only baseline: 0.894) |
| PR-AUC | 0.832 |
| PR-AUC of payer-only baseline (amount, hour, first-time payee, payer on call) | 0.482 |
| Fraud recall at WARN (score ≥ 40) | 90.7% (held-out test set) |
| Precision at BLOCK (score ≥ 75) | 55.4% |
| p95 scoring latency | 6.5 ms (p50 3.1 ms, p99 13.3 ms; one payment at a time, standard CPU, not yet on IBM Z) |
| Throughput | ~256 payments/sec, one Python process, one payment at a time |
| Stream catch rate (3,000 unseen payments) | 87 of 92 frauds flagged (94.6%), 148 false alarms |

The gap between PR-AUC and the payer-only baseline is the case for looking at the payee side.

## Why IBM Z

UPI handles hundreds of millions of payments a day, and a fraud check has to fit inside the payment's time budget. Sending every transaction off to a separate scoring cluster adds network hops and moves sensitive payment data off the platform.

IBM Z and LinuxONE are where much of India's core banking already runs. The Telum processor has an on-chip AI accelerator, so a model can score a transaction inside the same system that processes it. That is the deployment PayeeShield is designed for: score in-transaction, on-platform, for every payment rather than a sample, and return the decision before the payer confirms.

The model is a compact tabular classifier on 10 features, the kind of workload that suits in-transaction inference.

## Limitations and honesty

- **Synthetic data.** All training and evaluation data is generated by `src/data_gen.py`. Segments overlap on purpose and include label noise, but real-world performance will differ.
- **Telecom signal assumes a feed.** SIM reachability, SIM age and SIM swap history assume a telecom or DoT feed (for example through a DoT or telecom API) that a payment system can query in real time. That access does not exist for this prototype.
- **Not yet benchmarked on IBM Z.** This prototype runs on standard Python. It is designed for IBM Z / LinuxONE but has not been tested on it, and the latency figures above are from commodity hardware.
- **The dashboard uses its own scorer.** `dashboard/index.html` runs a transparent weighted rule model in JavaScript so it works offline. It mirrors the reason rules but is not the trained model.
- **False alarms on new shops.** Genuine new merchants share features with mule accounts (young UPI ID, many payers). Warnings rather than blocks are the right response there.

## Roadmap

1. Validate on real, anonymised UPI and telecom data with a bank or PSP partner.
2. Port the scorer to IBM Z / LinuxONE and benchmark on Telum's on-chip AI accelerator (for example via ONNX and the IBM Z Deep Learning Compiler).
3. Add graph signals: payees that share devices, beneficiaries or onward transfers.
4. Feed outcomes back from confirmed fraud reports (1930 helpline, cybercrime.gov.in) for retraining.
5. Payer-side warning screen in local languages.

## Team

Team ACHILLES, PSG Institute of Technology and Applied Research (PSG iTech), Coimbatore.
