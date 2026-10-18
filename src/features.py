"""Shared contract for PayeeShield. Every module imports from here.

PayeeShield scores a UPI payment in real time, before the payer confirms,
by looking at the RECEIVING side (is this payee behaving like a mule?)
as well as the payment itself.
"""

# Order matters: the model is trained and scored on columns in this order.
FEATURES = [
    "amount",                    # INR, float
    "hour",                      # 0-23, local hour of the payment
    "payee_sim_reachable",       # 1 if payee's mobile number is reachable on the telecom network, else 0
    "payee_sim_age_days",        # days since payee's current SIM was activated
    "payee_sim_changes_90d",     # number of SIM swaps on the payee number in last 90 days
    "payee_vpa_age_days",        # days since payee's UPI ID (VPA) was created
    "payee_unique_senders_24h",  # distinct people who paid this payee in the last 24h
    "payee_outflow_minutes",     # median minutes before money received by payee is moved out
    "first_time_payee",          # 1 if this payer has never paid this payee before
    "payer_on_call",             # 1 if the payer is on an active phone/video call while paying
]

LABEL = "is_fraud"

# Decision thresholds on risk_score (0-100)
WARN_AT = 40
BLOCK_AT = 75

PATHS = {
    "data": "data/transactions.csv",
    "model": "models/model.joblib",
    "metrics": "models/metrics.json",
}
