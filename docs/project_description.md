# PayeeShield: project description

## One-line pitch

PayeeShield checks the account receiving a UPI payment, before you confirm, and stops the money if that account looks like a mule.

## Description (about 150 words)

UPI checks a payee's SIM only once, at registration. Our team lead sent ₹1 to a stranger's number: the payment went through, but the number was out of service. Scam accounts work this way every day, and money sent to them is usually moved out within minutes.

PayeeShield scores each payment in real time, before the payer confirms. It combines signals about the receiving account (is the SIM reachable, how old the SIM and UPI ID are, recent SIM swaps, how many different people paid it in 24 hours, how fast money leaves it) with the payment context (first-time payee, payer on a call, as in "digital arrest" scams). It returns a 0–100 risk score, an ALLOW, WARN or BLOCK decision, and plain-language reasons the payer can understand.

It is designed to run inside the payment path on IBM Z / LinuxONE. The prototype uses synthetic data and has not yet been benchmarked on IBM Z.

## Short version (about 60 words)

UPI checks a payee's SIM only at registration, so dead numbers keep receiving money. PayeeShield scores each payment before the payer confirms, using signals about the receiving account (SIM reachability, account age, SIM swaps, sender count, outflow speed) and the payment context. It returns a risk score, ALLOW/WARN/BLOCK, and plain reasons. It is designed for in-transaction scoring on IBM Z.
