# Benford's Law
Anomaly Detection in accounting transactions using Benford's Law.

## Implemented as per:

* "The Use of Benford's Law as an Aid in Analytical Procedures" (Mark J. Nigrini and Linda J. Mittermaier)
* "Using Digital Frequencies to Detect Anomalies in Receivables and Payables" (Fabio Ciaponi, Francesca Mandanici)

## CONTENTS
* benfords_law.py - source code
* transactions_real.csv - list of transactions we'd like to check for anomalies 
* transactions_to_investigate.csv - table of transactions (and associated rules) that violate Benford's Law at the specified alpha level
* first_digit_plot.png, second_digit_plot.png - Expected vs. Observed frequency plots
