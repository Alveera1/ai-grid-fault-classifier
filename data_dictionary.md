# Data Dictionary: PMU_Fault_Data.csv

Source: ________ (add the exact dataset name or link from your faculty or download page)
Raw size: 134,406 rows x 14 columns. Cleaned size: 134,019 rows x 14 columns.

| Column | Meaning (inferred from name and values) | Notes |
|---|---|---|
| time_stamp | Time in seconds | Restarts at 0 for every class. Not used as a feature. |
| Ia, Ib, Ic | Currents of phases A, B, C | Range about -47,000 to +45,000 |
| Va, Vb, Vc | Voltages of phases A, B, C | Range about -307 to +306 |
| Im | Current magnitude | Nearly constant inside each class |
| Vm | Voltage magnitude | Nearly constant inside each class |
| Pai | Phase angle of current (degrees) | To be confirmed with the dataset source |
| Pav | Phase angle of voltage (degrees) | To be confirmed with the dataset source |
| Fi | Frequency from current (Hz) | About 50 Hz in every class, little information |
| Fv | Frequency from voltage (Hz) | About 50 Hz in every class, little information |
| fault_types | Target label | NF, LG, LL, LLG, LLL, LLLG |

The column meanings are inferred from names and values. They must be confirmed with the faculty or the data source before the final report.

## Classes
| Class | Meaning | Rows (raw) |
|---|---|---|
| NF | No fault (normal) | 22,401 |
| LG | Line to ground | 22,401 |
| LL | Line to line | 22,401 |
| LLG | Two lines to ground | 22,401 |
| LLL | All three lines shorted | 22,401 |
| LLLG | Three lines to ground | 22,401 |

## Features used for modeling
Ia, Ib, Ic, Va, Vb, Vc, Im, Vm, Pai, Pav, Fi, Fv (time_stamp excluded).
