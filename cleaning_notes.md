# Data cleaning summary (Member 1)

The raw file has 134,406 rows and 14 columns, with no missing values and 22,401 rows in each of the 6 classes.

Three cleaning steps were applied:
1. 17 fully duplicated rows were removed.
2. 370 warm-up rows were removed. These are placeholder rows at the start of each class run, where Im = 1, Vm = 1 and the phase angle is 0. They are not real fault measurements and would look like a fake extra pattern.
3. The data was sorted by class and time_stamp so that each class run is in time order, which the time-blocked split needs.

The cleaned file has 134,019 rows. The time-blocked split holds out the last 20 percent of every class run: 107,218 training rows and 26,801 test rows. A random row split is avoided because neighbouring samples are 0.3 ms apart and almost identical, which would leak test information into training and inflate the score.

time_stamp restarts at 0 in every class, so it is not used as a feature.
