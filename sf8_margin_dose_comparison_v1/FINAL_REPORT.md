# SF8 — margin dose comparison (control=0.0, low=0.25, medium=0.50; M=1.0 fixed)

**BOTH_DOSES_SUPPORTED**. Three fresh same-seed triplets (87017-87019); nine independent Pilot1 starts. SF7 (lambda=1.0) remains sealed and unmodified.

| Seed | Arm | Update | Correct/16 | Exact/16 | Rev/8 | Fam/4 | CE | PPL | D3 | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 87017 | control | 200 | 14 | 15 | 6 | 2 | 3.571193 | 35.559 | 0.00631864 | ACQUISITION_FAIL |
| 87017 | low | 200 | 16 | 16 | 8 | 4 | 3.587301 | 36.136 | 0.00703672 | ACQUISITION_SUCCESS |
| 87017 | medium | 200 | 16 | 16 | 8 | 4 | 3.563898 | 35.301 | 0.00814467 | ACQUISITION_SUCCESS |
| 87018 | control | 200 | 15 | 15 | 7 | 3 | 3.582028 | 35.946 | 0.00682939 | ACQUISITION_FAIL |
| 87018 | low | 200 | 16 | 16 | 8 | 4 | 3.574605 | 35.681 | 0.00667241 | ACQUISITION_SUCCESS |
| 87018 | medium | 200 | 16 | 16 | 8 | 4 | 3.571743 | 35.579 | 0.00718288 | ACQUISITION_SUCCESS |
| 87019 | control | 200 | 15 | 14 | 7 | 3 | 3.580296 | 35.884 | 0.00756697 | ACQUISITION_FAIL |
| 87019 | low | 200 | 16 | 16 | 8 | 4 | 3.594973 | 36.415 | 0.00783949 | ACQUISITION_SUCCESS |
| 87019 | medium | 200 | 16 | 16 | 8 | 4 | 3.582841 | 35.976 | 0.00898989 | ACQUISITION_SUCCESS |

## Dose summary /3

- **control**: successes=0/3, acquisition_fail=3/3, D3_stops=0/3, other_retention=0/3, endpoint@200=3/3
- **low**: successes=3/3, acquisition_fail=0/3, D3_stops=0/3, other_retention=0/3, endpoint@200=3/3
- **medium**: successes=3/3, acquisition_fail=0/3, D3_stops=0/3, other_retention=0/3, endpoint@200=3/3

## Same-seed triplets

- 87017: control=acquisition_fail, low=success, medium=success
- 87018: control=acquisition_fail, low=success, medium=success
- 87019: control=acquisition_fail, low=success, medium=success

## Mechanism (descriptive only; never a gate)

- seed_87017_control @200: residual=-0.0678, min g0:a0=-0.0678, mean g0:a0=0.0218, mean g0:a1=0.5301, mean g1=2.2530, D3=0.006319
  @100: residual=0.2417, min g0:a0=0.2417, D3=0.009335
- seed_87017_low @200: residual=3.2648, min g0:a0=3.2004, mean g0:a0=3.3083, mean g0:a1=4.9681, mean g1=4.8479, D3=0.007037
  @100: residual=1.1323, min g0:a0=1.1323, D3=0.006899
- seed_87017_medium @200: residual=4.5166, min g0:a0=4.5144, mean g0:a0=4.6794, mean g0:a1=5.4323, mean g1=4.9629, D3=0.008145
  @100: residual=1.7347, min g0:a0=1.5618, D3=0.009737
- seed_87018_control @200: residual=-0.0536, min g0:a0=-0.0536, mean g0:a0=0.0284, mean g0:a1=0.5052, mean g1=2.1007, D3=0.006829
  @100: residual=0.3432, min g0:a0=0.3432, D3=0.009253
- seed_87018_low @200: residual=3.6453, min g0:a0=3.6113, mean g0:a0=3.7606, mean g0:a1=4.8824, mean g1=5.0670, D3=0.006672
  @100: residual=0.9712, min g0:a0=0.9712, D3=0.008196
- seed_87018_medium @200: residual=4.3053, min g0:a0=4.3053, mean g0:a0=4.4316, mean g0:a1=5.6506, mean g1=5.0460, D3=0.007183
  @100: residual=1.3598, min g0:a0=1.3581, D3=0.008119
- seed_87019_control @200: residual=-0.0025, min g0:a0=-0.0160, mean g0:a0=0.0537, mean g0:a1=0.5846, mean g1=2.3710, D3=0.007567
  @100: residual=0.2972, min g0:a0=0.2972, D3=0.009810
- seed_87019_low @200: residual=3.4954, min g0:a0=3.3921, mean g0:a0=3.5201, mean g0:a1=5.0836, mean g1=5.1549, D3=0.007839
  @100: residual=0.7090, min g0:a0=0.6909, D3=0.008239
- seed_87019_medium @200: residual=4.3110, min g0:a0=4.3110, mean g0:a0=4.4286, mean g0:a1=5.5741, mean g1=5.1506, D3=0.008990
  @100: residual=1.9164, min g0:a0=1.9164, D3=0.009515

Transfer/copy/competing-name/FINAL/sacred panels remain locked and unscored. SF7 untouched.

Next action: review this completed dose comparison before any transfer evaluation or further treatment. No tenth run, no dose changes, no interim selection occurred.
