| Memory layer | Learning rate | Runs (seed: recall) | Mean recall |
|---|---|---|---|
| deltanet | 0.001 | s1: 1.8%, s2: 45.2% | 23.5% |
| deltanet | 0.002 | s1: 2.1%, s2: 77.6% | 39.8% |
| linear | 0.001 | s1: 89.4%, s2: 90.8% | 90.1% |
| linear | 0.002 | s1: 97.1%, s2: 98.2% | 97.7% |
| mesa_cg5 | 0.001 | s1: 1.9%, s2: 69.9% | 35.9% |
| mesa_cg5 | 0.002 | s1: 99.1%, s2: 99.6% | 99.4% |
| mesa_exact | 0.001 | s1: 50.1%, s2: 60.1% | 55.1% |
| mesa_exact | 0.002 | s1: 98.9%, s2: 99.5% | 99.2% |

Best learning rate per layer (mean over seeds):
    mesa_cg5:  99.4%  (lr 0.002, range 99.1-99.6%)
  mesa_exact:  99.2%  (lr 0.002, range 98.9-99.5%)
      linear:  97.7%  (lr 0.002, range 97.1-98.2%)
    deltanet:  39.8%  (lr 0.002, range 2.1-77.6%)
