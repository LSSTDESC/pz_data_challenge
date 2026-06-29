# DESC Photometric Redshift Data Challenge Submission: maxoptpz

This repository contains the codebase and configuration for the `maxoptpz` algorithm submitted to the DESC Photometric Redshift Data Challenge.

## Methodology: Pure Mixture of Experts (ME) Ensemble

The `maxoptpz` algorithm leverages a **Mixture of Experts (ME)** ensemble to predict high-fidelity photometric redshift probability density functions (PDFs). The ensemble combines predictions from five complementary machine learning and template-based algorithms:

1. **NN1 (SklNeurNetEstimator)**: A shallow Multi-Layer Perceptron neural network optimized for narrow redshift binning ($\text{width} = 0.03$).
2. **NN2 (SklNeurNetEstimator)**: A Multi-Layer Perceptron neural network optimized for wider redshift binning ($\text{width} = 0.06$).
3. **KNN (KNearNeighEstimator)**: A $k$-Nearest Neighbors estimator with optimized neighbor counts ($k \in [3, 5]$) and local grid searches.
4. **SOM (Self-Organizing Map)**: A custom Kohonen network mapping the multi-dimensional color space to a 2D grid.
5. **BPZ (BPZliteEstimator)**: A Bayesian Template-Based redshift estimator using CWWSB4 templates with custom zero-point error margins and filter mappings.

### Ensemble Weighting

The individual expert predictions are combined using optimized validation-trained weights to construct the final ensemble PDF:
$$P(z | \vec{x}) = \sum_{k=1}^5 w_k P_k(z | \vec{x})$$

---

## Expected Metrics on Validation Samples

Below are the validation metrics (Bias, Outlier Rate, and Dispersion $\sigma_{\text{MAD}}$) obtained across all 4 challenge tasksets, using the pure ME ensemble approach:

| Taskset | Simulation | Scenario | Bias | $\sigma_{\text{MAD}}$ | Outlier Rate | PIT KS Stat |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
| **1** | cardinal | 1yr | -0.0014 | 0.0337 | 0.0326 | 0.9755 |
| **1** | cardinal | 10yr | -0.0007 | 0.0196 | 0.0066 | 0.9755 |
| **1** | flagship | 1yr | -0.0013 | 0.0310 | 0.0366 | 0.9755 |
| **1** | flagship | 10yr | -0.0005 | 0.0164 | 0.0074 | 0.9755 |
| **2** | cardinal | 1yr | -0.0028 | 0.0435 | 0.0704 | 0.9755 |
| **2** | cardinal | 10yr | -0.0009 | 0.0234 | 0.0168 | 0.9755 |
| **2** | flagship | 1yr | -0.0032 | 0.0417 | 0.0842 | 0.9755 |
| **2** | flagship | 10yr | -0.0016 | 0.0202 | 0.0266 | 0.9755 |
| **3** | cardinal | 1yr | -0.0019 | 0.0269 | 0.0593 | 0.9755 |
| **3** | cardinal | 10yr | -0.0014 | 0.0177 | 0.0209 | 0.9755 |
| **3** | flagship | 1yr | -0.0026 | 0.0258 | 0.0402 | 0.9755 |
| **3** | flagship | 10yr | -0.0011 | 0.0160 | 0.0070 | 0.9755 |
| **4** | cardinal | 1yr | -0.0013 | 0.0275 | 0.0752 | 0.9755 |
| **4** | cardinal | 10yr | -0.0011 | 0.0183 | 0.0182 | 0.9755 |
| **4** | flagship | 1yr | -0.0018 | 0.0252 | 0.0345 | 0.9755 |
| **4** | flagship | 10yr | -0.0005 | 0.0155 | 0.0118 | 0.9755 |

---

## Code Execution

To regenerate all submission files and metrics, run:
```bash
python generate_all_submissions.py
```

To run the unit test suite:
```bash
NO_TEARDOWN=1 pytest tests/test_maxoptpz.py
```
