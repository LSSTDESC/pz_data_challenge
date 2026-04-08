# Introduction

Redshift inference is a key measurement for multiple DESC science goals
and redshift uncertainty is one of the leading contributors to overall
uncertainty on cosmological models from imaging survey data. Precursor
surveys took a variety of approaches to this problem, accounting for
differences in underlying data as well as modeling approaches. In all
cases, redshift uncertainty was significantly larger than DESC Science
Requirements listed in the LSST DESC Science Requirements Document.

This state-of-the-art motivates a data challenge to characterize and
improve existing methods, as well as provide infrastructure for the
development of improved methods. Overall, this requires generating
uniform input catalogs to use and infrastructure for comparing output
redshift posteriors to each other and to simulated truth catalogs.

# Challenge Format

The PZ data challenge comprises a set of tasks for participants. These
task will be used to evaluate how ready various algorithms are to be
used for cutting edge analysis. Readiness will be evaluated on a few
different fronts. 1) Does the algorithm meet performance requirements?
2) Is it robust, flexible and relatively easy to use on different
datasets? 3) Is it scalable up to the scales we will need to use it at.

This document and the associated web pages describe the data being
provided to participants, the task they will be asked to perform, and
the metrics by which the algorithms readiness will be evaluated.

## Scope and Timeline

The data challenge will include two major parts, with a set of task
emulating increasingly realistic scenarios in each part. The first part,
$p(z)$ estimation, will focus on estimating the redshift of individual
objets. The second part, tomogrpahy and $n(z)$ estimation will focus on
assigning object to tomographic bins and estimating the distribution of
redshifts in each bin.

The data challenge will run from April 13, 2026 to August 7, 2026. The
first set of data and tasks related to $p(z)$ estimation will be
released on April 13. A second set of data and tasks related to $n(z)$
estimation will be released on June 1.

Preliminary results will be released on August 14, 2026, with a
techincal note summarize those result to follow shortly thereafter and a
comprehensive journal publication to follow later.

## Challgenge Data

The preparation of the challenge data is described in the appendices.
The data is available as `tar` archives on the data challenge site.

Each task set in the data challenge has an associated set of files.
Typically these will be a collection of training files that contain
photometric data and reference redsfhits and a second set of files that
contain photometric data but do not include redshifts. Each task set
will invovle estimating something about the redshifts or redshift
distributions in the test files.

Typically there will be several training and test files for a particular
test set, covering different scenarios and using different input
simulations.

## Challenge Tasks and Submission Types

The challenge is organized as a series of sets tasks using increasingly
realistic representations of the data. In general, each set of task
includes 3 tasks.

1.  Estimate either per-object $p(z)$ or ensemble $n(z)$ distributions
    for a set of different scenarios and provide the estimates in a
    specfied format.

2.  Provide trained models for the different scenarios, and a generic
    script that can be used generate the estimates from task 1 on an
    arbitrary dataset.

3.  Provide a generic script that can be used generate the generate the
    models and estimates from tasks 1 and 2 on arbitrary datasets.

## Input data format

The input data for the challenge are presented in hdf5 files. The naming
convention for the files is
`{challenge}_{taskset}_{simulation}_{label}_{scenario}.hdf5`. The
meaning for the various fields are descrbied in
Tab. [1](#tab:file_fields){reference-type="ref"
reference="tab:file_fields"}. The columns in the files are described in
Tab. [2](#tab:columns){reference-type="ref" reference="tab:columns"}.

*Eric:*

::: {#tab:file_fields}
  field        Description
  ------------ --------------------------------------
  challenge    Challenge associated to file
  taskset      Task set associated to file
  simulation   Simulation used to produce file
  label        File label (e.g., "test", "training"
  scenario     Data Scenario

  : Fields in .
:::

::: {#tab:columns}
  Column                   Description
  ------------------------ ---------------------------------------
  redshift                 True redshift (training files only)
  ra                       Right Accension (training files only)
  dec                      Declination (training files only)
  object_id                Unique Object ID
  mag\_{band}\_lsst        Magnitude in LSST {band}
  mag\_{band}\_lsst_err    Magnitude uncertainty in LSST {band}
  mag\_{band}\_roman       Magnitude in Roman {band}
  mag\_{band}\_roman_err   Magnitude uncertainty in Roman {band}

  : Contents of input files.
:::

## Data format for per-object $p(z)$ estimates

The $p(z)$ estimates should be submitted as in `qp` format, which allows
users to specific a complete $p(z)$ distribution for each object, as
well as summary statistics for each object.

The `qp` packages supports several different representation of $p(z)$,
such as different functional forms as well as interpolated grids,
histograms, and others.

For users unfamiliar with `qp` we highly recommend representing the
$p(z)$ either an interpolated grid, or a Gaussian mixture model.

    # Interpolated grid
    import qp
    import numpy as np
    # Define the x-grid.  Note that we put all the
    # p(z) on the same x-grid
    xvals = np.array([0,0.5,1,1.5,2])
    # Define the y-values.  Note we provide n_grid_points x n_objects 
    # values, as we need to provide a y-value at each grid point 
    # for each object.
    yvals = np.array(
     [
       [0.01,0.2,0.3,0.2,0.01],
       [0.1,0.3,0.5,0.2,0.05]
     ]
    )
    ens = qp.interp.create_ensemble(xvals,yvals)

    # Mixture model
    import qp
    import numpy as np
    # Define the means, standard deviations and weights.
    # These should each have shape n_objectws, n_components.
    # In this case we are defining 3 objects with 2-Gaussian 
    # representations.
    # For each object the weights should sum to 1, or they
    # will be normalized.
    means = [[0.3, 0.4], [0.5, 0.5], [0.6, 0.8]]
    stds = [[0.2, 0.4], [0.1, 0.3], [0.05, 0.3]]
    weights = [[0.8, 0.2], [0.7, 0.3], [0.8, 0.2]]
    ens = qp.mixmod.create_ensemble(means=means,stds=stds,weights=weights)

The submission files should use the same file name conventions defined
in Tab. [1](#tab:file_fields){reference-type="ref"
reference="tab:file_fields"}. The labels will typically be `pz_estimate`
or `pz_model`, and will be specified in the descriptions of the various
tasks, e.g., `pz_challenge_taskset_1_cardinal_pz_estimate_yr1.hdf5` or
`pz_challenge_taskset_1_cardinal_pz_model_yr1.pkl`.

## Format for estimation only scripts and trained models

For the second sub-task, providing trained models and a script to run
estimation using those trained models, submitters should start with the
`run_estimate.sh` example script, and modify the portions of the script
to install their packages, download their models, and run the estimation
on the appropriate test files.

Once they have tested the script, submitters should open a pull request
to add `run_estimate.sh`

## Format for training and estimation scripts

For the second sub-task, providin a script to train models and run
estimation using those trained models, submitters should start with the
`run_train_and_estimate.sh` example script and modify the portions of
the script to install their packages, download their models, and run the
estimation on the appropriate test files.

# Metrics and Assesment Criteria

## Metrics for per-object point-estimates

*Eric:*

## Metrics for per-object $p(z)$ distributions

*Eric:*

## Metrics for per-object estimation computational performance

*Eric:*

# Challenge Tasks related to $p(z)$ estimation {#sec:tasks}

## Task set 1: estimate redshifts using representative training samples

The first, simplest, task is to estaimate redshifts using representative
training samples. I.e., the training samples are drawn for the same
distibutions as the test samples. For this task set we did not use an of
the spectroscopic selection emulation, but simply applied a uniform
magnitude cut $i < 23$ in selecting objects for both the training and
test samples.

The four `pz_challenge_taskset_1_{simulation}_training_{scenario}.hdf5`
files are the training sets for the "Flagship" and "Cardinal"
simulations, and emulating 1 year and 10 years of LSST data under
expected observing strategy and conditions. These files have true
redshifts to serve as labels.

The corresponding
`pz_challenge_taskset_1_{simulation}_test_{scenario}.hdf5` files were
drawn from the same distributions. The true redshifts have been removed
from these files. The task is to assign $p(z)$ estimates for all the
objects in thse 4 test files.

## Task set 2: estimate redshifts on non-representative samples

The second, slightly more challenging, task is to estimate redshifts
using non-representative training samples. I.e., the training samples
are not drawn for the same distibutions as the test samples. For this
task set we applied the spectroscopic selection emulation for the train
set, but retained all the objects down to $i < 25.4$ in the test set.
Accordingly, the training set will not be representative of the fainter
objects in the test set. This reflects that spectroscopic redshifts are
typically significantly more diffiicult to obtain than photometry.

The four `pz_challenge_taskset_1_{simulation}_training_{scenario}.hdf5`
files are the training sets for the "Flagship" and "Cardinal"
simulations, and emulating 1 year and 10 years of LSST data under
expected observing strategy and condition and with spectroscopic
selections emulated.

The corresponding
`pz_challenge_taskset_1_{simulation}_test_{scenario}.hdf5` files were
drawn from the distributions of all objects down to $i <
25.4$, and the true redshifts have been removed from these files. The
task is to assign $p(z)$ estimates for all the objects in thse 4 test
files.

# Input simulations

The challenge employs simulated galaxy catalogs derived from two
complementary N-body cosmological simulations: the Cardinal simulations
and the Flagship simulation. These synthetic datasets provide a
controlled environment where the true redshifts are known by
construction, enabling rigorous validation of photometric redshift
algorithms and systematic assessment of their performance
characteristics.

The Cardinal simulations comprise a suite of high-resolution N-body
simulations specifically designed to explore the sensitivity of
cosmological observables to variations in fundamental cosmological
parameters. The simulations employ state-of-the-art semi-analytic models
to populate dark matter halos with galaxies, incorporating realistic
prescriptions for star formation, dust attenuation, and spectral energy
distribution modeling.

The Flagship simulation represents a single, ultra-large cosmological
simulation run with fiducial cosmological parameters consistent with
current observational constraints. With a volume exceeding several cubic
gigaparsecs, the Flagship provides statistical power to probe rare
objects and the high-mass end of the galaxy population. Its primary
purpose in the photometric redshift challenge is to provide a realistic
mock catalog that captures the full complexity of galaxy populations
across cosmic time, including correlations between galaxy properties,
environmental dependencies, and the intricate relationships between
spectral features and redshift.

Together, these complementary simulation suites enable challenge
participants to test both the accuracy and the robustness of their
photometric redshift estimation methods under realistic observational
conditions.

# Emulating observational effects

To bridge the gap between the idealized simulation outputs and realistic
survey observations, we employ the RAIL (Redshift Assessment
Infrastructure Layers) software package to emulate observational
effects. RAIL provides a modular framework for injecting realistic
photometric uncertainties, applying survey-specific selection functions,
and simulating the measurement errors characteristic of modern
large-scale imaging surveys. This processing ensures that the simulated
galaxy catalogs reflect the complexities of actual observations,
including magnitude-dependent photometric scatter, incomplete sky
coverage, and the effects of source blending in crowded fields, thereby
providing a more stringent and realistic test bed for photometric
redshift estimation algorithms.

## Photometric Smearing

Central to our observational emulation is RAIL's wraping of the
photometric error module, photErr, which we have extended and wrapped to
account for realistic observing strategies and time-dependent survey
conditions. The standard photErr module provides basic photometric error
modeling based on magnitude-dependent noise characteristics, but our
enhanced version incorporates additional complexity including
spatially-varying depth maps. This wrapper accesses detailed operational
simulation outputs that emulate the expected LSST survey strategy.

Our photErr implementation computes photometric uncertainties by
combining the intrinsic Poisson noise from source photons with realistic
models of sky background, readout noise, and other systematic
contributions. For each simulated galaxy, use the expected co-addition
depth to derive final photometric error estimates. This approach
captures the heterogeneous nature of survey depth across the footprint,
where some regions benefit from numerous high-quality exposures while
others may be observed only during poor conditions. The resulting
photometric uncertainties vary realistically with position on the sky,
band-dependent limiting magnitudes, and local observing history,
providing challenge participants with mock catalogs whose noise
properties more closely match those expected from the actual survey.

## Spectroscopic and narrow-band photo-metric redshift selection

RAIL can emulate the selection functions of several different
spectroscopic redshift surverys, including VVDSf02, zCOSMOS, DEEP2_LSST,
and the DESI BGS, ELG and LRG samples.

We can also use RAIL to emulate narrow-band photometric surveys, include
small amounts of mis-labeled reference redshifts.

# Preparing Training and Test and Reserved datasets

All of the data prepration was performed using the `rail_projects` and
`rail_package_config` packages for bookkeeping and reproducibility.

::: {#tab:prep_scripts}
  Script            Command Run                          Purpose
  ----------------- ------------------------------------ -------------------------------------
  do_00_reduce      rail-project reduce                  Reduce input truth catalogs
                                                         (mag. cut and drop columns)
  do_01_build       rail-project build                   Build configurations to run
                                                         truth-to-observed pipeline
  do_02_t2o         rail-project run truth-to-observed   Run truth-to-observed
                                                         pipelines to make degraded catalogs
  do_03_merge       rail-project merge                   Combine spectroscopic selections
  do_04_subselect   rail-project subsample               Make train/test files
  from catalogs                                          

  : Scripts used in data preparation.
:::
