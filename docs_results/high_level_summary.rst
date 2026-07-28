High Level Summary of Results
=============================


Overall scores
--------------

We score the submissions based on how close they come to meeting requirements
for each of the assesment metrics for each simulation and scenario.  We assign
between 0 and 3 points for each tasket, metric, simulation and scenario combination,
total all the of the assigned points and divide by the possible points.  The
various points ranges are visible as progressively darker bands in the plots on this page.


.. sortable-table::
   :file: results/scores_summary.csv


Indvidual Metrics
-----------------

Marker shapes in the plots below indicate the type of method used by each
submission:

.. raw:: html

   <ul>
     <li>&#9679; Circle &mdash; template fitting</li>
     <li>&#9670; Diamond &mdash; machine learning</li>
     <li>&#9632; Square &mdash; mixed model</li>
     <li><span style="display:inline-block;width:0.9em;height:0.9em;border:2px solid black;border-radius:50%;vertical-align:middle;"></span> Black border &mdash; augmentation</li>
   </ul>

.. list-table:: Image Gallery
   :header-rows: 1

   * - Description
     - Plot
   * - **Mean of Estimated - True Redshift**
     - .. image:: results/plot_summary_point_mean.png
          :width: 600px
          :align: center
   * - **RMS of Estimated - True Redshift**
     - .. image:: results/plot_summary_point_rms.png
          :width: 600px
          :align: center
   * - **Outlier Fraction of Estimated - True Redshift**
     - .. image:: results/plot_summary_point_outliers.png
          :width: 600px
          :align: center
   * - **KS of PIT**
     - .. image:: results/plot_summary_pit_ks.png
          :width: 600px
          :align: center


.. list-table:: Image Gallery
   :header-rows: 1

   * - Description
     - Plot
   * - **Per-object estimation timing**
     - .. image:: results/plot_summary_timing_estimate.png
          :width: 600px
          :align: center
   * - **Total training timing**
     - .. image:: results/plot_summary_timing_inform.png
          :width: 600px
          :align: center


