Validation figures for taskset 3
================================

The data preparation for taskset 3 including the following steps:

1. Starting with either the Cardinal or Flagship simulation truth information.
2. Rotating the field into an area covered by the LSST survey.
3. Selecting objects with a true :math:`i < 25.5`.
4. Applying photometric smearing. In the Rubin bands this used expected observing
   conditions and depth maps for 1 and 10 years of observing.  In the Roman bands
   this used the expected depths for the medium tier of the
   High-Latitude wide area survey.
5. Emulating spectroscopic selections for the reference redshift creation.  For this taskset
   we emulate the area of the spectroscopic samples.  Specifically, we
   apply the DESI selections to the whole field, the zCOSMOS and
   COSMOS2020 selections to the same small area, and the other
   spectroscopic selections to different small areas.  This is
   intended to give roughly the number of objects we might expect in a
   joint reference redshift sample.
6. Drawing a training (100k objects) data set from the objects passing the spectroscopic selections
   and a test data sets (20k objects) from all the objects with and observed :math:`i < 25.5`.

.. container:: image-gallery

   .. image:: figures/pz_challenge_taskset_3_training_1yr/RA_DEC_footprint.jpg
      :alt: image
      :width: 45.0%

   .. image:: figures/pz_challenge_taskset_3_test_1yr/RA_DEC_footprint.jpg
      :alt: image
      :width: 45.0%

   .. image:: figures/pz_challenge_taskset_3_training_10yr/RA_DEC_footprint.jpg
      :alt: image
      :width: 45.0%
   
   .. image:: figures/pz_challenge_taskset_3_test_10yr/RA_DEC_footprint.jpg
      :alt: image
      :width: 45.0%

   Survey footprints for training (left) and test (right) data.  Within each side
   both 1 cardinal (left) and flagship (right) simulations are shown for
   both 1 year (top) and 10 year (bottom) data sets.


.. container:: image-gallery

   .. image:: figures/pz_challenge_taskset_3_training_1yr/mag_number_counts_Rubin.jpg
      :width: 45.0%

   .. image:: figures/pz_challenge_taskset_3_test_1yr/mag_number_counts_Rubin.jpg
      :width: 45.0%
	      
   .. image:: figures/pz_challenge_taskset_3_training_1yr/mag_number_counts_Roman.jpg
      :width: 45.0%

   .. image:: figures/pz_challenge_taskset_3_test_1yr/mag_number_counts_Roman.jpg
      :width: 45.0%
	       
   Number counts as a function of magnitude for Rubin (top) and Roman (bottom) bands
   for 1 year training (left) and test (right) data sets.


.. container:: image-gallery
   
   .. image:: figures/pz_challenge_taskset_3_training_10yr/gr_vs_ug_sidebyside.jpg
      :alt: image
      :width: 80.0%

   .. image:: figures/pz_challenge_taskset_3_training_10yr/ri_vs_gr_sidebyside.jpg
      :alt: image
      :width: 80.0%
   
   .. image:: figures/pz_challenge_taskset_3_training_10yr/iz_vs_ri_sidebyside.jpg
      :alt: image
      :width: 80.0%

   .. image:: figures/pz_challenge_taskset_3_training_10yr/zy_vs_iz_sidebyside.jpg
      :alt: image
      :width: 80.0%

   Color-color plots for 10 year training sets pairs of adjacent bands.

.. container:: image-gallery
   
   .. image:: figures/pz_challenge_taskset_3_training_1yr/sigma_nmad_vs_mag_i.jpg
      :alt: image
      :width: 80.0%

   .. image:: figures/pz_challenge_taskset_3_training_1yr/foutlier_vs_mag_i.jpg
      :alt: image
      :width: 80.0%

   Normalized median absolute deviation (left) and outlier fraction (right) of 
   emulated mock COSMOS2020 photometric redshifts as a function of i-band 
   magnitude for 1 and 10 year training data sets. 

..  LocalWords:  taskset
