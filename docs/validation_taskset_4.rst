Validation figures for taskset 4
================================

The data preparation for taskset 4 including the following steps:

1. Starting with either the Cardinal or Flagship simulation truth information.
2. Rotating the field into an area covered by the LSST survey.
3. Selecting objects with a true :math:`i < 27.0`.
4. Blending objects within the Rubin PSF into single objects by
   combining their fluxes.  After the blending we only retain objects
   with :math:`i < 25.5`.  For blended objects, we assign the blend
   the true redshift of the brightest object in the blend.
5. Applying photometric smearing. In the Rubin bands this used expected observing
   conditions and depth maps for 1 and 10 years of observing.  In the Roman bands
   this used the expected depths for the medium tier of the
   High-Latitude wide area survey.
6. Emulating spectroscopic selections for the reference redshift creation.  For this taskset
   we emulate the area of the spectroscopic samples.  Specifically, we
   apply the DESI selections to the whole field, the zCOSMOS and
   COSMOS2020 selections to the same small area, and the other
   spectroscopic selections to different small areas.  This is
   intended to give roughly the number of objects we might expect in a
   joint reference redshift sample.
7. Drawing a training (100k objects) data set from the objects passing the spectroscopic selections
   and a test data sets (20k objects) from all the objects with and observed :math:`i < 25.5`.

.. container:: image-gallery

   .. image:: figures/pz_challenge_taskset_4_training_1yr/RA_DEC_footprint.jpg
      :alt: image
      :width: 45.0%

   .. image:: figures/pz_challenge_taskset_4_test_1yr/RA_DEC_footprint.jpg
      :alt: image
      :width: 45.0%

   .. image:: figures/pz_challenge_taskset_4_training_10yr/RA_DEC_footprint.jpg
      :alt: image
      :width: 45.0%
   
   .. image:: figures/pz_challenge_taskset_4_test_10yr/RA_DEC_footprint.jpg
      :alt: image
      :width: 45.0%

   Survey footprints for training (left) and test (right) data.  Within each side
   both 1 cardinal (left) and flagship (right) simulations are shown for
   both 1 year (top) and 10 year (bottom) data sets.


.. container:: image-gallery

   .. image:: figures/pz_challenge_taskset_4_training_1yr/mag_number_counts_Rubin.jpg
      :width: 45.0%

   .. image:: figures/pz_challenge_taskset_4_test_1yr/mag_number_counts_Rubin.jpg
      :width: 45.0%
	      
   .. image:: figures/pz_challenge_taskset_4_training_1yr/mag_number_counts_Roman.jpg
      :width: 45.0%

   .. image:: figures/pz_challenge_taskset_4_test_1yr/mag_number_counts_Roman.jpg
      :width: 45.0%
	       
   Number counts as a function of magnitude for Rubin (top) and Roman (bottom) bands
   for 1 year training (left) and test (right) data sets.


.. container:: image-gallery
   
   .. image:: figures/pz_challenge_taskset_4_training_10yr/gr_vs_ug_sidebyside.jpg
      :alt: image
      :width: 80.0%

   .. image:: figures/pz_challenge_taskset_4_training_10yr/ri_vs_gr_sidebyside.jpg
      :alt: image
      :width: 80.0%
   
   .. image:: figures/pz_challenge_taskset_4_training_10yr/iz_vs_ri_sidebyside.jpg
      :alt: image
      :width: 80.0%

   .. image:: figures/pz_challenge_taskset_4_training_10yr/zy_vs_iz_sidebyside.jpg
      :alt: image
      :width: 80.0%

   Color-color plots for 10 year training sets pairs of adjacent bands.



.. container:: image-gallery

   .. image:: figures/fig_ts4_flag_counts.png
      :alt: Spectroscopic survey flag counts
      :width: 80.0%

   Number of training objects selected by each spectroscopic survey flag
   (DEEP2, VVDSf02, zCOSMOS, COSMOS, DESI BGS/ELG/LRG) for Cardinal and
   Flagship 1-year training files. Taskset 4 training is 1-year only;
   no 10-year training data is provided.

.. container:: image-gallery

   .. image:: figures/fig_ts4_nz_by_flag.png
      :alt: N(z) by spectroscopic survey flag
      :width: 80.0%

   Normalised redshift distributions N(z) of training galaxies split by
   spectroscopic survey flag. The dashed purple curve shows N(z) derived
   from the many-band photometric redshift (``redshift_manyband``) for
   COSMOS-selected objects, whose true spectroscopic redshifts are not
   available.

..  LocalWords:  taskset
