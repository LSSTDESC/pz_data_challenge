Validation figures for taskset 1
================================

The data preparation for taskset 1 including the following steps:

1. Starting with either the Cardinal or Flagship simulation truth information.
2. Rotating the field into an area covered by the LSST survey.
3. Selecting objects with a true :math:`i < 25.5`.
4. Applying photometric smearing. In the Rubin bands this used expected observing
   conditions and depth maps for 1 and 10 years of observing.  In the Roman bands
   this used the expected depths for the medium tier of the High-Latitude wide area survey.
5. Drawing training (100k objects) and test (20k objets) data sets from the catalogs, requiring
   :math:`i < 23.5` for both data sets.   

.. container:: image-gallery

   .. image:: figures/pz_challenge_taskset_1_training_1yr/RA_DEC_footprint.jpg
      :alt: image
      :width: 45.0%

   .. image:: figures/pz_challenge_taskset_1_test_1yr/RA_DEC_footprint.jpg
      :alt: image
      :width: 45.0%

   .. image:: figures/pz_challenge_taskset_1_training_10yr/RA_DEC_footprint.jpg
      :alt: image
      :width: 45.0%
   
   .. image:: figures/pz_challenge_taskset_1_test_10yr/RA_DEC_footprint.jpg
      :alt: image
      :width: 45.0%

   Survey footprints for training (left) and test (right) data.  Within each side
   both 1 cardinal (left) and flagship (right) simulations are shown for
   both 1 year (top) and 10 year (bottom) data sets.


.. container:: image-gallery

   .. image:: figures/pz_challenge_taskset_1_training_1yr/mag_number_counts_Rubin.jpg
      :width: 45.0%

   .. image:: figures/pz_challenge_taskset_1_test_1yr/mag_number_counts_Rubin.jpg
      :width: 45.0%
	      
   .. image:: figures/pz_challenge_taskset_1_training_1yr/mag_number_counts_Roman.jpg
      :width: 45.0%

   .. image:: figures/pz_challenge_taskset_1_test_1yr/mag_number_counts_Roman.jpg
      :width: 45.0%
	       
   Number counts as a function of magnitude for Rubin (top) and Roman (bottom) bands
   for 1 year training (left) and test (right) data sets.


.. container:: image-gallery
   
   .. image:: figures/pz_challenge_taskset_1_training_10yr/gr_vs_ug_sidebyside.jpg
      :alt: image
      :width: 80.0%

   .. image:: figures/pz_challenge_taskset_1_training_10yr/ri_vs_gr_sidebyside.jpg
      :alt: image
      :width: 80.0%
   
   .. image:: figures/pz_challenge_taskset_1_training_10yr/iz_vs_ri_sidebyside.jpg
      :alt: image
      :width: 80.0%

   .. image:: figures/pz_challenge_taskset_1_training_10yr/zy_vs_iz_sidebyside.jpg
      :alt: image
      :width: 80.0%

   Color-color plots for 10 year training sets pairs of adjacent bands.



..  LocalWords:  taskset
