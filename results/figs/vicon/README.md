<h5>Experiment 4 (2017-11-22-11-25-20)</h5>

<p align="justify">When carefully tracked starting from the initial stride, one can see that the 10<sup>th</sup> stride is not detected in the trajectory plot shown below. To detect the missed ZV interval(s), supplementary ZUPT detectors such as VICON, ARED, MBGTD or AMVD can be utilized. In general, VICON detector was able to generate ZV labels correctly; therefore, in many cases, only VICON ZUPT detector is used as the supplementary detector.</p>

<!--
<img src="results/figs/vicon_obsolete/exp4.jpg" alt="Optimal detector results for experiment 4 (2017-11-22-11-25-20) VICON dataset" width=%100 height=auto>
-->

<img src="../../../data/vicon/processed/experiment4_ZUPT_detectors_strides.png" alt="ZV labels for experiment 4 (2017-11-22-11-25-20) VICON dataset" width=%100 height=auto>

<p align="justify">Integration of filtered optimal ZUPT detector SHOE with the filtered supplementary ZUPT detector (i.e., VICON) enabled successful detection of the ZV interval as shown in the combined ZUPT detector plot above (located at the bottom). The corrected ground-truth data (as a sample-wise and a stride & heading system trajectory) and ZV signals can be seen below. Note that the annotation is only going to be used in extracting x-y axes displacement (or displacement and heading change) values for LLIO training dataset generation; therefore, corrected ZV labels are not used in any trajectory generation.</p>

<!--
<img src="results/figs/vicon_obsolete/exp4_corrected.jpg" alt="Experiment trajectory after corrections - VICON dataset" width=%100 height=auto>
-->

<img src="../../../results/figs/vicon_obsolete/gif/exp4.gif" alt="ZV correction results" width=%100 height=auto>

<!--
<img src="results/figs/vicon_obsolete/stride_detection_exp_4.png" alt="Stride detection results on imu data - VICON dataset">
-->

<h5>Experiment 6 (2017-11-22-11-26-46)</h5>

<p align="justify">We see that the 9<sup>th</sup> stride is not detected in the plots below.</p>

<!---
<img src="results/figs/vicon_obsolete/exp6.jpg" alt="optimal detector results for experiment 6 (2017-11-22-11-26-46) VICON dataset" width=%100 height=auto>
--->

<p align="justify">Just like the way we compensated for the errors in ZV interval and stride detection in experiment 4, here VICON ZUPT detector is selected again as the supplementary detector to correctly detect the missed ZV interval and the stride.</p>

<img src="../../../data/vicon/processed/experiment6_ZUPT_detectors_strides.png" alt="ZV labels for experiment 6 (2017-11-22-11-26-46) VICON dataset" width=%100 height=auto>

<p align="justify">Integration of filtered optimal ZUPT detector SHOE with the supplementary ZUPT detector (i.e., filtered VICON) enabled successfull detection of the missed stride as shown in the combined ZUPT detector plot above (located at the bottom). The corrected stride & heading system trajectory and ZV labels can be seen below for the experiment 6.</p>

<!---
<img src="results/figs/vicon_obsolete/exp6_corrected.jpg" alt="corrected results for experiment 6 (2017-11-22-11-26-46) VICON dataset - trajectory" width=%100 height=auto>
--->

<p align="justify">To see the correction by the supplementary ZUPT detector, check the <b>gif</b> file below.</p>

<img src="../../../results/figs/vicon_obsolete/gif/exp6.gif" alt="experiment 6 results after ZV correction" width=%100 height=auto>

<!---
<img src="results/figs/vicon_obsolete/stride_detection_exp_6.png" alt="stride detection results on imu data for experiment 6 of VICON dataset">
--->

<h4>Experiment 11 (2017-11-22-11-35-59)</h4>

<p align="justify">We see that the 7<sup>th</sup> stride is not detected in the plots below.</p>

<!---
<img src="results/figs/vicon_obsolete/exp11.jpg" alt="optimal detector results for experiment 11 (2017-11-22-11-35-59) VICON dataset" width=%100 height=auto>
--->

<p align="justify">Just like we compensated for the errors in ZUPT phase and stride detection in experiments 4 and 6, here VICON ZUPT detector is selected again as the supplementary detector to correctly detect the missed stride.</p>

<img src="../../../data/vicon/processed/experiment11_ZUPT_detectors_strides.png" alt="ZV labels for experiment 11 (2017-11-22-11-35-59) VICON dataset" width=%100 height=auto>

<p align="justify">Integration of filtered optimal ZUPT detector SHOE with the supplementary ZUPT detector (i.e., filtered VICON) enabled successfull detection of the missed stride as shown in the combined ZUPT detector plot above (located at the bottom). The corrected stride & heading system trajectory and ZV labels can be seen below.</p>

<!---
<img src="results/figs/vicon_obsolete/exp11_corrected.jpg" alt="corrected results for experiment 11 (2017-11-22-11-35-59) VICON dataset" width=%100 height=auto>
--->

<p align="justify">To see the correction by the supplementary ZUPT detector, check the gif file below.</p>

<img src="../../../results/figs/vicon_obsolete/gif/exp11.gif" alt="experiment 11 results after ZV correction" width=%100 height=auto>

<!---
<img src="results/figs/vicon_obsolete/stride_detection_exp_11.png" alt="stride detection results on imu data for experiment 11 of VICON dataset">
--->

<h4>Experiment 30 (2017-11-27-11-14-03)</h4>

<p align="justify">We see that the strides {2, 10} are not detected in the plots below.</p>

<!---
<img src="results/figs/vicon_obsolete/exp30.jpg" alt="optimal detector results for experiment 30 (2017-11-27-11-14-03) VICON dataset" width=%100 height=auto>
--->

<p align="justify">Unlike experiments {4, 6, 11, 18, 27}, here SHOE ZUPT detector is selected as the supplementary detector to correctly detect the missed strides.</p>

<img src="../../../data/vicon/processed/experiment30_ZUPT_detectors_strides.png" alt="ZV labels for experiment 30 (2017-11-27-11-14-03) VICON dataset" width=%100 height=auto>

<p align="justify">Integration of filtered optimal ZUPT detector VICON with the supplementary ZUPT detector (i.e., filtered SHOE) enabled successfull detection of the missed stride as shown in the combined ZUPT detector plot above (located at the bottom). The corrected stride & heading system trajectory and ZV labels can be seen below for the experiment 30.</p>

<!---
<img src="results/figs/vicon_obsolete/exp30_corrected.jpg" alt="corrected results for experiment 30 (2017-11-27-11-14-03) VICON dataset" width=%100 height=auto>
--->

<p align="justify">To see the correction by the supplementary ZUPT detector, check the gif file inserted below.</p>

<img src="../../../results/figs/vicon_obsolete/gif/exp30.gif" alt="experiment 30 results after ZV correction" width=%100 height=auto>

<!---
<img src="results/figs/vicon_obsolete/stride_detection_exp_30.png" alt="stride detection results on imu data for experiment 30 of VICON dataset">
--->

<h4>Experiment 32 (2017-11-27-11-17-28)</h4>

<p align="justify">We see that the strides {9, 11, 20} are not detected in the plots below.</p>

<!---
<img src="results/figs/vicon_obsolete/exp32.jpg" alt="optimal detector results for experiment 32 (2017-11-27-11-17-28) VICON dataset" width=%100 height=auto>
--->

<p align="justify">Unlike experiments {4, 6, 11, 18, 27, 30}, here supplementary detectors were not able to detect all missed strides. While first two was recovered by VICON ZV detector, the last stride needed to be introduced via manual annotation as can be seen below.</p>

<img src="../../../data/vicon/processed/experiment32_ZUPT_detectors_strides.png" alt="ZV labels for experiment 32 (2017-11-27-11-17-28) VICON dataset" width=%100 height=auto>

<p align="justify">Integration of filtered optimal ZUPT detector SHOE with the supplementary ZUPT detector (i.e., filtered VICON) and the MANUAL ANNOTATION enabled successfull detection of all missed strides as shown in the combined ZUPT detector plot above (located at the bottom). The corrected stride & heading system trajectory and ZV labels can be seen below for the experiment 32.</p>

<!---
<img src="results/figs/vicon_obsolete/exp32_corrected.jpg" alt="corrected results for experiment 32 (2017-11-27-11-17-28) VICON dataset" width=%100 height=auto>
--->

<p align="justify">To see the correction by the supplementary ZUPT detector, check the gif file inserted below.</p>

<img src="../../../results/figs/vicon_obsolete/gif/exp32.gif" alt="experiment 32 results after ZV correction" width=%100 height=auto>

<!---
<img src="results/figs/vicon_obsolete/stride_detection_exp_32.png" alt="stride detection results on imu data for experiment 32 of VICON dataset">
--->

<h4>Experiment 36 (2017-11-27-11-23-18)</h4>

<p align="justify">We see that the 7<sup>th</sup> stride is not detected in the plots below.</p>

<!---
<img src="results/figs/vicon_obsolete/exp36.jpg" alt="optimal detector results for experiment 36 (2017-11-27-11-23-18) VICON dataset" width=%100 height=auto>
--->

<p align="justify">Just like the 4<sup>th</sup> experiment, here the supplementary detector is selected as VICON, which was able to recover the missed stride.</p>

<img src="../../../data/vicon/processed/experiment36_ZUPT_detectors_strides.png" alt="ZV labels for experiment 36 (2017-11-27-11-23-18) VICON dataset" width=%100 height=auto>

<p align="justify">Integration of filtered optimal ZUPT detector SHOE with the supplementary ZUPT detector (i.e., filtered VICON) enabled successfull detection of the missed stride as shown in the combined ZUPT detector plot above (located at the bottom). The corrected stride & heading system trajectory and ZV labels can be seen below for the experiment 36.</p>

<!---
<img src="results/figs/vicon_obsolete/exp36_corrected.jpg" alt="corrected results for experiment 36 (2017-11-27-11-23-18) VICON dataset" width=%100 height=auto>
--->

<p align="justify">To see the correction by the supplementary ZUPT detector, check the gif file below.</p>

<img src="../../../results/figs/vicon_obsolete/gif/exp36.gif" alt="experiment 36 results after ZV correction" width=%100 height=auto>

<!---
<img src="results/figs/vicon_obsolete/stride_detection_exp_36.png" alt="stride detection results on imu data for experiment 36 of VICON dataset">
--->

<h4>Experiment 38 (2017-11-27-11-25-12)</h4>

<p align="justify">We see that the strides {3, 27, 33} are not detected in the plots below.</p>

<!---
<img src="results/figs/vicon_obsolete/exp38.jpg" alt="optimal detector results for experiment 38 (2017-11-27-11-25-12) VICON dataset" width=%100 height=auto>
--->

<p align="justify">The supplementary detector is selected as VICON, which was able to recover the missed strides all.</p>

<img src="../../../data/vicon/processed/experiment38_ZUPT_detectors_strides.png" alt="ZV labels for experiment 38 (2017-11-27-11-25-12) VICON dataset" width=%100 height=auto>

<p align="justify">Integration of filtered optimal ZUPT detector SHOE with the supplementary ZUPT detector (i.e., filtered VICON) enabled successfull detection of the missed strides as shown in the combined ZUPT detector plot above (located at the bottom). The corrected stride & heading system trajectory and ZV labels can be seen below for the experiment 38.</p>

<!---
<img src="results/figs/vicon_obsolete/exp38_corrected.jpg" alt="corrected results for experiment 38 (2017-11-27-11-25-12) VICON dataset" width=%100 height=auto>
--->

<p align="justify">To see the correction by the supplementary ZUPT detector, check the gif file inserted below.</p>

<img src="../../../results/figs/vicon_obsolete/gif/exp38.gif" alt="experiment 38 results after ZV correction" width=%100 height=auto>

<!---
<img src="results/figs/vicon_obsolete/stride_detection_exp_38.png" alt="stride detection results on imu data for experiment 38 of VICON dataset">
--->

<h4>Experiment 43 (2017-12-15-18-01-18)</h4>

<p align="justify">We see that the strides {3, 14, 16} are not detected in the plots below.</p>

<!---
<img src="results/figs/vicon_obsolete/exp43.jpg" alt="optimal detector results for experiment 43 (2017-12-15-18-01-18) VICON dataset" width=%100 height=auto>
--->

<p align="justify">The supplementary detector is selected as VICON, which was able to recover the missed strides all.</p>

<img src="../../../data/vicon/processed/experiment43_ZUPT_detectors_strides.png" alt="ZV labels for experiment 43 (2017-12-15-18-01-18) VICON dataset" width=%100 height=auto>

<p align="justify">Integration of filtered optimal ZUPT detector SHOE with the supplementary ZUPT detector (i.e., filtered VICON) enabled successfull detection of the missed strides as shown in the combined ZUPT detector plot above (located at the bottom). The corrected stride & heading system trajectory and ZV labels can be seen below for the experiment 43.</p>

<!---
<img src="results/figs/vicon_obsolete/exp43_corrected.jpg" alt="corrected results for experiment 43 (2017-12-15-18-01-18) VICON dataset" width=%100 height=auto>
--->

<p align="justify">To see the correction by the supplementary ZUPT detector, check the gif file inserted below.</p>

<img src="../../../results/figs/vicon_obsolete/gif/exp43.gif" alt="experiment 43 results after ZV correction" width=%100 height=auto>

<!---
<img src="results/figs/vicon_obsolete/stride_detection_exp_43.png" alt="stride detection results on imu data for experiment 43 of VICON dataset">
--->