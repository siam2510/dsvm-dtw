# A DTW-Based Kernel Extension of D-SVM Charts for Robust Process Monitoring

This repository contains code for the D-SVM chart and DTW-based kernel experiments on the ECG5000 dataset.

Summary : The distance-based support vector machine (D-SVM) chart is a nonparametric monitoring method designed to detect process shifts by quantifying the SVM-derived distance (score) between in-control reference data and real-time observations. It has demonstrated strong performance for high-dimensional and non-normal data. However, the conventional D-SVM chart employs an RBF kernel based on Euclidean distance, which limits its effectiveness when time-series data exhibit temporal misalignment.
To overcome this limitation, this study proposes a DTW-based D-SVM chart, in which the Dynamic Time Warping (DTW) distance is incorporated into the RBF kernel. The proposed chart effectively captures temporal distortions among time-series samples, maintaining the desired ARL₀ under in-control conditions and achieving a low ARL₁ when the process shifts out of control, as demonstrated in simulation results. A simulation study using the ECG5000 dataset was conducted to evaluate the proposed approach, and the results confirm that the DTW-based D-SVM chart remains robust under temporal misalignment conditions.

Keywords : Anomaly Detection, Dynamic Time Warping, Process Monitoring, Average Run Length
