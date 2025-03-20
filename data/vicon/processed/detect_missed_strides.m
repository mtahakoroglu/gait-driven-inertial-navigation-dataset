clear all; close all; clc;
k = 75; % heuristic filter window size
%% EXPERIMENT 4
load('2017-11-22-11-25-20'); % misses 10th stride
expIndex = 4; nGT = 18; % actual number of strides
figure(expIndex); clf; set(gcf, 'position', [565 165 842 545]);
subplot(4,1,1); % OPTIMAL DETECTOR (RAW DATA)
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_shoe_opt, 1);
fprintf(sprintf('There are %i strides detected by (filtered) SHOE ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv_shoe_opt, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.77    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); 
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Optimal ZUPT Detector (SHOE) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [13.3280 1.0985 0]);

subplot(4,1,2); % OPTIMAL DETECTOR (FILTERED DATA)
[zv_shoe_opt_filtered, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_shoe_opt, k);
fprintf(sprintf('There are %i strides detected by (filtered) SHOE ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv_shoe_opt_filtered, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv_shoe_opt_filtered(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.54    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'});
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('ZUPT Detector (SHOE filtered) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [13.3280 1.0985 0]);

subplot(4,1,3); % SUPPLEMENTARY DETECTOR (FILTERED DATA)
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_vicon_opt, k);
fprintf(sprintf('There are %i strides detected by (filtered) VICON ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv, 'LineWidth', 1.5, 'Color', 'k');
hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.31    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); set(gca, 'FontSize', 12);
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Supplementary ZUPT Detector (VICON filtered) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [13.3280 1.0985 0]);

subplot(4,1,4); % COMBINED DETECTOR
tolerance = 1e-4;  % Define a small tolerance
indexStart = find(abs(ts-14.005) < tolerance);
indexEnd = find(abs(ts-14.0701) < tolerance);
T = 0;
zv = zv_shoe_opt_filtered; zv(indexStart-T:indexEnd+T) = 1;
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv, 1);
fprintf(sprintf('There are %i strides detected by combined ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv, 'LineWidth', 1.5, 'Color', 'k');
hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.08    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Combined ZUPT Detector - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [13.3280 1.0985 0]);
h = xlabel('Time [s]', 'FontSize', 14); set(h, 'Position', [13.2921   -0.1649   -1.0000]);
print(sprintf('-f%i', expIndex),sprintf('experiment%i_ZUPT_detectors_strides', expIndex),'-dpng','-r800');
%% EXPERIMENT 6
load('2017-11-22-11-26-46'); % misses 9th stride
expIndex = 6; nGT = 24; % actual number of strides
figure(expIndex); clf; set(gcf, 'position', [565 165 842 545]);
subplot(4,1,1); % OPTIMAL DETECTOR (RAW DATA)
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_ared_opt, 1);
fprintf(sprintf('There are %i strides detected by (filtered) ARED ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv_ared_opt, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.77    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); 
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Optimal ZUPT Detector (ARED) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [15.3280 1.0985 0]);

subplot(4,1,2); % OPTIMAL DETECTOR (FILTERED DATA)
[zv_ared_opt_filtered, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_ared_opt, k);
fprintf(sprintf('There are %i strides detected by (filtered) ARED ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv_ared_opt_filtered, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv_ared_opt_filtered(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.54    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'});
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('ZUPT Detector (ARED filtered) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [15.3280 1.0985 0]);

subplot(4,1,3); % SUPPLEMENTARY DETECTOR (FILTERED DATA)
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_vicon_opt, k);
fprintf(sprintf('There are %i strides detected by (filtered) VICON ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv, 'LineWidth', 1.5, 'Color', 'k');
hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.31    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); set(gca, 'FontSize', 12);
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Supplementary ZUPT Detector (VICON filtered) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [15.3280 1.0985 0]);

subplot(4,1,4); % COMBINED DETECTOR
tolerance = 1e-4;  % Define a small tolerance
indexStart = find(abs(ts-12.725) < tolerance);
indexEnd = find(abs(ts-13.135) < tolerance);
T = 0;
zv = zv_ared_opt_filtered; zv(indexStart-T:indexEnd+T) = 1;
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv, 1);
fprintf(sprintf('There are %i strides detected by combined ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv, 'LineWidth', 1.5, 'Color', 'k');
hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.08    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Combined ZUPT Detector - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [15.3280 1.0985 0]);
h = xlabel('Time [s]', 'FontSize', 14); set(h, 'Position', [17.2921   -0.1649   -1.0000]);
print(sprintf('-f%i', expIndex),sprintf('experiment%i_ZUPT_detectors_strides', expIndex),'-dpng','-r800');
%% EXPERIMENT 11
load('2017-11-22-11-35-59'); % misses 7th stride
expIndex = 11; nGT = 29; % actual number of strides
figure(expIndex); clf; set(gcf, 'position', [565 165 842 545]);
subplot(4,1,1); % OPTIMAL DETECTOR (RAW DATA)
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_shoe_opt, 1);
fprintf(sprintf('There are %i strides detected by (filtered) SHOE ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv_shoe_opt, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.77    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); 
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Optimal ZUPT Detector (SHOE) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [18.3280 1.0985 0]);

subplot(4,1,2); % OPTIMAL DETECTOR (FILTERED DATA)
[zv_shoe_opt_filtered, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_shoe_opt, k);
fprintf(sprintf('There are %i strides detected by (filtered) SHOE ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv_shoe_opt_filtered, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv_shoe_opt_filtered(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.54    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'});
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('ZUPT Detector (SHOE filtered) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [18.3280 1.0985 0]);

subplot(4,1,3); % SUPPLEMENTARY DETECTOR (FILTERED DATA)
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_vicon_opt, k);
fprintf(sprintf('There are %i strides detected by (filtered) VICON ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv, 'LineWidth', 1.5, 'Color', 'k');
hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.31    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); set(gca, 'FontSize', 12);
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Supplementary ZUPT Detector (VICON filtered) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [18.3280 1.0985 0]);

subplot(4,1,4); % COMBINED DETECTOR
tolerance = 1e-4;  % Define a small tolerance
indexStart = find(abs(ts-10.69) < tolerance);
indexEnd = find(abs(ts-10.81) < tolerance);
T = 0;
zv = zv_shoe_opt_filtered; zv(indexStart-T:indexEnd+T) = 1;
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv, 1);
fprintf(sprintf('There are %i strides detected by combined ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv, 'LineWidth', 1.5, 'Color', 'k');
hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.08    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Combined ZUPT Detector - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [18.3280 1.0985 0]);
h = xlabel('Time [s]', 'FontSize', 14); set(h, 'Position', [20.2921   -0.1649   -1.0000]);
print(sprintf('-f%i', expIndex),sprintf('experiment%i_ZUPT_detectors_strides', expIndex),'-dpng','-r800');
%% EXPERIMENT 18
load('2017-11-22-11-48-35'); % misses 7th stride
expIndex = 18; nGT = 15; % actual number of strides
figure(expIndex); clf; set(gcf, 'position', [796, 364, 641, 413]);
subplot(4,1,1); % OPTIMAL DETECTOR (RAW DATA)
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_shoe_opt, 1);
fprintf(sprintf('There are %i strides detected by (filtered) SHOE ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv_shoe_opt, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0574    0.7700    0.9343    0.1700]);
h = legend('ZV signal', 'Stride index'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); 
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV label', 'FontSize', 14, 'FontWeight', 'normal');
set(gca, 'XTick', 0:3:21, 'XTickLabel', {'','','','','','','',''});
titleText = sprintf('Optimal ZUPT Detector (SHOE) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [10.3280 1.0585 0], 'FontWeight', 'normal');

subplot(4,1,2); % OPTIMAL DETECTOR (FILTERED DATA)
[zv_shoe_opt_filtered, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_shoe_opt, k);
fprintf(sprintf('There are %i strides detected by (filtered) SHOE ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv_shoe_opt_filtered, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv_shoe_opt_filtered(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0574    0.5400    0.9343    0.1700]);
h = legend('ZV signal', 'Stride index'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'});
set(gca, 'XTick', 0:3:21, 'XTickLabel', {'','','','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV label', 'FontSize', 14, 'FontWeight', 'normal');
titleText = sprintf('ZUPT Detector (SHOE filtered) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [10.3280 1.0585 0], 'FontWeight', 'normal');

subplot(4,1,3); % SUPPLEMENTARY DETECTOR (FILTERED DATA)
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_vicon_opt, k);
fprintf(sprintf('There are %i strides detected by (filtered) VICON ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv, 'LineWidth', 1.5, 'Color', 'k');
hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0574    0.3100    0.9343    0.1700]);
h = legend('ZV signal', 'Stride index'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); set(gca, 'FontSize', 12);
set(gca, 'XTick', 0:3:21, 'XTickLabel', {'','','','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV label', 'FontSize', 14, 'FontWeight', 'normal');
titleText = sprintf('Supplementary ZUPT Detector (VICON filtered) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [10.3280 1.0585 0], 'FontWeight', 'normal');

subplot(4,1,4); % COMBINED DETECTOR
tolerance = 1e-4;  % Define a small tolerance
indexStart = find(abs(ts-9.41515) < tolerance);
indexEnd = find(abs(ts-9.70017) < tolerance);
T = 0;
zv = zv_shoe_opt_filtered; zv(indexStart-T:indexEnd+T) = 1;
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv, 1);
fprintf(sprintf('There are %i strides detected by combined ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv, 'LineWidth', 1.5, 'Color', 'k');
hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0574    0.0800    0.9343    0.1700]);
h = legend('ZV signal', 'Stride index'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'});
set(gca, 'XTick', 0:3:21, 'XTickLabel', {'','3','6','9','12','15','18','21'});
set(gca, 'FontSize', 12);
ylabel('ZV label', 'FontSize', 14, 'FontWeight', 'normal');
titleText = sprintf('Combined ZUPT Detector - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [10.3280 1.0585 0], 'FontWeight', 'normal');
h = xlabel('Time [s]', 'FontSize', 14, 'FontWeight','normal');
set(h, 'Position', [10.45   -0.1249   -1.0000]);
print(sprintf('-f%i', expIndex),sprintf('experiment%i_ZUPT_detectors_strides', expIndex),'-dpng','-r800');
set(figure(expIndex), 'PaperSize', [6.70, 4.28]);
print(sprintf('-f%i', expIndex),sprintf('experiment%i_ZUPT_detectors_strides', expIndex),'-dpdf','-r600');
%% EXPERIMENT 27
% first three strides can be retrieved by VICON zupt detector while 
% the last one can be retrieved by MBGTD
load('2017-11-27-11-12-44.mat'); % missed strides {9, 16, 17, 18}
expIndex = 27; nGT = 20; % actual number of strides
figure(expIndex); clf; set(gcf, 'position', [595 60 842 722]);
subplot(5,1,1); % OPTIMAL DETECTOR (RAW DATA)
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_shoe_opt, 1);
fprintf(sprintf('There are %i strides detected by (filtered) SHOE ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv_shoe_opt, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.82    0.948    0.14]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); 
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Optimal ZUPT Detector (SHOE) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [11.3280 1.0985 0]);

subplot(5,1,2); % OPTIMAL DETECTOR (FILTERED DATA)
[zv_shoe_opt_filtered, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_shoe_opt, k);
fprintf(sprintf('There are %i strides detected by (filtered) SHOE ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv_shoe_opt_filtered, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv_shoe_opt_filtered(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.63    0.948    0.14]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'});
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('ZUPT Detector (SHOE filtered) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [11.3280 1.0985 0]);

subplot(5,1,3); % SUPPLEMENTARY DETECTOR (FILTERED DATA for VICON DETECTOR)
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_vicon_opt, k);
fprintf(sprintf('There are %i strides detected by (filtered) VICON ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.44    0.948    0.14]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); set(gca, 'FontSize', 12);
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Supplementary ZUPT Detector (VICON filtered) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [11.3280 1.0985 0]);

subplot(5,1,4); % SUPPLEMENTARY DETECTOR (FILTERED DATA for VICON DETECTOR)
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_mbgtd_opt, k);
fprintf(sprintf('There are %i strides detected by (filtered) MBGTD ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.25    0.948    0.14]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); set(gca, 'FontSize', 12);
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Supplementary ZUPT Detector (MBGTD filtered) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [11.3280 1.0985 0]);

subplot(5,1,5); % COMBINED DETECTOR
tolerance = 1e-4;  % Define a small tolerance
indexStart1 = find(abs(ts-9.08498) < tolerance);
indexEnd1 = find(abs(ts-9.14998) < tolerance);
indexStart2 = find(abs(ts-14.9496) < tolerance);
indexEnd2 = find(abs(ts-15.0098) < tolerance);
indexStart3 = find(abs(ts-15.7747) < tolerance);
indexEnd3 = find(abs(ts-15.8398) < tolerance);
index4 = find(abs(ts-16.6496) < tolerance);
T = 0; % expand to right and left (extend ZV period)
zv = zv_shoe_opt_filtered; zv(indexStart1-T:indexEnd1+T) = 1;
zv(indexStart2-T:indexEnd2+T) = 1; zv(indexStart3-T:indexEnd3+T) = 1;
zv(index4-3:index4+3) = 1;
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv, 1);
fprintf(sprintf('There are %i strides detected by combined ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv, 'LineWidth', 1.5, 'Color', 'k');
hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.06    0.948    0.14]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Combined ZUPT Detector - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [11.3280 1.0985 0]);
h = xlabel('Time [s]', 'FontSize', 14); set(h, 'Position', [11.2921   -0.1949   -1.0000]);
print(sprintf('-f%i', expIndex),sprintf('experiment%i_ZUPT_detectors_strides', expIndex),'-dpng','-r800');
%% EXPERIMENT 30
% three missed strides {2, 10, 11}. Both strides are detected by SHOE detector.
load('2017-11-27-11-14-03.mat'); % missed strides {9, 16, 17, 18}
expIndex = 30; nGT = 12; % actual number of strides
figure(expIndex); clf; set(gcf, 'position', [565 165 842 545]);
subplot(4,1,1); % OPTIMAL DETECTOR (RAW DATA)
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_vicon_opt, 1);
fprintf(sprintf('There are %i strides detected by (filtered) VICON ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv_vicon_opt, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.77    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); 
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Optimal ZUPT Detector (VICON) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [10.3280 1.0985 0]);

subplot(4,1,2); % OPTIMAL DETECTOR (FILTERED DATA)
[zv_vicon_opt_filtered, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_vicon_opt, k);
fprintf(sprintf('There are %i strides detected by (filtered) VICON ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv_vicon_opt_filtered, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv_vicon_opt_filtered(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.54    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'});
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('ZUPT Detector (VICON filtered) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [10.3280 1.0985 0]);

subplot(4,1,3); % SUPPLEMENTARY DETECTOR (FILTERED DATA for VICON DETECTOR)
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_shoe_opt, k);
fprintf(sprintf('There are %i strides detected by (filtered) SHOE ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.31    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); set(gca, 'FontSize', 12);
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Supplementary ZUPT Detector (SHOE filtered) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [10.3280 1.0985 0]);

subplot(4,1,4); % COMBINED DETECTOR
tolerance = 1e-4;  % Define a small tolerance
indexStart1 = find(abs(ts-3.10498) < tolerance);
indexEnd1 = find(abs(ts-3.15017) < tolerance);
indexStart2 = find(abs(ts-8.92976) < tolerance);
indexEnd2 = find(abs(ts-8.94988) < tolerance);
T = 0; % expand to right and left (extend ZV period)
zv = zv_vicon_opt_filtered;
zv(indexStart1-T:indexEnd1+T) = 1;
zv(indexStart2-T:indexEnd2+T) = 1;
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv, 1);
fprintf(sprintf('There are %i strides detected by combined ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.31-0.23    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Combined ZUPT Detector - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [10.3280 1.0985 0]);
h = xlabel('Time [s]', 'FontSize', 14); set(h, 'Position', [10.2921   -0.1949   -1.0000]);
print(sprintf('-f%i', expIndex),sprintf('experiment%i_ZUPT_detectors_strides', expIndex),'-dpng','-r800');
%% EXPERIMENT 32
% three missed strides {9, 11, 20}. All strides are detected by VICON detector.
load('2017-11-27-11-17-28.mat'); % missed strides {9, 16, 17, 18}
expIndex = 32; nGT = 26; % actual number of strides
figure(expIndex); clf; set(gcf, 'position', [595 60 842 722]);
subplot(7,1,1); % OPTIMAL DETECTOR (RAW DATA)
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_shoe_opt, 1);
fprintf(sprintf('There are %i strides detected by (filtered) SHOE ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv_shoe_opt, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.87    0.948    0.095]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); 
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Optimal ZUPT Detector (SHOE) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [12.3280 1.0985 0]);

subplot(7,1,2); % OPTIMAL DETECTOR (FILTERED DATA)
[zv_shoe_opt_filtered, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_shoe_opt, k);
fprintf(sprintf('There are %i strides detected by (filtered) SHOE ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv_shoe_opt_filtered, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv_shoe_opt_filtered(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.735    0.948    0.095]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'});
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('ZUPT Detector (SHOE filtered) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [12.3280 1.0985 0]);

subplot(7,1,3); % SUPPLEMENTARY DETECTOR (FILTERED DATA for VICON DETECTOR)
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_vicon_opt, k);
fprintf(sprintf('There are %i strides detected by (filtered) VICON ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.6    0.948    0.095]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); set(gca, 'FontSize', 12);
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Supplementary ZUPT Detector (VICON filtered) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [12.3280 1.0985 0]);

subplot(7,1,4); % SUPPLEMENTARY DETECTOR (FILTERED DATA for ARED DETECTOR)
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_ared_opt, k);
fprintf(sprintf('There are %i strides detected by (filtered) ARED ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.465    0.948    0.095]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); set(gca, 'FontSize', 12);
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Supplementary ZUPT Detector (ARED filtered) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [12.3280 1.0985 0]);

subplot(7,1,5); % SUPPLEMENTARY DETECTOR (FILTERED DATA for MBGTD DETECTOR)
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_mbgtd_opt, k);
fprintf(sprintf('There are %i strides detected by (filtered) MBGTD ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.33    0.948    0.095]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); set(gca, 'FontSize', 12);
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Supplementary ZUPT Detector (MBGTD filtered) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [12.3280 1.0985 0]);

subplot(7,1,7); % COMBINED DETECTOR
tolerance = 1e-4;  % Define a small tolerance
index1 = find(abs(ts-9.26008) < tolerance);
indexStart2 = find(abs(ts-10.7101) < tolerance);
indexEnd2 = find(abs(ts-10.7149) < tolerance);
indexStart3 = find(abs(ts-19.2296) < tolerance);
indexEnd3 = find(abs(ts-20.7747) < tolerance);
index3 = round((indexEnd3 - indexStart3)/2 + indexStart3);
T = 3; % expand to right and left (extend ZV period)
zv = zv_shoe_opt_filtered;
zv(index1-T:index1+T) = 1;
zv(indexStart2-T:indexEnd2+T) = 1;
zv(index3-T:index3+T) = 1;
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv, 1);
fprintf(sprintf('There are %i strides detected by combined ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.055    0.948    0.095]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Combined ZUPT Detector - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [12.3280 1.0985 0]);
h = xlabel('Time [s]', 'FontSize', 14); set(h, 'Position', [12.7921   -0.1949   -1.0000]);

subplot(7,1,6); % SUPPLEMENTARY DETECTOR (MANUAL DETECTOR for the LAST STRIDE)
% [zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_amvd_opt, k);
zv = zeros(1,length(zv));
tolerance = 1e-4;  % Define a small tolerance
index3 = round((indexEnd3 - indexStart3)/2 + indexStart3);
T = 3; % expand to right and left (extend ZV period)
zv(index3-T:index3+T) = 1;
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv, k);
zv(1:50) = 0;
fprintf(sprintf('There are %i strides detected by the manual ZV detector in experiment %i.\n', n-1, expIndex));
plot(ts, zv, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(index3+T), zv(index3+T), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.195    0.948    0.095]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); set(gca, 'FontSize', 12);
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Supplementary ZUPT Detector (MANUAL annotation) - %i/%i strides detected', n-1, nGT);
h = title(titleText); set(h, 'position', [12.3280 1.0985 0]);
print(sprintf('-f%i', expIndex),sprintf('experiment%i_ZUPT_detectors_strides', expIndex),'-dpng','-r800');
%% EXPERIMENT 36 - 7th stride is missed
load('2017-11-27-11-23-18'); % misses 10th stride
expIndex = 36; nGT = 24; % actual number of strides
figure(expIndex); clf; set(gcf, 'position', [565 165 842 545]);
subplot(4,1,1); % OPTIMAL DETECTOR (RAW DATA)
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_shoe_opt, 1);
fprintf(sprintf('There are %i strides detected by (filtered) SHOE ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv_shoe_opt, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.77    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); 
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Optimal ZUPT Detector (SHOE) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [13.3280 1.0985 0]);

subplot(4,1,2); % OPTIMAL DETECTOR (FILTERED DATA)
[zv_shoe_opt_filtered, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_shoe_opt, k);
fprintf(sprintf('There are %i strides detected by (filtered) SHOE ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv_shoe_opt_filtered, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv_shoe_opt_filtered(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.54    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'});
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('ZUPT Detector (SHOE filtered) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [13.3280 1.0985 0]);

subplot(4,1,3); % SUPPLEMENTARY DETECTOR (FILTERED DATA)
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_vicon_opt, k);
fprintf(sprintf('There are %i strides detected by (filtered) VICON ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv, 'LineWidth', 1.5, 'Color', 'k');
hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.31    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); set(gca, 'FontSize', 12);
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Supplementary ZUPT Detector (VICON filtered) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [13.3280 1.0985 0]);

subplot(4,1,4); % COMBINED DETECTOR
tolerance = 1e-4;  % Define a small tolerance
indexStart = find(abs(ts-9.32501) < tolerance);
indexEnd = find(abs(ts-9.45018) < tolerance);
T = 0;
zv = zv_shoe_opt_filtered; zv(indexStart-T:indexEnd+T) = 1;
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv, 1);
fprintf(sprintf('There are %i strides detected by combined ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv, 'LineWidth', 1.5, 'Color', 'k');
hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.08    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Combined ZUPT Detector - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [13.3280 1.0985 0]);
h = xlabel('Time [s]', 'FontSize', 14); set(h, 'Position', [13.2921   -0.1649   -1.0000]);
print(sprintf('-f%i', expIndex),sprintf('experiment%i_ZUPT_detectors_strides', expIndex),'-dpng','-r800');
%% EXPERIMENT 38 - {3,27,33} stride is missed
load('2017-11-27-11-25-12'); % misses 10th stride
expIndex = 38; nGT = 42; % actual number of strides
figure(expIndex); clf; set(gcf, 'position', [565 165 842 545]);
subplot(4,1,1); % OPTIMAL DETECTOR (RAW DATA)
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_shoe_opt, 1);
fprintf(sprintf('There are %i strides detected by (filtered) SHOE ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv_shoe_opt, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.77    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); 
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Optimal ZUPT Detector (SHOE) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [18.3280 1.0985 0]);

subplot(4,1,2); % OPTIMAL DETECTOR (FILTERED DATA)
[zv_shoe_opt_filtered, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_shoe_opt, k);
fprintf(sprintf('There are %i strides detected by (filtered) SHOE ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv_shoe_opt_filtered, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv_shoe_opt_filtered(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.54    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'});
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('ZUPT Detector (SHOE filtered) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [18.3280 1.0985 0]);

subplot(4,1,3); % SUPPLEMENTARY DETECTOR (FILTERED DATA)
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_vicon_opt, k);
fprintf(sprintf('There are %i strides detected by (filtered) VICON ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv, 'LineWidth', 1.5, 'Color', 'k');
hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.31    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); set(gca, 'FontSize', 12);
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Supplementary ZUPT Detector (VICON filtered) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [18.3280 1.0985 0]);

subplot(4,1,4); % COMBINED DETECTOR
tolerance = 1e-4;  % Define a small tolerance
index1 = find(abs(ts-4.37514) < tolerance);
index2 = find(abs(ts-22.6046) < tolerance);
indexStart3 = find(abs(ts-27.0543) < tolerance);
indexEnd3 = find(abs(ts-27.1042) < tolerance);
T = 3;
zv = zv_shoe_opt_filtered; zv(index1-T:index1+T) = 1;
zv(index2-T:index2+T) = 1; zv(indexStart3:indexEnd3) = 1;
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv, 1);
fprintf(sprintf('There are %i strides detected by combined ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv, 'LineWidth', 1.5, 'Color', 'k');
hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.08    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Combined ZUPT Detector - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [18.3280 1.0985 0]);
h = xlabel('Time [s]', 'FontSize', 14); set(h, 'Position', [17.7921   -0.1649   -1.0000]);
print(sprintf('-f%i', expIndex),sprintf('experiment%i_ZUPT_detectors_strides', expIndex),'-dpng','-r800');
%% EXPERIMENT 43 - {3,14,16} stride is missed
load('2017-12-15-18-01-18'); % misses 10th stride
k = 39;
expIndex = 43; nGT = 24; % actual number of strides
figure(expIndex); clf; set(gcf, 'position', [565 165 842 545]);
subplot(4,1,1); % OPTIMAL DETECTOR (RAW DATA)
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_ared_opt, 1);
fprintf(sprintf('There are %i strides detected by (filtered) ARED ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv_shoe_opt, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.77    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); 
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Optimal ZUPT Detector (ARED) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [12.0280 1.0985 0]);

subplot(4,1,2); % OPTIMAL DETECTOR (FILTERED DATA)
[zv_ared_opt_filtered, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_ared_opt, k);
fprintf(sprintf('There are %i strides detected by (filtered) SHOE ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv_ared_opt_filtered, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv_ared_opt_filtered(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.54    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'});
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('ZUPT Detector (ARED filtered) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [12.0280 1.0985 0]);

subplot(4,1,3); % SUPPLEMENTARY DETECTOR (FILTERED DATA)
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_vicon_opt, k);
fprintf(sprintf('There are %i strides detected by (filtered) VICON ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv, 'LineWidth', 1.5, 'Color', 'k');
hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.31    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); set(gca, 'FontSize', 12);
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Supplementary ZUPT Detector (VICON filtered) - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [12.0280 1.0985 0]);

subplot(4,1,4); % COMBINED DETECTOR
tolerance = 1e-4;  % Define a small tolerance
indexStart1 = find(abs(ts-4.52997) < tolerance);
indexEnd1 = find(abs(ts-4.71999) < tolerance);
indexStart2 = find(abs(ts-13.0697) < tolerance);
indexEnd2 = find(abs(ts-13.3098) < tolerance);
indexStart3 = find(abs(ts-14.6298) < tolerance);
indexEnd3 = find(abs(ts-14.8697) < tolerance);
T = 0;
zv = zv_ared_opt_filtered; zv(indexStart1-T:indexEnd1+T) = 1;
zv(indexStart2-T:indexEnd2+T) = 1; zv(indexStart3:indexEnd3) = 1;
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv, 1);
fprintf(sprintf('There are %i strides detected by combined ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv, 'LineWidth', 1.5, 'Color', 'k');
hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0437    0.08    0.948    0.17]);
h = legend('ZV labels', 'strides'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); set(gca, 'FontSize', 12);
ylabel('ZV labels', 'FontSize', 12, 'FontWeight', 'normal');
titleText = sprintf('Combined ZUPT Detector - %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [12.3280 1.0985 0]);
h = xlabel('Time [s]', 'FontSize', 14); set(h, 'Position', [11.7921   -0.1949   -1.0000]);
print(sprintf('-f%i', expIndex),sprintf('experiment%i_ZUPT_detectors_strides', expIndex),'-dpng','-r800');