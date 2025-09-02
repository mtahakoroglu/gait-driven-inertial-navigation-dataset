clear all; close all; clc;
k = 75;
%% EXPERIMENT 18 including trajectory
load('2017-11-22-11-48-35'); % misses 7th stride
load('2017-11-22-11-48-35_GDLBIO')
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
% print(sprintf('-f%i', expIndex),sprintf('experiment%i_ZUPT_detectors_strides', expIndex),'-dpng','-r800');
% set(figure(expIndex), 'PaperSize', [6.70, 4.28]);
% print(sprintf('-f%i', expIndex),sprintf('experiment%i_ZUPT_detectors_strides', expIndex),'-dpdf','-r600');