clear all; close all; clc;
k = 75;
%% EXPERIMENT 18 including trajectory
load('2017-11-22-11-48-35'); % misses 7th stride
load('2017-11-22-11-48-35_GDLBIO')
expIndex = 18; nGT = 15; % actual number of strides
figure(expIndex); clf; set(gcf, 'position', [796, 364, 641, 413]);
subplot(4,1,1); % OPTIMAL DETECTOR (RAW DATA)
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_shoe_opt, 1);
fprintf(sprintf('There are %i strides detected by SHOE ZV detector in experiment %i.\n', n, expIndex));
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
zv = zv_shoe_opt_filtered; zv(indexStart-T:indexEnd+T) = 1; % combine ZUPT detectors
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

%% Experiment expID paper figure generation
figure(expIndex+1); clf; set(gcf, 'position', 1e3*[ 0.0410    0.080    1.0368    0.7]);
height = 0.13; width = 0.59; verticalGap = 0.19; topVerticalStart = 0.825;

subplot(5,2,5); % SUPPLEMENTARY DETECTOR (VICON FILTERED DATA)
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_vicon_opt, k);
fprintf(sprintf('There are %i strides detected by (filtered) VICON ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv, 'LineWidth', 1.5, 'Color', 'k');
hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0363, topVerticalStart-2*verticalGap, width, height]);
h = legend('ZV signal', 'Stride index'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); set(gca, 'FontSize', 12);
set(gca, 'XTick', 0:3:21, 'XTickLabel', {'','','','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV label', 'FontSize', 14, 'FontWeight', 'normal');
titleText = sprintf('Supplementary ZUPT Detector (VICON filtered) | %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [10.3280 1.0585 0], 'FontWeight', 'normal');

subplot(5,2,9); % PyShoe (LSTM) DETECTOR
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_lstm, 1);
fprintf(sprintf('There are %i strides detected by PyShoe (LSTM) ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv, 'LineWidth', 1.5, 'Color', 'k');
hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0363, topVerticalStart-4*verticalGap, width, height]);
h = legend('ZV signal', 'Stride index'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'});
set(gca, 'XTick', 0:3:21, 'XTickLabel', {'','3','6','9','12','15','18','21'});
set(gca, 'FontSize', 12);
ylabel('ZV label', 'FontSize', 14, 'FontWeight', 'normal');
titleText = sprintf('ZUPT Detector (PyShoe (LSTM)) | %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [10.3280 1.0585 0], 'FontWeight', 'normal');
h = xlabel('Time [s]', 'FontSize', 14, 'FontWeight','normal');
set(h, 'Position', [10.45   -0.225   -1.0000]);

subplot(5,2,8); % GT vs. GT stride-wise with PyShoe (LSTM) trajectory
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_lstm, 1);
plot(gt(:,1), gt(:,2), 'LineWidth', 1.2, 'Color', 'r'); hold on;
plot(pyshoeTrajectory(:,1), pyshoeTrajectory(:,2), 'b', 'LineWidth', 1.2);
% ([0:4, 6, 9:15]+1) are the detected stride numbers
plot(pyshoeTrajectory(strideIndex,1), ...
    pyshoeTrajectory(strideIndex,2), 'bo', ...
    'LineWidth', 1.2, 'MarkerFaceColor', 'b');
tolerance = 1e-4;  % Define a small tolerance
indexStart = find(abs(ts-9.41515) < tolerance);
indexEnd = find(abs(ts-9.70017) < tolerance);
T = 0;
zv = zv_shoe_opt_filtered; zv(indexStart-T:indexEnd+T) = 1; % Combining ZUPT detectors
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv, 1);
% [5,7,8]+1 are undetected (missed) stride numbers by LSTM
plot(pyshoeTrajectory(strideIndex([5,7,8]+1),1), pyshoeTrajectory(strideIndex([5,7,8]+1),2), ...
    'bo', 'LineWidth', 1.2);
grid on; set(gca, 'GridLineStyle', '--', 'FontSize', 12);
% set(gca, 'position', [0.0363, topVerticalStart, width, height]);
h = legend({"GT", "PyShoe" + newline + "(LSTM)", "Detected" + newline + "Strides", ...
    "Missed" + newline + "strides"}, 'location', 'west', 'FontWeight', 'bold');
ylabel('y [m]', 'FontSize', 14);
xShift = 1;
% set(gca, 'XTick', [-2:2]-xShift, 'XTickLabel', {'-3','-2','','0','1'});
set(gca, 'YTick', -1:0.5:1.5, 'YTickLabel', {'-1','','0','','1', ''});
set(gca, 'position', [0.6805    0.0640    0.3172    0.2784]);
% axis([-1.6, 1.25, -1.5, 2]);
axis equal;
xShift = 0.95;
x_limits = xlim-xShift;
xlim([x_limits]);
h = xlabel('x[m]', 'FontSize', 14);
set(h, 'Position', [-1.2   -1.37   -1.0000]);

subplot(5,2,4); % GT vs. GT stride-wise with combined ZUPT detector
tolerance = 1e-4;  % Define a small tolerance
indexStart = find(abs(ts-9.41515) < tolerance);
indexEnd = find(abs(ts-9.70017) < tolerance);
T = 0;
zv = zv_shoe_opt_filtered; zv(indexStart-T:indexEnd+T) = 1; % Combining ZUPT detectors
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv, 1);
plot(gt(:,1), gt(:,2), 'LineWidth', 1.2, 'Color', 'r'); hold on;
plot(gt(strideIndex,1), gt(strideIndex,2), 'bo-', 'LineWidth', 1.2, 'MarkerFaceColor', 'b');
grid on; set(gca, 'GridLineStyle', '--', 'FontSize', 12);
% set(gca, 'position', [0.0363, topVerticalStart, width, height]);
h = legend({"GT", "Stride-wise" + newline + "GT" + newline + "(Combined)"});
set(h, 'FontWeight', 'bold', 'location', 'southwest', 'location', 'southwest');
set(h, 'position', [0.6908-0.005    0.3973-0.005    0.1264    0.1116]);
ylabel('y [m]', 'FontSize', 14);
set(gca, 'XTick', [-3:1], 'XTickLabel', {'','','','',''});
set(gca, 'YTick', -1:0.5:1.5, 'YTickLabel', {'-1','','0','','1', ''});
set(gca, 'position', [0.6805    0.3840    0.3172    0.2784]);
% axis([-1.6, 1.25, -1.5, 2]);
xShift = 0.95;
axis equal;
x_limits = xlim-xShift;
xlim([x_limits]);

subplot(5,2,3); % OPTIMAL DETECTOR (FILTERED DATA)
[zv_shoe_opt_filtered, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_shoe_opt, k);
fprintf(sprintf('There are %i strides detected by (filtered) SHOE ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv_shoe_opt_filtered, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv_shoe_opt_filtered(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0363, topVerticalStart-verticalGap, width, height]);
h = legend('ZV signal', 'Stride index'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'});
set(gca, 'XTick', 0:3:21, 'XTickLabel', {'','','','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV label', 'FontSize', 14, 'FontWeight', 'normal');
titleText = sprintf('ZUPT Detector (SHOE filtered) | %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [10.3280 1.0585 0], 'FontWeight', 'normal');

subplot(5,2,2); % GT vs. GT stride-wise with optimal ZUPT detector (SHOE filtered)
[zv_shoe_opt_filtered, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_shoe_opt, k);
plot(gt(:,1), gt(:,2), 'LineWidth', 1.2, 'Color', 'r'); hold on;
plot(gt(strideIndex,1), gt(strideIndex,2), 'bo-', 'LineWidth', 1.2, 'MarkerFaceColor', 'b');
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv, 1);
plot(gt(strideIndex(8),1), gt(strideIndex(8),2), 'bo', 'LineWidth', 1.2);
grid on; set(gca, 'GridLineStyle', '--', 'FontSize', 12);
% set(gca, 'position', [0.0363, topVerticalStart, width, height]);
h = legend({"GT", "Stride-wise" + newline + "GT (SHOE"+newline+"filtered)", "Missed" + newline + "stride"});
set(h, 'FontWeight', 'bold', 'location', 'southwest');
set(h, 'Position', [0.6858    0.6942    0.1240    0.1649]);
ylabel('y [m]', 'FontSize', 14);
set(gca, 'XTick', [-3:1], 'XTickLabel', {'','','','',''});
set(gca, 'YTick', -1:0.5:1.5, 'YTickLabel', {'-1','','0','','1', ''});
set(gca, 'position', [0.6805    0.7040    0.3172    0.2784]);
% axis([-1.6, 1.25, -1.5, 2]);
axis equal;
xShift = 0.95;
x_limits = xlim-xShift;
xlim([x_limits]);
% set(h, 'FontSize', 12, 'location', 'southeast');
% set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); 
% set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
% ylabel('ZV label', 'FontSize', 14, 'FontWeight', 'normal');
% titleText = sprintf('Optimal ZUPT Detector (SHOE) | %i/%i strides detected', n, nGT);
% h = title(titleText); set(h, 'position', [10.3280 1.0585 0], 'FontWeight', 'normal');


subplot(5,2,1); % OPTIMAL DETECTOR (RAW DATA)
[zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_shoe_opt, 1);
fprintf(sprintf('There are %i strides detected by (filtered) SHOE ZV detector in experiment %i.\n', n, expIndex));
plot(ts, zv_shoe_opt, 'LineWidth', 1.5, 'Color', 'k'); hold on;
plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on; set(gca, 'GridLineStyle', '--'); axis tight;
set(gca, 'position', [0.0363, topVerticalStart, width, height]);
h = legend('ZV signal', 'Stride index'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'}); 
set(gca, 'XTickLabel', {'','','','','',''}); set(gca, 'FontSize', 12);
ylabel('ZV label', 'FontSize', 14, 'FontWeight', 'normal');
set(gca, 'XTick', 0:3:21, 'XTickLabel', {'','','','','','','',''});
titleText = sprintf('Optimal ZUPT Detector (SHOE) | %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [10.3280 1.0585 0], 'FontWeight', 'normal');

subplot(5,2,7); % COMBINED DETECTOR
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
set(gca, 'position', [0.0363, topVerticalStart-3*verticalGap, width, height]);
h = legend('ZV signal', 'Stride index'); set(h, 'FontSize', 12, 'location', 'southeast');
set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'});
set(gca, 'XTick', 0:3:21, 'XTickLabel', {'','','','','','','',''}); set(gca, 'FontSize', 12);
set(gca, 'FontSize', 12);
ylabel('ZV label', 'FontSize', 14, 'FontWeight', 'normal');
titleText = sprintf('Combined ZUPT Detector | %i/%i strides detected', n, nGT);
h = title(titleText); set(h, 'position', [10.3280 1.0585 0], 'FontWeight', 'normal');
% h = xlabel('Time [s]', 'FontSize', 14, 'FontWeight','normal');
% set(h, 'Position', [10.45   -0.1249   -1.0000]);

% subplot(6,2,11); % PyShoe (LSTM) FILTERED DETECTOR
% [zv, n, strideIndex] = heuristic_zv_filter_and_stride_detector(zv_lstm, k);
% fprintf(sprintf('There are %i strides detected by (filtered) PyShoe (LSTM) ZV detector in experiment %i.\n', n, expIndex));
% plot(ts, zv, 'LineWidth', 1.5, 'Color', 'k');
% hold on;
% plot(ts(strideIndex), zv(strideIndex), 'ko', 'LineWidth', 1.1, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
% grid on; set(gca, 'GridLineStyle', '--'); axis tight;
% set(gca, 'position', [0.0363, topVerticalStart-5*verticalGap, width, height]);
% h = legend('ZV signal', 'Stride index'); set(h, 'FontSize', 12, 'location', 'southeast');
% set(gca, 'YTick', [0,1], 'YTickLabel', {'0','1'});
% set(gca, 'XTick', 0:3:21, 'XTickLabel', {'','3','6','9','12','15','18','21'});
% set(gca, 'FontSize', 12);
% ylabel('ZV label', 'FontSize', 14, 'FontWeight', 'normal');
% titleText = sprintf('ZUPT Detector (PyShoe (LSTM) filtered) | %i/%i strides detected', n, nGT);
% h = title(titleText); set(h, 'position', [10.3280 1.0585 0], 'FontWeight', 'normal');
% h = xlabel('Time [s]', 'FontSize', 14, 'FontWeight','normal');
% set(h, 'Position', [10.45   -0.1249   -1.0000]);

set(figure(expIndex+1), 'PaperSize', [11.00, 7.50]);
print(sprintf('-f%i', expIndex+1),sprintf('PyShoe-Vicon-experiment%i', expIndex),'-dpng','-r300');