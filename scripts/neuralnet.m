%% Initialization
clear ; close all; clc

%% Setup the parameters
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   

%% =========== Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  We will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('../data/imageDataSet.mat');
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Loading Pameters ======================
% In this part we load some pre-initialized 
% neural network parameters.

fprintf('\nLoading Saved Neural Network Parameters ...\n')

% Load the weights into variables Theta1 and Theta2
load('../data/learntWeights.mat');

%% ==================== Predict =========================
%  After training the neural network, we would like to use it to predict
%  the labels.

pred = predict(Theta1, Theta2, X);

fprintf('Training Set Accuracy: %f\n', mean(double(pred == y)) * 100);

fprintf('Program paused. Press enter to continue.\n');
pause;

%  Randomly permute examples
rp = randperm(m);

for i = 1:m
    % Display 
    fprintf('\n>>Displaying Example Image\n');
    displayData(X(rp(i), :));

    pred = predict(Theta1, Theta2, X(rp(i),:));
    fprintf('\n>>Neural Network Prediction: %d (DIGIT %d)\n\n', pred, mod(pred, 10));
    
    % Pause
    % fprintf('Program paused. Press enter to continue.\n');
    pause;
end