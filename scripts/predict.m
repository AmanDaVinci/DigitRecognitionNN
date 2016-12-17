function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Adding bias to layer 1
X = [ones(m,1) X];

% Layer 2 feedforward propagation
z2 = X * Theta1';
a2 = sigmoid(z2);

% Adding bias to layer 2
a2 = [ones(m,1) a2];

% Layer 3 feedforward propagation
z3 = a2 * Theta2';
a3 = sigmoid(z3);

% Prediction as a vector
% with max probabilities
[h, p] = max(a3, [], 2);


% =========================================================================


end
