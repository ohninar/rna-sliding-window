function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);%pega a quantidade de linhas da matriz
num_labels = size(Theta2, 1);%pega a quantidade de linhas da matriz

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1); % somente faz p = 0;

h1 = sigmoid([ones(m, 1) X] * Theta1');% adiciona uma coluna de 1 a X e mutiplica pela transposta de Theta1
h2 = sigmoid([ones(m, 1) h1] * Theta2');
[dummy, p] = max(h2, [], 2);% retorna o valor MAXIMO DE CADA LINHA [MAXIMO DE CADA LINHA, INDICE COLUNA]
% pegar o indice COLUNA que contem o valor MAIOR DA LINHA
% =========================================================================
 %Theta1 = [7][10]
% Theta2 = [6][8]

end
