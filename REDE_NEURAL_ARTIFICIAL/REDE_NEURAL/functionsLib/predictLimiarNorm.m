function [dummy, p]= predictLimiarNorm(Theta1, Theta2, X, limiar)
%PREDICT Predict the label of an input given a trained neural network
%   [dummy, p]= predictLimiarNorm(Theta1, Theta2, X, limiar) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

    % Useful values
    m = size(X, 1);%pega a quantidade de linhas da matriz
    num_labels = size(Theta2, 1);%pega a quantidade de linhas da matriz

    % You need to return the following variables correctly 
    p = zeros(size(X, 1), 1); % somente faz p = 0;

    h1 = sigmoid([ones(m, 1) X] * Theta1');
    h2 = sigmoid([ones(m, 1) h1] * Theta2');
    % Min-max normalize
    Xmin = min(h2, [], 2); 
    Xmax = max(h2, [], 2);
    %New = X./somaX
    h3 = zeros(m,size(h2, 2));

    for l=1:m,
        h3(l,:) = h2(l,:) ./ sum(h2(l,:));
    end

    [dummy, p] = max(h3, [], 2);
	
    for i=1:m,
        if dummy(i) >= limiar,
            p(i) = p(i);
        else
            p(i) = 0;
        end
    end

end
