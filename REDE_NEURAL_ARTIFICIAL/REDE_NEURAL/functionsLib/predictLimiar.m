function p = predictLimiar(Theta1, Theta2, X, limiar)
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
    [dummy, p] = max(h2, [], 2);% retorna o valor MAXIMO DE CADA LINHA
    %[MAXIMO DE CADA LINHA(dummy), INDICE COLUNA(p)]
    % =========================================================================
    %percorro o valor do acerto encontrado e se não for maior que a limiar,
    %seto como zero, caso seja maior, continua o mesmo valor.
    for i=1:m,
        if dummy(i) >= limiar,
            p(i) = p(i);
        else
            p(i) = 0;
        end
    end

end
