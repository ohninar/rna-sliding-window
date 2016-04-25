%% Rede Neural Artificial usando Cross-Validation
% Aprendizagem Supervisionada para Sistema Classificador de caracteres
% Tecnologia Redes Neurais
% Autor: Bruno Adonis de Sa
%------ Observações ----------------------------------------------------------
% Parametros que devem ser variados afim de obter melhor percentual de acerto:
%
% hidden_layer_size: quantidade de neuronios da camada oculta
%
% lambdaInput: recomenda-se valores entre 0 e 1, parametro de aprendizado,
% define o quão rapido e o quão devagar deve ser o aprendizado
%
% numFolds: Define a quantidade de K-folds que deve ser utilizado na validação cruzada.
% recomenda-se para testes, utilizar 1 ou 2, uma vez escolhido os melhores parametros,
% utiliza-se 5 a 10 folds.
%  
% qtd_iteracoes: define quantas iterações devem ser feitas durante o treinamento
% recomenda-se valores entre 100 e 1000, uma vez escolhido os melhores parametros
% utiliza 1000 a infinity... Depende do problema e o decrescimento do 'cost'.
%
% Os demais parametros para este problema especifico não devem ser variados.
 
%% Initialization
clear all; close all; clc
%---- load dataset --------------------------------------------------------
load dataset_graynorm.mat;
samples = graynorm;
% load dataset_blackWhitenorm.mat;
% samples = blackWhitenorm;

%----- Definindo constantes ------------------------------------------------
input_layer_size  = 900;    % features
hidden_layer_size = 100;    % hidden units
num_labels = 62;            % labels 
%valor de lambda
lambdaInput = 0.4;
%limiar da predição
limiar = 0.7;
%Selecionar o maior percentual durante a validacao cruzada
temporario = 0;
%numero de Kfold a ser utilizado
numFolds = 2;
%quantidade de iterações
qtd_iteracoes = 1000;

%----- Dataset -----------------------------------------------------------
%numero de colunas 
n_col = size(samples,2);
%obtendo os vetores colunas (caracteristicas)
featuresMatriz = samples(:,(1:n_col-1));
%obtendo o vetor etiqueta
classificationVector =  samples(:,n_col);

%---------------- Plot DATA ----------------------------------------------
% m = size(featuresMatriz, 1);
% sel = randperm(size(featuresMatriz, 1));
% % sel = sel(1:2024);
% displayData(featuresMatriz(sel,:));

%---- Separando o conjunto em treino e teste ----------------------------
%porcentagem do conjunto de teste (20% de todo o Dataset)
percentage = 0.20;
%dividindo os dados aleatoriamente[ treino, teste]
[train, test] = crossvalind('HoldOut', classificationVector,percentage);%RANDOM
%Obtendo conjunto de treino.
TrainingSample = featuresMatriz(train, :); 
TrainingLabel = classificationVector(train, :);
%Obtendo conjunto de teste.
TesteSample = featuresMatriz(test, :);
TesteLabel = classificationVector(test, :);
%função gera indices aleatorios de 1 a numFolds, para ser feita a validação
%cruzada
indices = crossvalind('kfold', TrainingLabel,numFolds);%RANDOM
%----- Resultados -----------------------------------------------------
percentualAcerto_CV = zeros(numFolds,1);
percentualAcerto_Teste = zeros(numFolds,1);
percentualAcerto_Teste_Limiar = zeros(numFolds,1);
backup_Thetas = zeros((input_layer_size +1) * hidden_layer_size + (hidden_layer_size +1) * num_labels,numFolds);

tic %obtem o tempo inicial
for i=1:numFolds
    
    %Random initialization of weights (randInitializeWeights.m)
    fprintf('\nInitializing Neural Network Parameters ...\n')
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
    initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];  % Unroll parameters
  
    %Particao a ser testada
    TestingFoldSample = TrainingSample(indices==i,:); 
    TestingFoldLabel = TrainingLabel(indices==i,:);
    %As demais será utilizada para treino
    TrainingFoldSample = TrainingSample(indices~=i,:);
    TrainingFoldLabel = TrainingLabel(indices~=i,:);
    
    fprintf('\nTraining Neural Network... \n')
    %  After you have completed the assignment, change the MaxIter to a larger
    %  value to see how more training helps.
    options = optimset('MaxIter',qtd_iteracoes);
    %  You should also try different values of lambda
    lambda = lambdaInput;
    % Create "short hand" for the cost function to be minimized
    costFunction = @(p) nnCostFunction(p, ...
                                       input_layer_size, ...
                                       hidden_layer_size, ...
                                       num_labels, TrainingFoldSample, TrainingFoldLabel, lambda);
                                   
    % Now, costFunction is a function that takes in only one argument (the
    % neural network parameters)
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
    % armazena todos os thetas da validacao cruzada
    backup_Thetas(:,i) = nn_params;
    
    % Obtain Theta1 and Theta2 back from nn_params
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));

    fprintf('CrossValidation fold: %f \n',i);
    featuresM = TestingFoldSample;
    classificationV =  TestingFoldLabel;

    pred = predict(Theta1, Theta2,featuresM);
    percentualAcerto_CV(i,:) = mean(double(pred == classificationV)) * 100;
    fprintf('\nTraining Set Accuracy sem limiar %d-Fold: %f\n',i, percentualAcerto_CV(i,:));                                

    predMaior = predict(Theta1, Theta2,TesteSample);
    percentualAcerto_Teste(i,:) = mean(double(predMaior == TesteLabel)) * 100;
    fprintf('\nTest Set Accuracy sem limiar: %f\n', percentualAcerto_Teste(i,:));   

    [dummy, predMaiorL] = predictLimiarNorm(Theta1, Theta2,TesteSample,limiar);
    percentualAcerto_Teste_Limiar(i,:) = mean(double(predMaiorL == TesteLabel)) * 100;
    fprintf('\nTest Set Accuracy com limiar: %f\n', percentualAcerto_Teste_Limiar(i,:));  

    if(percentualAcerto_Teste(i,:) > temporario)
        temporario = percentualAcerto_Teste(i,:);
        Theta1Maior = Theta1; %armazena os thetas de maior percentual de acerto
        Theta2Maior = Theta2;
    end
%      pause;
end
toc % finaliza a contagem e exibe a duração total do treino/teste.

%calcula a média dos K-fold
mediaFinal_CV = mean(double(percentualAcerto_CV));
%calcula a média dos K testes contra o test set
mediaFinal_Teste = mean(double(percentualAcerto_Teste));
%calcula a média dos K testes contra o test set com limiar
mediaFinal_Teste_L = mean(double(percentualAcerto_Teste_Limiar));