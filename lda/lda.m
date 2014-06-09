function [D, W_lda] = lda(X,y) 
  dimension = columns(X);
  labels = unique(y);
  C = length(labels);
  Sw = zeros(dimension,dimension);
  Sb = zeros(dimension,dimension);
  mu = mean(X);

  for i = 1:C
    Xi = X(find(y == labels(i)),:);
    n = rows(Xi);
    mu_i = mean(Xi);
    XMi = bsxfun(@minus, Xi, mu_i);
    Sw = Sw + (XMi' * XMi );
    MiM =  mu_i - mu;
    Sb = Sb + n * MiM' * MiM; 
  endfor

  [W_lda, D] = eig(Sw\Sb);
  [D, i] = sort(diag(D), 'descend');
  W_lda = W_lda(:,i);
end
