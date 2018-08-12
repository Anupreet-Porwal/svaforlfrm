function [Qobj]= Qobj_extra(w,train_san,Z,lambda2,premask) % function to calculate the value of objective function whenever required
    C=length(w);
    Pval = (Z*w)*Z';
    Qobj = (-1*sum(sum(train_san.*Pval))) + sum(sum(premask.*(log(1+exp(Pval))))) + C*lambda2;
end