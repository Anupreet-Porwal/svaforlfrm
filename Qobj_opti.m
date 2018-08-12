function [Qobj]= Qobj_opti(w,train,Z,lambda2) % function to calculate the value of objective function whenever required
    C=length(w);
    Pval = (Z*w)*Z';
    premask = ~isnan(train);
    train(isnan(train))=0;
    Qobj = (-1*sum(sum(train.*Pval))) + sum(sum(premask.*(log(1+exp(Pval))))) + C*lambda2;
end