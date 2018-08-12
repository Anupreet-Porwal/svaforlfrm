function zobj= zoptimiser_opti(z,zchoice,w,train,i)
% function to calculate the objective function value with ith row of z replaced with zchoice given all the other rows fixed 
    C=length(w);
    n=length(train);
    z(i,:)=zchoice;
    Pval=(z*w)*(z');
    premask = ~isnan(train);
    train(isnan(train))=0;
    zobj = (-1*sum(sum(train.*Pval))) + sum(sum(premask.*(log(1+exp(Pval)))));
end