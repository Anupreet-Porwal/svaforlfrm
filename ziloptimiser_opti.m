function zobj= ziloptimiser_opti(z,zchoice,w,train,i,l,lambda2) % function to calculate the objective function value with ith row of z replaced with zchoice given all the other rows fixed 
    C=size(z,2);
    n=length(train);
    z(i,l)=zchoice;
    did_we_lose=(sum(z(:,l))-z(i,l))==0;
    Pval=(z*w)*z';
    premask = ~isnan(train);
    train(isnan(train))=0;
    zobj = (-1*sum(sum(train.*Pval))) + sum(sum(premask.*(log(1+exp(Pval))))) + (C-did_we_lose)*lambda2;
end
