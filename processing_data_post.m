function [train,reconst]=processing_data_post(X_train,meanp,stdp,W)
    data = X_train;  
    [D, N] = size(data);   
    pnproj = W*data;
    reconstructed=W'*pnproj;
    reconst=reconstructed;
    for i=1:N       
        p_reconstructed(:,i)=reconstructed(:,i).*stdp+meanp;
    end
    reconst = p_reconstructed;
    train = pnproj;
end 