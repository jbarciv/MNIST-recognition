load Trainnumbers.mat;
k=138; %es el indice del numero que se va a dinujar
for i=1:28
    for j=1:28
        digito(i,j)=Trainnumbers.image((i-1)*28+j,k);
    end
end
imshow(digito);
Trainnumbers.label(k)