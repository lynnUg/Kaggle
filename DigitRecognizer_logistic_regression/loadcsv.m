function [X, y]=loadcsv(fname,sname)

d1=csvread(fname);
Xin=d1(:,2:end);
yin=d1(:,1);
yin(find(yin==0))=10;

m = size(Xin, 1);

save(sname,"Xin","yin");
endfunction