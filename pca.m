clear all;
clc;
for i=1:10
    filename=sprintf('D:\\Shen\\Desktop\\myf\\n%d.jpg',i);
    src=imread(filename);
    gray=double(rgb2gray(src));
    if(i==1)
%         M=gray;
        M=imresize(gray,[1,112*92]);
    else
%         M=M+gray;
        M =[M;imresize(gray,[1,112*92])];
    end  
end
% imshow(uint8(M/10))

meanImgS=ones(size(M,1),1)*mean(M);

meanImgSM=M-meanImgS;
covv=meanImgSM'*meanImgSM;
% 
[vm,d]=eig(covv);
k=size(gray,2)/4;

eigV=vm(:,k:size(gray,2));

result=M*eigV;
