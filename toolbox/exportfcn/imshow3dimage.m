function rearrange_data = imshow3dimage(data,ydim)

data = squeeze(data);
if nargin==1
    ydim = ceil(sqrt(size(data,3)));
end
xdim = ceil(size(data,3)/ydim);
[xnum,ynum,znum]=size(data);
count = 1;
rearrange_data=[];
for n=1:1:xdim
    rearrange_data_temp=zeros(xnum,ynum*ydim);
    for m=1:1:ydim
        rearrange_data_temp(:,(ynum*(m-1)+1):ynum*m) = data(:,:,count);
        count = count+1;
        if count > znum
            break;
        end
    end
    rearrange_data = [rearrange_data;rearrange_data_temp];
end
% rearrange_data = abs(rearrange_data);
    
end