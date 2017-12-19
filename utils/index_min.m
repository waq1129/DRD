function Is = index_min(M, n)
% find the index for the minimum in a multidimensional array (nd<=10)
[~,I] = min(M(:));
[I1,I2,I3,I4,I5,I6,I7,I8,I9,I10] = ind2sub(size(M),I);
Is = [I1,I2,I3,I4,I5,I6,I7,I8,I9,I10];
Is = Is(1:n);

