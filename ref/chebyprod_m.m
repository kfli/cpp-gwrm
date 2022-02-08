function c=chebyprod_m(K,L,a,b)
c = zeros(K+1,L+1);
for k = 0:K
  for l = 0:L
     q = 0;
     for i = 0:k
        r = 0;
        for j = 0:l
           r = r + a(i+1,j+1) * b(k-i+1,l-j+1);
        end
        for j = 1:L-l
           r = r + a(i+1,j+1) * b(k-i+1,j+l+1) + a(i+1,j+l+1) * b(k-i+1,j+1);
        end
        q = q + r;
     end
     for i = 1:K-k
        r = 0;
        for j = 0:l
           r = r + a(i+1,j+1) * b(i+k+1,l-j+1) + a(i+k+1,j+1) * b(i+1,l-j+1);
        end
        for j = 1:L-l
           r = r + a(i+1,j+1) * b(i+k+1,l+j+1)...
                   +  a(i+1,j+l+1) * b(i+k+1,j+1)...
                   +  a(i+k+1,j+1) * b(i+1,j+l+1)...
                   +  a(i+k+1,+j+l+1) * b(i+1,j+1);
        end
        q = q + r;
     end
     c(k+1,l+1) = 0.25*q;
  end
end
