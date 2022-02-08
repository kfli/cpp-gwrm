function C = clenshaw(A,x,N)
B = zeros(1,N);
A(1)    = A(1)/2;
B(N)   = A(N);
B(N-1) = A(N-1)+2*x*A(N);
for k=N-2:-1:1
    B(k) = A(k) + 2*x*B(k+1) - B(k+2);
end
C = B(1) - x*B(2);
