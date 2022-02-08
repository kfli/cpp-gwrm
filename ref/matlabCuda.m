clear all
%% Init
T = 0.5;
N = 5;
M = 20;

ci = ones((M+2),1,'gpuArray');
tic; ci = initcond(M+2,ci); toc

a = ones((N+1)*(M+1),1,'gpuArray');
tic; c = matlab_test(T,N+1,M+1,a,ci); toc

a1 = ones(N+1,M+1);
b1 = ones(N+1,M+1);
tic; c1 = chebyprod_m(N,M,a1,b1); toc

%% Call AndersonAcceleration
INPUT.mMax=10000;
INPUT.itmax=10000;
INPUT.rtol=1e-6;
INPUT.atol=1e-5;
INPUT.droptol=1e6;
INPUT.beta=0.1;
INPUT.AAstart=0;
MP = struct;
x0 = zeros((N+1)*(M+1),1);
tic;U = AndersonAcceleration(T,N+1,M+1,x0,MP,INPUT,ci); toc

%% Call SIR
%tic; U = SIR(T,N+1,M+1,x0,0,ci); toc

%% Plot
u = zeros(N+1,M+1);

for i=1:N+1
    for j=1:M+1
        u(i,j)=U((j-1)*(N+1)+i);
    end
end
%disp(u);

px = 50; pt = 50;
XX=(0:1/(px-1):1);
TT=(0:T/(pt-1):T);

f=zeros(pt,px);
for k=1:pt
    for l=1:px
        f_prel = zeros(1,N+1);
        for i=1:N+1
         f_prel(i)=clenshaw(u(i,1:M+1),(XX(l)-0.5)/0.5,M+1);
        end
        f(k,l)=clenshaw(f_prel(1:N+1),(TT(k)-T/2)/(T/2),N+1);
    end
end

%%
surf(XX,TT,f)
title({'1-D Burgers'' equation (\kappa = 0.01)';['time= ',num2str(T)]});
view(-40,50)
xlabel('Spatial co-ordinate (x) \rightarrow');
ylabel('Temporal co-ordinate (t) \leftarrow');
zlabel('Transport property profile (u)');