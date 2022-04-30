clf;
figure(1);
X = [200:100:1000];
Y = bar(X, [10.606837, 42.793103
     5.479452, 23.529411
     9.435000, 38.510206
     17.715556, 83.041665
     4.906977, 22.607142
     3.448905, 16.293103
     4.369906, 19.095890
     4.138810, 18.037037
     5.383812, 22.413044]);
set (Y(1), "facecolor", "r");
set (Y(2), "facecolor", "g");
set(gca, "fontsize", 15);
%title ("GPU acceleration (Parallel CPU) of 1D Chebyshev series product (Tesla K80)");
h = legend("Monolithic kernel", "Stride kernel");
set(h, "FontSize", 12);
%legend (h, "location", "northeastoutside");
ylabel ("Speedup",'fontsize',15);

figure(2);
X = [200:100:1000];
Y = bar(X, [0.923729, 3.303030
     1.302632, 5.351351
     1.225000, 5.297297
     1.543147, 6.909091
     1.937778, 8.549020
     3.690909, 16.111112
     2.678322, 11.432836
     2.959877, 12.618422
     3.512894, 14.255814]);
set (Y(1), "facecolor", "r");
set (Y(2), "facecolor", "g");
set(gca, "fontsize", 15);
%title ("GPU acceleration (Parallel CPU) of 1D Chebyshev series product (Tesla K80)");
h = legend("Monolithic kernel", "Stride kernel");
set(h, "FontSize", 12);
%legend (h, "location", "northeastoutside");
ylabel ("Speedup",'fontsize',15);

figure(3);
X = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200];
Y = bar(X, [4.384053, 1997.666673
            455.387093, 2823.400010
            1449.800040, 9665.333014
            3825.047572, 20123.750022
            3860.931813, 42470.250386
            9576.230776, 62245.499500
            18172.070135, 127204.493009
            21073.545306, 173856.750570
            22077.100537, 220771.000347
            22763.395498, 218528.610028]);
set (Y(1), "facecolor", "r");
set (Y(2), "facecolor", "g");
set(gca, "fontsize", 15);
%title ("GPU acceleration (Parallel CPU) of 2D Chebyshev series product (Tesla K80)");
h = legend("Monolithic kernel", "Stride kernel");
set(h, "FontSize", 12);
ylabel ("Speedup",'fontsize',15);
