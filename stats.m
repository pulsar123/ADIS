clear std;
clf;

dirname = 'I:\NINA\Canon6D\2026-06-07\NGC 5915\process\' # Working directory with the trailing slash

hold on;

xlabel('R');
ylabel('p, std');
xlim([0 200]);

listname = [dirname 'list_fine.dat'];
[n, Jx, Jy, x, y, p, i] = textread (listname, "%d %f %f %d %d %f %d");

R = sqrt(Jx.^2+Jy.^2);

scatter(R, p, 1);


statsname = [dirname 'stats_fine.dat'];
[n, p, x, y, Jx, Jy, p_min, Npix, M, dx, dJ] = textread (statsname, "%d %f %d %d %d %d %f %d %f %f %f");

R = sqrt(Jx.^2+Jy.^2);

scatter(R, p, 3);

hold off;
