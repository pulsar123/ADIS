
// Finding clouds clouse to the real asteroid (ANNOTATE output of ASTAP)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define NMAX  1000 // maximum number of clouds and annotations

int main(int argc, char **argv)
{

	float xa[NMAX];
	float ya[NMAX];
	char name[NMAX][80];


	if (argc != 4)
	{
		printf("Syntax: %s stats.dat annotate.txt r_min\n", argv[0]);
		exit(0);
	}
	
	FILE *fstats = fopen(argv[1],"r");
	FILE *fannot = fopen(argv[2],"r");
	
	float r_min = atof(argv[3]);
	
	char buffer[256];
	char *token;
    char *token2;


	// Reading the annotation file
	int i = -1;
    while (fgets(buffer, sizeof(buffer), fannot) != NULL) 
	{
      token = strtok(buffer, "'");

	  if (token == NULL) {
		  continue;
	  }

	  // This should contain ;-delimted numbers
	  token = strtok(NULL, "'");
	  if (token == NULL) {
		  continue;
	  }

	  token2 = strtok(token, ";");
	  if (token2 == NULL)
		  continue;
	  float y1 = atof(token2);

	  token2 = strtok(NULL, ";");
	  if (token2 == NULL)
		  continue;
	  float x1 = atof(token2);

	  token2 = strtok(NULL, ";");
	  if (token2 == NULL)
		  continue;
	  float y2 = atof(token2);

	  token2 = strtok(NULL, ";");
	  if (token2 == NULL)
		  continue;
	  float x2 = atof(token2);

	  token2 = strtok(NULL, ";");
	  if (token2 == NULL)
		  continue;

	  token2 = strtok(NULL, ";");
	  if (token2 == NULL)
		  continue;
	  
	  i++;
	  strcpy(name[i],token2);
	  
	  xa[i] = (x1+x2)/2;
	  ya[i] = (y1+y2)/2;
	
//	  printf("%f %f %s\n",xa[i],ya[i],name[i]);     
    }
	
	int NA = i + 1;
	
	// Reading the stats file
	int icloud = -1;
    while (fgets(buffer, sizeof(buffer), fstats)) 
	{
		icloud++;
		char *ptr = buffer;
		float number;
		int bytes_read;

		// Use %n to track how far along the line buffer sscanf has parsed
    	i = 0;
		float xc, yc;
		while (sscanf(ptr, "%f%n", &number, &bytes_read) == 1) 
		{
			i++;
			ptr += bytes_read; // Move pointer forward past the read number
			if (i == 3)
				xc = number;
			if (i == 4)
			{
				yc = number;
				break;
			}
		}
		
		float rmin = 1e30;
		int jmax = 0;
		for (int j=0; j<NA; j++)
		{
			float r = sqrt(pow(xa[j]-xc,2)+pow(ya[j]-yc,2));
			if (r < rmin)
			{
				rmin = r;
				jmax = j;
			}
		}
		
		if (rmin < r_min)
		{
			buffer[strcspn(buffer,"\n")] = 0;
			printf ("%d %s %f: %s\n", jmax, name[jmax], rmin, buffer);
		}
		
	}
	
	
	



	fclose(fstats);
	fclose(fannot);

}


