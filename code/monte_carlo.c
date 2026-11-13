#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
static uint64_t state;
static uint64_t next(void){uint64_t z=(state+=UINT64_C(0x9e3779b97f4a7c15));z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);return z^(z>>31);}
static double uniform(void){return ((next()>>11)+0.5)*0x1.0p-53;}
int main(void){
 double rates[]={.1,5,20};
 puts("rate,seed,block,cycles,time,log_growth,exponent");
 for(int ri=0;ri<3;ri++)for(int seed=1;seed<=2;seed++){
  state=180018+seed;double x=1,y=0,r=rates[ri];
  for(int block=-5;block<100;block++){
   double elapsed=0,growth=0;
   for(int n=0;n<20000;n++){
    double t0=-log(uniform())/r,t1=-log(uniform())/r;
    double c=cos(t0),s=sin(t0),xx=c*x-s*y,yy=s*x+c*y;
    double z=exp(-4*t1)*yy,norm=hypot(xx,z);
    x=xx/norm;y=z/norm;
    growth+=-.75*t0+1.25*t1+log(norm);elapsed+=t0+t1;
   }
   if(block>=0)printf("%.17g,%d,%d,20000,%.17g,%.17g,%.17g\n",r,seed,block,elapsed,growth,growth/elapsed);
  }
 }
 return 0;
}
