/*
 *  PIXON
 *  A Pixon-based method for reconstructing velocity-delay map in reverberation mapping.
 * 
 *  Yan-Rong Li, liyanrong@mail.ihep.ac.cn
 * 
 */
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <random>
#include <nlopt.hpp>
#include <fftw3.h>

#include "proto.hpp"

using namespace std;

int main(int argc, char ** argv)
{
  if(argc < 2)
  {
    cout<<"Please input pixon type."<<endl;
    exit(0);
  }
  int pixon_type = atoi(argv[1]);
  cout<<"Pixon type: "<<pixon_type<<","<<PixonBasis::pixonbasis_name[pixon_type]<<endl;
  
  bool fix_bg = true;
  double bg = 0.0;

  run(pixon_type, fix_bg, bg);
}