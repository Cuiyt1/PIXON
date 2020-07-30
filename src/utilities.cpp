/*
 *  PIXON
 *  A Pixon-based method for reconstructing velocity-delay map in reverberation mapping.
 * 
 *  Yan-Rong Li, liyanrong@mail.ihep.ac.cn
 * 
 */
#include <iostream>
#include <fstream> 
#include <vector>
#include <iomanip>
#include <cstring>

#include "utilities.hpp"

using namespace std;

void Data::load(const string& fname)
{
  ifstream fin;
  string line;
  unsigned int i;
  
  /* first determine number of lines */
  fin.open(fname);
  i = 0;
  while(1)
  {
    getline(fin, line);
    if(fin.good())
    {
      i++;
    }
    else
    {
      break;
    }
  }

  /* delete old data if sizes do not match */
  if(i != size)
  {
    delete[] time;
    delete[] flux;
    delete[] error;
    size = i;
  }
  cout<<"file \""+fname+"\" has "<<size<<" lines."<<endl;

  /* allocate memory */
  time = new double[size];
  flux = new double[size];
  error = new double[size];

  /* now read data */
  fin.clear();  // clear flags
  fin.seekg(0); // go to the beginning
  for(i=0; i<size; i++)
  {
    fin>>time[i]>>flux[i]>>error[i];
    if(fin.fail())
    {
      cout<<"# Error in reading the file \""+fname+"\", no enough points in line "<<i<<"."<<endl;
      exit(0);
    }
  }
  fin.close();
}

double func(const vector<double> &x, vector<double> &grad, void *f_data)
{
  Pixon *pixon = (Pixon *)f_data;
  double chisq;

  pixon->compute_rm_pixon(x);
  if (!grad.empty()) 
  {
    pixon->chisquare_grad(x, grad);
  }
  chisq = pixon->chisquare(x);
  return chisq;
}