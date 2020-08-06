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

#include "utilities.hpp"
#include "tnc.h"

using namespace std;


void test();
void test_nlopt();
void run_uniform();
void run();

int main(int argc, char ** argv)
{
  unsigned int pixon_type = 3;

  pixon_size_factor = 1;
  pixon_sub_factor = 1;
  
  switch(pixon_type)
  {
    case 0:  /* Gaussian */
      PixonBasis::norm_gaussian = erf(pixon_size_factor/sqrt(2.0));
      pixon_map_low_bound = pixon_sub_factor - 1;

      pixon_function = PixonBasis::gaussian;
      pixon_norm = PixonBasis::gaussian_norm;
      break;
    
    case 1:  /* parabloid */
      if(pixon_size_factor = 1)
        pixon_sub_factor = 1;
      else 
        pixon_sub_factor = fmax(pixon_sub_factor, pixon_size_factor);
      
      pixon_map_low_bound = pixon_sub_factor - 1;
      pixon_function = PixonBasis::parabloid;
      pixon_norm = PixonBasis::parabloid_norm;
      break;
    
    case 2:  /* top-hat */
      if(pixon_size_factor = 1)
        pixon_sub_factor = 1;
      else 
        pixon_sub_factor = fmax(pixon_sub_factor, pixon_size_factor);
      
      pixon_map_low_bound = pixon_sub_factor - 1;
      pixon_function = PixonBasis::tophat;
      pixon_norm = PixonBasis::tophat_norm;
      break;
    
    case 3:  /* triangle */
      if(pixon_size_factor = 1)
        pixon_sub_factor = 1;
      else 
        pixon_sub_factor = fmax(pixon_sub_factor, pixon_size_factor);
      
      pixon_map_low_bound = pixon_sub_factor - 1;
      pixon_function = PixonBasis::triangle;
      pixon_norm = PixonBasis::triangle_norm;
      break;
    
    default:  /* default */
      PixonBasis::norm_gaussian = erf(pixon_size_factor/sqrt(2.0));
      pixon_map_low_bound = pixon_sub_factor - 1;

      pixon_function = PixonBasis::gaussian;
      pixon_norm = PixonBasis::gaussian_norm;
      break;
  }

  //test_nlopt();
  run_uniform();
  run();
  return 0;
}

void run()
{
  Data cont, line;
  string fcon, fline;
  fcon="data/con.txt";
  cont.load(fcon);
  fline = "data/line.txt";
  line.load(fline);
  
  const unsigned int npixel = 100;
  unsigned int npixon = 10*pixon_sub_factor;
  unsigned int i, iter;
  Pixon pixon(cont, line, npixel, npixon);
  void *args = (void *)&pixon;
  bool flag;
  
  double f, f_old, df, num, num_old, dnum, chisq, chisq_old, dchisq;
  double *image=new double[npixel], *itline=new double[line.size];
  
  /* TNC */
  int rc, maxCGit = npixel, maxnfeval = 2000, nfeval, niter;
  double low[npixel], up[npixel], eta = -1.0, stepmx = 10000.0,
    accuracy = 1.0e-15, fmin = cont.size, ftol = -1.0, xtol = -1.0, pgtol = -1.0,
    rescale = -1.0;
  
  /* NLopt */
  nlopt::opt opt0(nlopt::LN_BOBYQA, npixel);
  vector<double> x(npixel), g(npixel), x_old(npixel);
  opt0.set_min_objective(func_nlopt, args);
  opt0.set_lower_bounds(-100.0);
  opt0.set_upper_bounds(1.0);
  opt0.set_maxeval(10000);
  //opt0.set_xtol_rel(1e-11);
  //opt0.set_ftol_rel(1e-11);
  opt0.set_ftol_abs(1.0e-15);
  opt0.set_xtol_abs(1.0e-15);

  for(i=0; i<npixel; i++)
  {
    low[i] = -100.0;
    up[i] =  1.0;
    x[i] = log(1.0/(npixel * pixon.dt));
  }

  opt0.optimize(x, f);
  //opt1.optimize(x, f);
  rc = tnc(npixel, x.data(), &f, g.data(), func_tnc, args, low, up, NULL, NULL, TNC_MSG_ALL,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);
  
  f_old = f;
  num_old = pixon.compute_pixon_number();
  chisq_old = pixon.chisq;
  cout<<f_old<<"  "<<num_old<<"   "<<chisq_old<<endl;
  memcpy(image, pixon.image, npixel*sizeof(double));
  memcpy(itline, pixon.itline, line.size*sizeof(double));
  memcpy(x_old.data(), x.data(), npixel*sizeof(double));
 
  /* pixel-dependent pixon size */
  iter = 0;
  do
  {
    iter++;
    cout<<"iter:"<<iter<<endl;

    num = pixon.compute_pixon_number();
    
    opt0.optimize(x, f);
    //opt1.optimize(x, f);

    rc = tnc(npixel, x.data(), &f, g.data(), func_tnc, args, low, up, NULL, NULL, TNC_MSG_ALL,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);
    
    chisq = pixon.chisq;
    cout<<f<<"  "<<num<<"  "<<chisq<<endl;

    if(f <= line.size)
    {
      memcpy(image, pixon.image, npixel*sizeof(double));
      memcpy(itline, pixon.itline, line.size*sizeof(double));
      memcpy(x_old.data(), x.data(), npixel*sizeof(double));
      break;
    }
    
    df = f-f_old;
    dnum = num - num_old;
    dchisq = chisq - chisq_old;

    flag = pixon.update_pixon_map();
    if(!flag)
      break;

    num_old = num;
    f_old = f;
    chisq_old = chisq;
    memcpy(image, pixon.image, npixel*sizeof(double));
    memcpy(itline, pixon.itline, line.size*sizeof(double));
    memcpy(x_old.data(), x.data(), npixel*sizeof(double));
  }while(pixon.pfft.get_ipxion_min() >= pixon_map_low_bound); 

  ofstream fout;
  fout.open("data/resp_tnc.txt");
  for(i=0; i<npixel; i++)
  {
    fout<<pixon.dt*i<<"  "<<exp(x_old[i])<<"   "<<image[i]<<"  "<<pixon_function(pixon.dt*i, 300.0, 50.0)<<endl;
  }
  fout.close();

  fout.open("data/line_sim_tnc.txt");
  for(i=0; i<line.size; i++)
  {
    fout<<line.time[i]<<"  "<<itline[i]<<endl;
  }
  fout.close();

  fout.open("data/pixon_map.txt");
  for(i=0; i<npixel; i++)
  {
    fout<<i<<"  "<<pixon.pfft.pixon_sizes[pixon.pixon_map[i]]<<endl;
  }
  fout.close();

  delete[] image;
  delete[] itline;

}

void run_uniform()
{
  Data cont, line;
  string fcon, fline;
  fcon="data/con.txt";
  cont.load(fcon);
  fline = "data/line.txt";
  line.load(fline);

  const unsigned int npixel = 100;
  unsigned int npixon = 10*pixon_sub_factor;
  unsigned int i;
  Pixon pixon(cont, line, npixel, npixon);
  void *args = (void *)&pixon;
  double f, f_old, num_old, num, df, dnum, chisq, chisq_old, dchisq;
  double *image=new double[npixel], *itline=new double[line.size];
 
  int rc, maxCGit = npixel, maxnfeval = 2000, nfeval, niter;
  double low[npixel], up[npixel],
    eta = -1.0, stepmx = 10000.0,
    accuracy = 1.0e-15, fmin = cont.size, ftol = -1.0, xtol = -1.0, pgtol = -1.0,
    rescale = -1.0;

  /* NLopt */
  nlopt::opt opt0(nlopt::LN_BOBYQA, npixel);
  vector<double> x(npixel), g(npixel), x_old(npixel);
  opt0.set_min_objective(func_nlopt, args);
  opt0.set_lower_bounds(-100.0);
  opt0.set_upper_bounds(1.0);
  opt0.set_maxeval(10000);
  //opt0.set_xtol_rel(1e-11);
  //opt0.set_ftol_rel(1e-11);
  opt0.set_ftol_abs(1.0e-15);
  opt0.set_xtol_abs(1.0e-15);
   
  for(i=0; i<npixel; i++)
  {
    low[i] = -100.0;
    up[i] =  1.0;
    x[i] = log(1.0/(npixel * pixon.dt));
    //x[i] = log(1.0/sqrt(2.0*M_PI)/10.0 * exp( - 0.5*pow((pixon.dt*i - 300.0)/10.0, 2.0) ) + 1.0e-10);
  }
  
  opt0.optimize(x, f);

  rc = tnc(npixel, x.data(), &f, g.data(), func_tnc, args, low, up, NULL, NULL, TNC_MSG_ALL,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);
  
  f_old = f;
  num_old = pixon.compute_pixon_number();
  chisq_old = pixon.chisq;
  memcpy(image, pixon.image, npixel*sizeof(double));
  memcpy(itline, pixon.itline, line.size*sizeof(double));
  memcpy(x_old.data(), x.data(), npixel*sizeof(double));
  cout<<f_old<<"  "<<num_old<<"  "<<chisq_old<<endl;
  while(npixon>pixon_map_low_bound+1)
  {
    npixon--;
    cout<<"npixon:"<<npixon<<",  size: "<<pixon.pfft.pixon_sizes[npixon]<<endl;

    pixon.reduce_pixon_map_all();
    num = pixon.compute_pixon_number();
    
    opt0.optimize(x, f);

    rc = tnc(npixel, x.data(), &f, g.data(), func_tnc, args, low, up, NULL, NULL, TNC_MSG_ALL,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);
    
    chisq = pixon.chisq;
    cout<<f<<"  "<<num<<"  "<<chisq<<endl;

    if(f <= line.size)
    {
      memcpy(image, pixon.image, npixel*sizeof(double));
      memcpy(itline, pixon.itline, line.size*sizeof(double));
      memcpy(x_old.data(), x.data(), npixel*sizeof(double));
      break;
    }
    
    df = f-f_old;
    dnum = num - num_old;
    dchisq = chisq - chisq_old;

    if(-df < dnum * (1.0 + 1.0/sqrt(2.0*num)))
      break;

    num_old = num;
    f_old = f;
    chisq_old = chisq;
    memcpy(image, pixon.image, npixel*sizeof(double));
    memcpy(itline, pixon.itline, line.size*sizeof(double));
    memcpy(x_old.data(), x.data(), npixel*sizeof(double));
  }

  ofstream fout;
  fout.open("data/resp_tnc_uniform.txt");
  for(i=0; i<npixel; i++)
  {
    fout<<pixon.dt*i<<"  "<<exp(x_old[i])<<"   "<<image[i]<<"  "<<pixon_function(pixon.dt*i, 300.0, 50.0)<<endl;
  }
  fout.close();

  fout.open("data/line_sim_tnc_uniform.txt");
  for(i=0; i<line.size; i++)
  {
    fout<<line.time[i]<<"  "<<itline[i]<<endl;
  }
  fout.close();

  delete[] image;
  delete[] itline;
}

void test_nlopt()
{
  Data cont, line;
  string fcon, fline;
  fcon="data/con.txt";
  cont.load(fcon);
  fline = "data/line.txt";
  line.load(fline);

  unsigned int npixel = 100, npixon = 8;
  unsigned int i;
  Pixon pixon(cont, line, npixel, npixon);
  nlopt::opt opt0(nlopt::GN_ISRES, npixel);
  nlopt::opt opt1(nlopt::LN_BOBYQA, npixel);
  nlopt::opt opt2(nlopt::LD_SLSQP, npixel);
  vector<double> x(npixel), x0(npixel), x_old(npixel);
  void *args = (void *)&pixon;
  double minf_old, num_old, minf, num;
  double *image=new double[npixel], *itline=new double[line.size];
  
  for(i=0; i<npixel; i++)
  {
    x[i] = log(1.0/(npixel * pixon.dt));
    //x[i] = log(1.0/sqrt(2.0*M_PI)/10.0 * exp( - 0.5*pow((pixon.dt*i - 300.0)/10.0, 2.0) ) + 1.0e-10);
  }
  x0 = x;

  opt0.set_min_objective(func_nlopt, args);
  opt0.set_lower_bounds(-100.0);
  opt0.set_upper_bounds(1.0);
  opt0.set_maxeval(10000);
  //opt0.set_xtol_rel(1e-11);
  //opt0.set_ftol_rel(1e-11);
  opt0.set_ftol_abs(1.0e-15);
  opt0.set_xtol_abs(1.0e-15);

  opt1.set_min_objective(func_nlopt, args);
  opt1.set_lower_bounds(-100.0);
  opt1.set_upper_bounds(1.0);
  opt1.set_maxeval(10000);
  //opt1.set_xtol_rel(1e-11);
  //opt1.set_ftol_rel(1e-11);
  opt1.set_ftol_abs(1.0e-15);
  opt1.set_xtol_abs(1.0e-15);

  opt2.set_min_objective(func_nlopt, args);
  opt2.set_lower_bounds(-100.0);
  opt2.set_upper_bounds(1.0);
  opt2.set_maxeval(10000);
  //opt2.set_xtol_rel(1e-11);
  //opt2.set_ftol_rel(1e-11);
  opt2.set_ftol_abs(1.0e-15);
  opt2.set_xtol_abs(1.0e-15);

  num_old = pixon.compute_pixon_number();
  //opt0.optimize(x, minf_old);
  opt1.optimize(x, minf_old);
  opt2.optimize(x, minf_old);
  cout<<minf_old<<"  "<<num_old<<endl;
  memcpy(image, pixon.image, npixel*sizeof(double));
  memcpy(itline, pixon.itline, line.size*sizeof(double));
  while(npixon>1)
  {
    npixon--;
    cout<<"npixon:"<<npixon<<endl;

    pixon.reduce_pixon_map_all();
    num = pixon.compute_pixon_number();
    //opt0.optimize(x, minf);
    opt1.optimize(x, minf);
    opt2.optimize(x, minf);
    cout<<minf<<"  "<<num<<endl;

    if(minf_old - minf < num - num_old)
      break;

    num_old = num;
    minf_old = minf;
    memcpy(image, pixon.image, npixel*sizeof(double));
    memcpy(itline, pixon.itline, line.size*sizeof(double));
    x_old = x;
  }
  
  ofstream fout;
  double xr;
  fout.open("data/resp_nlopt.txt");
  for(i=0; i<npixel; i++)
  {
    xr = exp(-0.5 * pow(pixon.dt*i-300.0, 2)/(50.0*50.0)) * 1.0/sqrt(2.0*M_PI)/50.0;
    fout<<pixon.dt*i<<"  "<<exp(x_old[i])<<"  "<<image[i]<<"  "<<pixon_function(pixon.dt*i, 300.0, 50.0)<<"  "<<xr<<endl;
  }
  fout.close();

  fout.open("data/line_sim_nlopt.txt");
  for(i=0; i<line.size; i++)
  {
    fout<<line.time[i]<<"  "<<itline[i]<<endl;
  }
  fout.close();
  
  delete[] image;
  delete[] itline;
}


void test()
{
  Data cont, line;
  string fcon, fline;
  fcon={"data/con.txt"};
  cont.load(fcon);
  fline = "data/line.txt";
  line.load(fline);

  unsigned int i, nr = 100, np=13;
  double *resp, *delay,*conv;
  double sigma = 20.0, fft_dx;
  double *pseudo_img, *conv_img;
  unsigned int *pixon_map;
  
  default_random_engine generator;
  uniform_int_distribution<int> distribution(0,np-1);

  resp = new double[nr];
  delay = new double[nr];
  conv = new double[cont.size];

  pseudo_img = new double[nr];
  pixon_map = new unsigned int[nr];
  conv_img = new double[nr];

  fft_dx = (cont.time[1] - cont.time[0]);
  for(i=0; i<nr; i++)
  {
    delay[i] = cont.time[i] - cont.time[0];
    resp[i] = 1.0/sqrt(2.0*M_PI)/sigma * exp(-0.5 * (delay[i] - 200.0) * (delay[i] - 200.0)/sigma/sigma);
  }
  
  RMFFT rmfft(cont);
  rmfft.convolve(resp, nr, conv);
  
  for(i=0; i<nr; i++)
  {
    pseudo_img[i] = resp[i];
    pixon_map[i] = np-1;
  }
  PixonFFT pfft(nr, np);
  pfft.convolve(pseudo_img, pixon_map, conv_img);

  ofstream fout;
  fout.open("data/conv.txt");
  for(i=0; i<cont.size; i++)
  {
    fout<<cont.time[i]<<"  "<<conv[i]<<endl;
  }
  fout.close();
  fout.open("data/resp.txt");
  for(i=0; i<nr; i++)
  {
    fout<<delay[i]<<" "<<resp[i]<<endl;
  }
  fout.close();

  fout.open("data/conv_img.txt");
  for(i=0; i<nr; i++)
  {
    fout<<pseudo_img[i]<<"  "<<conv_img[i]<<endl;
  }
  fout.close();

  delete[] resp;
  delete[] delay;
  delete[] conv;

  delete[] pseudo_img;
  delete[] pixon_map;
  delete[] conv_img;
}