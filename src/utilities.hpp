/*
 *  PIXON
 *  A Pixon-based method for reconstructing velocity-delay map in reverberation mapping.
 * 
 *  Yan-Rong Li, liyanrong@mail.ihep.ac.cn
 * 
 */
#ifndef _UTILITIES_HPP

#define _UTILITIES_HPP

#include <iostream>
#include <fstream> 
#include <vector>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <random>
#include <nlopt.hpp>
#include <fftw3.h>

#include "tnc.h"

using namespace std;

class Data;
class DataFFT;
class RMFFT;
class PixonFFT;
class Pixon;

/* 
 * Data class for light curves.
 */
class Data 
{
  public:
    Data();
    Data(unsigned int n);
    ~Data();
    /* copy constructor */
    Data(Data& data);
    /* operator = reload */
    Data& operator = (Data& data);
    /* load data from a file */
    void load(const string& fname);
    /* variables */
    unsigned int size;
    double *time, *flux, *error;
};

/* class to FFT Data */
class DataFFT
{
  public:
    DataFFT();
    /* constructor, fft_dx is space width of the grid */
    DataFFT(unsigned int nd, double fft_dx, unsigned int npad=20);
    /* copy constructor, cont is continuum light curve */
    DataFFT(Data& cont, unsigned int npad = 20);
    /* operator = reload */
    DataFFT& operator = (DataFFT& df);
    /* destructor */
    ~DataFFT();
    /* convolution with resp, output to conv */
    void convolve_simple(double *conv);
    double get_fft_norm(){return fft_norm;}
 
  protected:
    unsigned int nd, npad, nd_fft, nd_fft_cal;
    double fft_norm;
    fftw_complex *data_fft, *resp_fft, *conv_fft;
    double *data_real, *resp_real, *conv_real;
    fftw_plan pdata, presp, pback;
};

/* 
 * class to do RM FFT, inherits DataFFT class 
 * 
 * The FFT of continuum ligh curve only need to calculate once, 
 * so do that at initialization 
 * 
 */
class RMFFT:public DataFFT
{
  public:
    /* default constructor */
    RMFFT(){}
    /* constructor */
    RMFFT(unsigned int n, double *cont, double norm);
    RMFFT(Data& cont);
    /* destructor */
    ~RMFFT(){}
    /* convolution with resp, output to conv */
    void convolve(const double *resp, unsigned int n, double *conv);
    
    friend class Pixon;
  private:
};

/* class to do Pixon FFT, inherits DataFFT class */
class PixonFFT:public DataFFT
{
  public:
    PixonFFT();
    PixonFFT(unsigned int npixel, unsigned int npixon);
    ~PixonFFT();
    void convolve(const double *pseudo_img, unsigned int *pixon_map, double *conv);
    /* reduce the minimum pixon size */
    void reduce_pixon_min();

    friend class Pixon;

  protected:
    unsigned int npixon;  /* number of pixons */
    unsigned int ipixon_min;
    double *pixon_sizes; /* sizes of pixons */
};

class Pixon
{
  public:
    Pixon();
    Pixon(Data& cont, Data& line, unsigned int npixel,  unsigned int npixon);
    ~Pixon();
    double interp(double t);
    void compute_rm_pixon(const vector<double> &x);
    void compute_rm_pixon(const double *x);
    double chisquare(const vector<double> &x);
    double chisquare(const double *x);
    void chisquare_grad(const vector<double> &x, vector<double> &grad);
    void chisquare_grad(const double *x, double *grad);
    double compute_pixon_number();
    void update_pixon_map();

    Data cont, line;
    RMFFT rmfft;
    PixonFFT pfft;

    unsigned int npixel;
    unsigned int *pixon_map;
    double *image;
    double *pseudo_image;

    double dt; 
    double *rmline;
    double *itline; /* interpolation */
    double *residual; 

  private:
};

double func_nlopt(const vector<double> &x, vector<double> &grad, void *f_data);
double pixon_function(double x, double y, double psize);
tnc_function func_tnc;
int func_tnc(double x[], double *f, double g[], void *state);
#endif