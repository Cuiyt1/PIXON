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

extern unsigned int pixon_size_factor;
extern unsigned int pixon_sub_factor;
extern unsigned int pixon_map_low_bound;

class PixonBasis;
class Data;
class DataFFT;
class RMFFT;
class PixonFFT;
class Pixon;

/* 
 *  class for pixon basis functions
 */
class PixonBasis
{
  public:
    static double norm_gaussian;
    static double gaussian(double x, double y, double psize);
    static double gaussian_norm(double psize);
    static double parabloid(double x, double y, double psize);
    static double parabloid_norm(double psize);
    static double tophat(double x, double y, double psize);
    static double tophat_norm(double psize);
    static double triangle(double x, double y, double psize);
    static double triangle_norm(double psize);
    static double lorentz(double x, double y, double psize);
    static double lorentz_norm(double psize);
};

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
    void increase_pixon_min();
    unsigned int get_ipxion_min();

    friend class Pixon;

    unsigned int npixon;  /* number of pixons */
    unsigned int ipixon_min;
    double *pixon_sizes; /* sizes of pixons */
    double *pixon_sizes_num;
  protected:
};

class Pixon
{
  public:
    Pixon();
    Pixon(Data& cont, Data& line, unsigned int npixel,  unsigned int npixon);
    ~Pixon();
    double interp_line(double t);
    double interp_cont(double t);
    double interp_pixon(double t);
    void compute_rm_pixon(const double *x);
    double compute_chisquare(const double *x);
    double compute_mem(const double *x);
    void compute_chisquare_grad(const double *x);
    void compute_chisquare_grad_pixon_low();
    void compute_chisquare_grad_pixon_up();
    void compute_mem_grad(const double *x);
    void compute_mem_grad_pixon_low();
    void compute_mem_grad_pixon_up();
    double compute_pixon_number();
    void reduce_pixon_map_all();
    bool reduce_pixon_map_uniform();
    void increase_pixon_map_all();
    void reduce_pixon_map(unsigned int);
    void increase_pixon_map(unsigned int);
    bool update_pixon_map();
    bool increase_pixon_map();
    void smooth_pixon_map();

    Data cont, line;
    RMFFT rmfft;
    PixonFFT pfft;

    unsigned int npixel;
    unsigned int *pixon_map;
    unsigned int *pixon_map_smooth;
    double *image;
    double *pseudo_image;

    double dt; 
    double chisq;
    double mem;
    double *rmline;
    double *itline; /* interpolation */
    double *residual; 
    double *grad_pixon_low;
    double *grad_pixon_up;
    double *grad_chisq;
    double *grad_mem;
    double *grad_mem_pixon_low;
    double *grad_mem_pixon_up;
    double *resp_pixon;
    double *conv_pixon;

  private:
};

/* pixon functions */
typedef double (*PixonFunc)(double x, double y, double psize);
typedef double (*PixonNorm)(double);
extern PixonFunc pixon_function;
extern PixonNorm pixon_norm;

/* functions for nlopt and tnc */
double func_nlopt(const vector<double> &x, vector<double> &grad, void *f_data);
int func_tnc(double x[], double *f, double g[], void *state);
#endif