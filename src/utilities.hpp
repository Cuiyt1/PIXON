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

#define EPS (1.0e-50)

using namespace std;

enum PRIOR_TYPE {GAUSSIAN=1, UNIFORM=2};

extern int pixon_size_factor;
extern int pixon_sub_factor;
extern int pixon_map_low_bound;

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
    static double coeff1_modified_gaussian;
    static double coeff2_modified_gaussian;
    static double norm_modified_gaussian;
    static double gaussian(double x, double y, double psize);
    static double gaussian_norm(double psize);
    static double modified_gaussian(double x, double y, double psize);
    static double modified_gaussian_norm(double psize);
    static double parabloid(double x, double y, double psize);
    static double parabloid_norm(double psize);
    static double tophat(double x, double y, double psize);
    static double tophat_norm(double psize);
    static double triangle(double x, double y, double psize);
    static double triangle_norm(double psize);
    static double lorentz(double x, double y, double psize);
    static double lorentz_norm(double psize);
    static string pixonbasis_name[];
};

/* 
 * Data class for light curves.
 */
class Data 
{
  public:
    Data();
    Data(int n);
    ~Data();
    /* copy constructor */
    Data(Data& data);
    /* operator = reload */
    Data& operator = (Data& data);
    void set_size(int n);
    /* load data from a file */
    void load(const string& fname);
    void set_data(double *data);
    void set_norm(double norm_in);
    void normalize();
    /* variables */
    int size;
    double norm;
    double *time, *flux, *error;
};

/* class to FFT Data */
class DataFFT
{
  public:
    DataFFT();
    /* constructor, fft_dx is space width of the grid */
    DataFFT(int nd_in, double fft_dx, int npad_in=20);
    /* copy constructor, cont is continuum light curve */
    DataFFT(Data& cont, int npad_in=20);
    /* operator = reload */
    DataFFT& operator = (DataFFT& df);
    /* destructor */
    ~DataFFT();
    /* convolution with resp, output to conv */
    void convolve_simple(double *conv);
    double get_fft_norm(){return fft_norm;}
    void set_resp_real(const double *resp, int nall, int ipositive);
 
  protected:
    int nd, npad, nd_fft, nd_fft_cal;
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
    RMFFT(int n, double dx, int npad_in = 20);
    RMFFT(int n, double *cont, double dx, int npad_in = 20);
    RMFFT(Data& cont, int npad_in = 20);
    /* destructor */
    ~RMFFT(){}
    /* set data using cont */
    void set_data(Data& cont);
    /* set data using array */
    void set_data(double *cont, int n);
    /* convolution with resp, output to conv */
    void convolve(const double *resp, int n, double *conv);
    void convolve_bg(const double *resp, int n, double *conv, double bg = 0.0);
    void convolve_bg(const double *resp, int n, int ipositive, double *conv, double bg = 0.0);

    friend class Pixon;
  private:
};

/* class to do Pixon FFT, inherits DataFFT class */
class PixonFFT:public DataFFT
{
  public:
    PixonFFT();
    PixonFFT(int npixel, int npixon);
    ~PixonFFT();
    void convolve(const double *pseudo_img, int *pixon_map, double *conv);
    void convolve_pixon_diff_low(const double *pseudo_img, int *pixon_map, double *conv);
    void convolve_pixon_diff_up(const double *pseudo_img, int *pixon_map, double *conv);
    /* reduce the minimum pixon size */
    void reduce_pixon_min();
    void increase_pixon_min();
    int get_ipxion_min();

    friend class Pixon;

    int npixon;  /* total number of pixons */
    int ipixon_min; /* minimum pixon index */
    double *pixon_sizes; /* pixon sizes */
    double *pixon_sizes_num; /* number of pixon at each size */

    double *conv_tmp;
  protected:
};

/* class to do uniform pixon FFT */
class PixonUniFFT:public DataFFT
{
  public:
    PixonUniFFT();
    PixonUniFFT(int npixel, int npixon);
    ~PixonUniFFT();
    void convolve(const double *pseudo_img, int ipixon, double *conv);
    /* reduce the minimum pixon size */
    void reduce_pixon_min();
    void increase_pixon_min();
    int get_ipxion_min();

    friend class Pixon;

    int npixon;  /* total number of pixons */
    int ipixon_min; /* minimum pixon index */
    double *pixon_sizes; /* pixon sizes */
  protected:
};

/* class Pixon */
class Pixon
{
  public:
    Pixon();
    Pixon(Data& cont_in, Data& line_in, int npixel_in,  int npixon_in, int ipositive_in=0);
    ~Pixon();
    double interp_image(double t);
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
    void reduce_pixon_map(int);
    void increase_pixon_map(int);
    bool update_pixon_map();
    bool increase_pixon_map();

    Data cont, line;
    RMFFT rmfft;
    PixonFFT pfft;

    int npixel;   /* number of pixels */
    int *pixon_map;   /* pixon map */
    bool *pixon_map_updated;  /* pixons updated */
    double *image;           /* image */
    double *pseudo_image;    /* pseudo image */

    double tau0;
    int ipositive;

    double dt;          /* time interval of continuum, image grid */
    double chisq;       /* chi square */
    double mem;         /* entropy */
    double bg;          /* background */
    double *rmline;     /* reverberated line */
    double *itline;     /* interpolation of line to observed epochs */
    double *residual;   /* residual */
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