#ifndef _PIXON_CONT_HPP

#define _PIXON_CONT_HPP

#include "utilities.hpp"

/* class PixonCont */
class PixonCont:public Pixon
{
  public:
    PixonCont();
    PixonCont(Data& cont_data_in, Data& cont_in, Data& line_data_in, int npixel_in,  
              int npixon_in, int npixon_in_cont, int ipositive_in=0);
    ~PixonCont();
    void compute_cont(const double *x);
    void compute_rm_pixon(const double *x);
    double compute_chisquare(const double *x);
    double compute_chisquare_cont(const double *x);
    void compute_chisquare_grad(const double *x);
    void compute_chisquare_grad_cont(const double *x);
    double compute_mem(const double *x);
    double compute_mem_cont(const double *x);
    void compute_mem_grad(const double *x);
    void compute_mem_grad_cont(const double *x);
    double compute_pixon_number_cont();
    void reduce_ipixon_cont();

    Data cont_data;  /* continuum data */
    PixonUniFFT pfft_cont; /* uniform pixon, for continuum */
    RMFFT rmfft_pixon;
    
    double chisq_cont, mem_cont, chisq_line, mem_line;

    double *residual_cont;   /* residual for continuum */
    int ipixon_cont;   /* pixon map  for continuum */
    double *image_cont;
    double *pseudo_image_cont;
    double *grad_chisq_cont;
    double *grad_mem_cont;

  private:
};

double func_nlopt_cont(const vector<double> &x, vector<double> &grad, void *f_data);
int func_tnc_cont(double x[], double *f, double g[], void *state);

double func_nlopt_cont_rm(const vector<double> &x, vector<double> &grad, void *f_data);
int func_tnc_cont_rm(double x[], double *f, double g[], void *state);
#endif