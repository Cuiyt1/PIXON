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

/*==================================================================*/
/* class Data */
Data::Data()
{
  size = 0;
  time = flux = error = NULL;
}

Data::Data(unsigned int n)
{
  if(n > 0)
  {
    size = n;
    time = new double[size];
    flux = new double[size];
    error = new double[size];
  }
  else
  {
    cout<<"Data size must be positive."<<endl;
    exit(0);
  }
}

Data::~Data()
{
  if(size > 0)
  {
    delete[] time;
    delete[] flux;
    delete[] error;
  }
}

Data::Data(Data& data)
{
  if(size != data.size)
  {
    size = data.size;
    time = new double[size];
    flux = new double[size];
    error = new double[size];
    memcpy(time, data.time, size*sizeof(double));
    memcpy(flux, data.flux, size*sizeof(double));
    memcpy(error, data.error, size*sizeof(double));
  }
  else 
  {
    memcpy(time, data.time, size*sizeof(double));
    memcpy(flux, data.flux, size*sizeof(double));
    memcpy(error, data.error, size*sizeof(double));
  }
}

Data& Data::operator = (Data& data)
{
  if(size != data.size)
  {
    delete[] time;
    delete[] flux;
    delete[] error;
    size = data.size;
    time = new double[size];
    flux = new double[size];
    error = new double[size];
    memcpy(time, data.time, size*sizeof(double));
    memcpy(flux, data.flux, size*sizeof(double));
    memcpy(error, data.error, size*sizeof(double));
  }
  else 
  {
    memcpy(time, data.time, size*sizeof(double));
    memcpy(flux, data.flux, size*sizeof(double));
    memcpy(error, data.error, size*sizeof(double));
  }
  return *this;
}

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
/*==================================================================*/
/* class DataFFT */
DataFFT::DataFFT()
{
  nd = npad = nd_fft = nd_fft_cal = 0;
  data_fft = resp_fft = conv_fft = NULL;
  data_real = resp_real = conv_real = NULL;
}

DataFFT::DataFFT(unsigned int nd, double fft_dx, unsigned int npad)
      :nd(nd), npad(npad) 
{
  int i;

  nd_fft = nd + npad;
  nd_fft_cal = nd_fft/2 + 1;

  data_fft = (fftw_complex *) fftw_malloc((nd_fft_cal) * sizeof(fftw_complex));
  resp_fft = (fftw_complex *) fftw_malloc((nd_fft_cal) * sizeof(fftw_complex));
  conv_fft = (fftw_complex *) fftw_malloc((nd_fft_cal) * sizeof(fftw_complex));

  data_real = new double[nd_fft];
  resp_real = new double[nd_fft];
  conv_real = new double[nd_fft];
      
  pdata = fftw_plan_dft_r2c_1d(nd_fft, data_real, data_fft, FFTW_PATIENT);
  presp = fftw_plan_dft_r2c_1d(nd_fft, resp_real, resp_fft, FFTW_PATIENT);
  pback = fftw_plan_dft_c2r_1d(nd_fft, conv_fft, conv_real, FFTW_PATIENT);
      
  /* normalization */
  fft_norm = fft_dx/nd_fft;

  for(i=0; i < nd_fft; i++)
  {
    data_real[i] = resp_real[i] = conv_real[i] = 0.0;
  }
}

DataFFT::DataFFT(Data& cont, unsigned int npad)
      :npad(npad)
{
  int i;

  nd = cont.size;
  nd_fft = nd + npad;
  nd_fft_cal = nd_fft/2 + 1;

  data_fft = (fftw_complex *) fftw_malloc((nd_fft_cal) * sizeof(fftw_complex));
  resp_fft = (fftw_complex *) fftw_malloc((nd_fft_cal) * sizeof(fftw_complex));
  conv_fft = (fftw_complex *) fftw_malloc((nd_fft_cal) * sizeof(fftw_complex));

  data_real = new double[nd_fft];
  resp_real = new double[nd_fft];
  conv_real = new double[nd_fft];
      
  pdata = fftw_plan_dft_r2c_1d(nd_fft, data_real, data_fft, FFTW_PATIENT);
  presp = fftw_plan_dft_r2c_1d(nd_fft, resp_real, resp_fft, FFTW_PATIENT);
  pback = fftw_plan_dft_c2r_1d(nd_fft, conv_fft, conv_real, FFTW_PATIENT);
      
  fft_norm = (cont.time[1] - cont.time[0]) / nd_fft;

  for(i=0; i < nd_fft; i++)
  {
    data_real[i] = resp_real[i] = conv_real[i] = 0.0;
  }
}

DataFFT& DataFFT::operator = (DataFFT& df)
{
  if(nd != df.nd)
  {
    fftw_free(data_fft);
    fftw_free(resp_fft);
    fftw_free(conv_fft);

    delete[] data_real;
    delete[] resp_real;
    delete[] conv_real;

    fftw_destroy_plan(pdata);
    fftw_destroy_plan(presp);
    fftw_destroy_plan(pback);
        
    int i;
    nd = df.nd;
    npad = df.npad;
    nd_fft = nd + npad;
    nd_fft_cal = nd_fft/2 + 1;
  
    data_fft = (fftw_complex *) fftw_malloc((nd_fft_cal) * sizeof(fftw_complex));
    resp_fft = (fftw_complex *) fftw_malloc((nd_fft_cal) * sizeof(fftw_complex));
    conv_fft = (fftw_complex *) fftw_malloc((nd_fft_cal) * sizeof(fftw_complex));
  
    data_real = new double[nd_fft];
    resp_real = new double[nd_fft];
    conv_real = new double[nd_fft];
        
    pdata = fftw_plan_dft_r2c_1d(nd_fft, data_real, data_fft, FFTW_PATIENT);
    presp = fftw_plan_dft_r2c_1d(nd_fft, resp_real, resp_fft, FFTW_PATIENT);
    pback = fftw_plan_dft_c2r_1d(nd_fft, conv_fft, conv_real, FFTW_PATIENT);
        
    fft_norm = df.fft_norm;
  
    for(i=0; i < nd_fft; i++)
    {
      data_real[i] = resp_real[i] = conv_real[i] = 0.0;
    }
  }
  return *this;
}

DataFFT::~DataFFT()
{
  if(nd > 0)
  {
    fftw_free(data_fft);
    fftw_free(resp_fft);
    fftw_free(conv_fft);

    delete[] data_real;
    delete[] resp_real;
    delete[] conv_real;

    fftw_destroy_plan(pdata);
    fftw_destroy_plan(presp);
    fftw_destroy_plan(pback);
  }
}

/* convolution with resp, output to conv */
void DataFFT::convolve_simple(double *conv)
{
  int i;
  for(i=0; i<nd_fft_cal; i++)
  {
    conv_fft[i][0] = data_fft[i][0]*resp_fft[i][0] - data_fft[i][1]*resp_fft[i][1];
    conv_fft[i][1] = data_fft[i][0]*resp_fft[i][1] + data_fft[i][1]*resp_fft[i][0];
  }
  fftw_execute(pback);

  /* normalize */
  for(i=0; i<nd_fft; i++)
  {
    conv_real[i] *= fft_norm;
  }

  /* copy back, the first npad points are discarded */
   memcpy(conv, conv_real, nd*sizeof(double));      
   return;
}
/*==================================================================*/
/* class RMFFT */
RMFFT::RMFFT(unsigned int n, double *cont, double norm)
      :DataFFT(n, norm)
{
  /* fft of cont setup only once */
  memcpy(data_real+npad, cont, nd*sizeof(double));
  fftw_execute(pdata);
}
    
RMFFT::RMFFT(Data& cont):DataFFT(cont)
{
  memcpy(data_real, cont.flux, nd*sizeof(double));
  fftw_execute(pdata);
}

/* convolution with resp, output to conv */
void RMFFT::convolve(const double *resp, unsigned int n, double *conv)
{
  /* fft of resp */
  memcpy(resp_real, resp, n * sizeof(double));
  fftw_execute(presp);
  
  DataFFT::convolve_simple(conv);
  return;
}
/*==================================================================*/
/* class PixonFFT */
PixonFFT::PixonFFT()
{
  npixon = ipixon_min = 0;
  pixon_sizes = NULL;
}
PixonFFT::PixonFFT(unsigned int npixel, unsigned int npixon)
      :npixon(npixon), DataFFT(npixel, 1.0, npixon*3)
{
  unsigned int i;

  ipixon_min = npixon-1;
  pixon_sizes = new double[npixon];
  for(i=0; i<npixon; i++)
  {
    pixon_sizes[i] = i+1;
  }
}

PixonFFT::~PixonFFT()
{
  if(npixon > 0)
  {
    delete[] pixon_sizes;
  }
}

void PixonFFT::convolve(const double *pseudo_img, unsigned int *pixon_map, double *conv)
{
  int ip, j;
  double psize;
  double *conv_tmp = new double[nd];

  /* fft of pseudo image */
  memcpy(data_real, pseudo_img, nd*sizeof(double));
  fftw_execute(pdata);

  /* loop over all pixon sizes */
  for(ip=ipixon_min; ip<npixon; ip++)
  {
    psize = pixon_sizes[ip];
    /* setup resp */
    for(j=0; j<nd_fft/2; j++)
    {
      resp_real[j] = 1.0/sqrt(2.0*M_PI)/psize * exp(-0.5 * (j)*(j)/psize/psize);
    }
    for(j=nd_fft-1; j>=nd_fft/2; j--)
    {
      resp_real[j] = 1.0/sqrt(2.0*M_PI)/psize * exp(-0.5 * (nd_fft-j)*(nd_fft-j)/psize/psize);
    }
    fftw_execute(presp);
    
    DataFFT::convolve_simple(conv_tmp);
    for(j=0; j<nd; j++)
    {
      if(pixon_map[j] == ip)
        conv[j] = conv_tmp[j];
    }
  }

  delete[] conv_tmp;
}

/* reduce the minimum pixon size */
void PixonFFT::reduce_pixon_min()
{
  if(ipixon_min > 0)
  {
    ipixon_min--;
  }
}
/*==================================================================*/
/* class Pixon */

Pixon::Pixon()
{
  npixel = 0;
  pixon_map = NULL; 
  image = pseudo_image = NULL;
  rmline = NULL;
}

Pixon::Pixon(Data& cont, Data& line, unsigned int npixel,  unsigned int npixon)
  :cont(cont), line(line), npixel(npixel), rmfft(cont), pfft(npixel, npixon)
{
  pixon_map = new unsigned int[npixel];
  image = new double[npixel];
  pseudo_image = new double[npixel];
  rmline = new double[cont.size];
  itline = new double[line.size];
  residual = new double[line.size];

  dt = cont.time[1]-cont.time[0];
  unsigned int i;
  for(i=0; i<npixel; i++)
  {
    pixon_map[i] = npixon-1;
  }
}

Pixon::~Pixon()
{
  if(npixel > 0)
  {
    delete[] pixon_map;
    delete[] image;
    delete[] pseudo_image;
    delete[] rmline;
    delete[] itline;
    delete[] residual;
  }
}

double Pixon::interp(double t)
{
  int it;
  it = (t - cont.time[0])/dt;
  if(it <=0 || it >= cont.size)
    return 0.0;
  return rmline[it] + (rmline[it+1] - rmline[it])/(cont.time[it+1] - cont.time[it]) * (t - cont.time[it]);
}
    
void Pixon::compute_rm_pixon(const vector<double> &x)
{
  int i;
  double t, it;
  /* convolve with pixons */
  for(i=0; i<npixel; i++)
  {
    pseudo_image[i] = exp(x[i]);
  }
  pfft.convolve(pseudo_image, pixon_map, image);
  /* reverberation mapping */
  rmfft.convolve(image, npixel, rmline);

  /* interpolation */
  for(i=0; i<line.size; i++)
  {
    t = line.time[i];
    itline[i] = interp(t);
    residual[i] = itline[i] - line.flux[i];
  }
}

double Pixon::chisquare(const vector<double> &x)
{
  double chisq;
  int i;

  /* calculate chi square */
  chisq = 0.0;
  for(i=0; i<line.size; i++)
  {
    chisq += (residual[i] * residual[i])/(line.error[i] * line.error[i]);
  }
  //printf("%f\n", chisq);
  return chisq;
}

void Pixon::chisquare_grad(const vector<double> &x, vector<double> &grad)
{
  int i, k, j, joffset, jrange1, jrange2;
  double psize, t, tau, cont_intp, grad_in, grad_out, K;
  for(i=0; i<npixel; i++)
  {
    grad_out = 0.0;
    psize = pixon_map[i] + 1;
    joffset = 5 * psize;
    jrange1 = fmax(i - joffset, 0.0);
    jrange2 = fmin(i + joffset, npixel);
    
    for(k=0; k<line.size; k++)
    {          
      grad_in = 0.0;
      t = line.time[k];
      for(j=jrange1; j<jrange2; j++)
      {
        tau = j * dt;
        cont_intp = interp(t-tau);
        K = 1.0/sqrt(2.0*M_PI)/psize * exp( - 0.5*(j-i)*(j-i)/(psize*psize) );
        grad_in += K * cont_intp;
      }
      grad_out += grad_in * residual[k]/line.error[k]/line.error[k];
    }
    grad[i] = pseudo_image[i] * grad_out * 2.0*dt;
  }
}

double Pixon::compute_pixon_number()
{
  int i;
  double num, psize;
    
  num = 0.0;
  for(i=0; i<pfft.nd; i++)
  {
    psize = pfft.pixon_sizes[pixon_map[i]];
    num += 1.0/(sqrt(2.0*M_PI) * psize);
  }
  return num;
}

void Pixon::update_pixon_map()
{
  int i;
  pfft.reduce_pixon_min();
  for(i=0; i<npixel; i++)
  {
    pixon_map[i]--;
  }
}
/*==================================================================*/

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