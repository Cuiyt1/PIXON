
/*
 *  PIXON
 *  A Pixon-based method for reconstructing velocity-delay map in reverberation mapping.
 * 
 *  Yan-Rong Li, liyanrong@mail.ihep.ac.cn
 * 
 */
#include <iostream>    %包含输入输出流的标准头文件。
#include <vector>      %包含向量（Vector）容器的标准头文件。
#include <iomanip>     %包含控制输出格式的标准头文件。
#include <cmath>       %包含数学函数的标准头文件。
#include <cstring>     %包含字符串操作函数的标准头文件。
#include <random>      %包含生成随机数的标准头文件。
#include <nlopt.hpp>   %包含用于非线性优化的NLopt库的头文件。
#include <fftw3.h>     %包含用于高性能FFT（快速傅里叶变换）的FFTW库的头文件。
% 包含一些自定义头文件，这些头文件可能包含了一些函数和数据结构的声明。
#include "proto.hpp"
#include "utilities.hpp"
#include "cont_model.hpp"
#include "pixon_cont.hpp"
#include "drw_cont.hpp"
#include "tnc.h"

using namespace std;

int run(Config &cfg)
{
  Data cont, line;                                %定义了两个名为cont和line的Data类型对象。这些对象可能是用于存储光谱数据的数据结构。
  cont.load(cfg.fcont);  /* load cont data */     %调用cont对象的load方法，从指定的文件cfg.fcont中加载光谱数据。
  line.load(cfg.fline);  /* load line data */     %调用line对象的load方法，从指定的文件cfg.fline中加载另一组光谱数据。

  /* continuum reconstruction */                  
  cout<<"Start cont reconstruction."<<endl;       %打印输出提示信息，开始进行连续谱的重构。
  %时间相关计算：
  /* time extending of reconstruction, 1% of time span */            /*重建的时间长，时间跨度的1% */ 
  double text_rec = 0.1 * fmax((cont.time[cont.size-1] - cont.time[0]), (line.time[line.size-1]-line.time[0]));         %计算重构时间的扩展，为时间跨度的1%。
  /* time backward */                                                                                             
  double tback = fmax(cont.time[0] - (line.time[0] - cfg.tau_range_up - text_rec), text_rec);                           %计算时间的后退值。
  /* time forward */
  double tforward = fmax((line.time[line.size-1] - cfg.tau_range_low + text_rec) - cont.time[cont.size-1], text_rec);   %计算时间的前进值。
  double sigmad, taud, syserr;                         %定义三个变量，分别用于存储标准差、时间延迟以及系统误差。
  %连续谱重构：
  /* use drw to reconstruct continuum */               /*使用DRW来重建连续体*/
  cont_model = new ContModel(cont, tback, tforward, cfg.tau_interval);     %创建一个ContModel对象，使用给定的参数进行初始化。
  cont_model->mcmc();                                                      %调用mcmc方法执行马尔可夫链蒙特卡洛（MCMC）算法。
  cont_model->get_best_params();                                           %获取MCMC算法得到的最佳参数。
  cont_model->recon();                                                     %执行连续谱的重构。
  taud = exp(cont_model->best_params[2]);                                  %获取时间延迟的指数形式。
  sigmad = exp(cont_model->best_params[1])*sqrt(taud);                     %计算标准差。
  syserr = (exp(cont_model->best_params[0]) - 1.0) * cont_model->mean_error;  %计算系统误差。
  %一些额外的变量定义：
  int npixel;  /* number of pixels */像素数量。
  int npixon_size, npixon_size0; 
  int ipositive_tau; /* index of zero lag */零时滞的索引
  double *pimg;                     %定义用于存储图像的数组。

  pixon_sub_factor = cfg.pixon_sub_factor;
  pixon_size_factor = cfg.pixon_size_factor;
  pixon_map_low_bound = cfg.pixon_map_low_bound;
  npixon_size0 = cfg.max_pixon_size*pixon_sub_factor/pixon_size_factor;

  /* number of pixels */  计算变量 npixel 的值，它表示像素的数量。计算是通过时间范围除以时间间隔实现的。
  npixel = (cfg.tau_range_up - cfg.tau_range_low) / (cont_model->cont_recon.time[1]-cont_model->cont_recon.time[0]);

  /* index at which positive lags starts */   表示正时间滞开始的索引。计算是通过零减去 cfg.tau_range_low 除以时间间隔实现的。
  ipositive_tau = (0.0 - cfg.tau_range_low) / (cont_model->cont_recon.time[1]-cont_model->cont_recon.time[0]);

  /* used to restore image  */  动态分配一个 double 数组，其大小为 npixel+1+cont_model->cont_recon.size+1，并将其地址赋给指针 pimg。这个数组可能用于存储恢复后的图像。
  pimg = new double[npixel+1+cont_model->cont_recon.size+1];

  /* setup pixon type */
  switch(cfg.pixon_basis_type)                   %根据cfg.pixon_basis_type的值，选择相应的Pixon类型。
  {
    case 0:  /* parabloid */     抛物面 
      pixon_function = PixonBasis::parabloid;
      pixon_norm = PixonBasis::parabloid_norm;
      break;

    case 1:  /* Gaussian */      高斯函数
      PixonBasis::norm_gaussian = sqrt(2.0*M_PI) * erf(3.0*pixon_size_factor/sqrt(2.0));  %计算高斯函数的归一化系数

      pixon_function = PixonBasis::gaussian;
      pixon_norm = PixonBasis::gaussian_norm;
      break;
    
    case 2: /* modified Gaussian */  修改后的高斯函数
      PixonBasis::coeff1_modified_gaussian = exp(-0.5 * pixon_size_factor*3.0*pixon_size_factor*3.0);  %计算修改后高斯函数的系数
      PixonBasis::coeff2_modified_gaussian = 1.0 - PixonBasis::coeff1_modified_gaussian;
      PixonBasis::norm_gaussian = (sqrt(2.0*M_PI) * erf(3.0*pixon_size_factor/sqrt(2.0)) 
                    - 2.0*3.0*pixon_size_factor * PixonBasis::coeff1_modified_gaussian)/PixonBasis::coeff2_modified_gaussian;  %计算高斯函数的归一化系数
      
      pixon_function = PixonBasis::modified_gaussian;
      pixon_norm = PixonBasis::modified_gaussian_norm;
      break;
    
    case 3:  /* Lorentz */    洛伦兹函数
      pixon_function = PixonBasis::lorentz;
      pixon_norm = PixonBasis::lorentz_norm;
      break;
    
    case 4: /* Wendland */
      pixon_function = PixonBasis::wendland;
      pixon_norm = PixonBasis::wendland_norm;
      break;
    
    case 5:  /* triangle */ 
      pixon_function = PixonBasis::triangle;
      pixon_norm = PixonBasis::triangle_norm;
      break;
    
    case 6:  /* top-hat */     顶帽函数
      pixon_sub_factor = 1; /* enforce to 1 */
      pixon_function = PixonBasis::tophat;
      pixon_norm = PixonBasis::tophat_norm;
      break;
    
    default:  /* default */    默认情况
      PixonBasis::norm_gaussian = sqrt(2.0*M_PI) * erf(3.0*pixon_size_factor/sqrt(2.0));  %计算高斯函数的归一化系数

      pixon_function = PixonBasis::gaussian;
      pixon_norm = PixonBasis::gaussian_norm;
      break;
  }
  
  if(cfg.drv_lc_model == 0 || cfg.drv_lc_model == 3)     %如果 cfg.drv_lc_model 的值等于 0 或 3，则执行以下代码块。这个条件语句检查驱动（drv_lc_model）的模型类型。
  {
    /* continuum free with pixon, line with pixon 
     * resp_pixon_uniform.txt, resp_pixon.txt
     * line_pixon_uniform.txt, line_pixon.txt
     * cont_pixon_uniform.txt, cont_pixon.txt
     */
    if(cfg.pixon_uniform)       %如果 cfg.pixon_uniform 为真（非零）
    {
      npixon_size = npixon_size0;    %将变量 npixon_size 的值设置为另一个变量 npixon_size0 的值。这通常用于初始化或重置 npixon_size。
      run_pixon_uniform(cont, cont_model->cont_recon, line, pimg, npixel, npixon_size, ipositive_tau, cfg);
    }
    else 
    {
      npixon_size = npixon_size0;
      run_pixon(cont, cont_model->cont_recon, line, pimg, npixel, npixon_size, ipositive_tau, cfg);  %调用了名为 run_pixon_uniform 的函数，传递了一系列参数给该函数。
    }                                                                                                %这段代码的目的是调用 run_pixon_uniform 函数来处理特定类型的数据，其中使用了一些参数以及先前设置的 npixon_size 值。
  }
  if(cfg.drv_lc_model == 1 || cfg.drv_lc_model == 3)
  {
    /* continuum free with drw, line with pixon 
     * resp_drw_uniform.txt, resp_drw.txt
     * line_drw_uniform.txt, line_drw.txt
     * cont_drw_uniform.txt, cont_drw.txt
     */
    if(cfg.pixon_uniform)
    {
      npixon_size = npixon_size0;
      run_drw_uniform(cont, cont_model->cont_recon, line, pimg, npixel, npixon_size, ipositive_tau, sigmad, taud, syserr, cfg);
    }
    else 
    {
      npixon_size = npixon_size0;
      run_drw(cont, cont_model->cont_recon, line, pimg, npixel, npixon_size, ipositive_tau, sigmad, taud, syserr, cfg);
    }
  }
  if(cfg.drv_lc_model == 2 || cfg.drv_lc_model == 3)
  {
    /* continuum fixed with drw, line with pixon 
     * resp_contfix_uniform.txt, resp_contfix.txt
     * line_contfix_uniform.txt line_contfix.txt 
     */
    if(cfg.pixon_uniform)
    {
      npixon_size = npixon_size0;
      run_contfix_uniform(cont_model->cont_recon, line, pimg, npixel, npixon_size, ipositive_tau, cfg);
    }
    else 
    {
      npixon_size = npixon_size0;
      run_contfix(cont_model->cont_recon, line, pimg, npixel, npixon_size, ipositive_tau, cfg);
    }
  }

  delete[] pimg;
  return 0;
}

/*
 * continuum free with drw, line with pixon
 *
 */
void run_drw(Data& cont_data, Data& cont_recon, Data& line, double *pimg, int npixel, 
                    int& npixon_size, int ipositive_tau, double sigmad, double taud, double syserr, Config& cfg)
{
  cout<<"************************************************************"<<endl;
  cout<<"Start run_drw..."<<endl;
  cout<<"npixon_size:"<<npixon_size<<endl;
  int i, iter;
  bool flag;
  %这段代码主要是一个包含优化过程的算法实现。
  PixonDRW pixon(cont_data, cont_recon, line, npixel, npixon_size, sigmad, taud, syserr, ipositive_tau, cfg.sensitivity);%通过构造函数初始化一个名为 pixon 的 PixonDRW 对象，该对象包含了多个参数
  void *args = (void *)&pixon;
  double f, f_old, num, num_old, chisq, chisq_old, df, dnum;

  int ndim = npixel + 1 + cont_recon.size + 1;  /* include one parameter for background */  %计算了优化过程中参数的总数量，包括像素数、背景参数、以及连续重建参数等。
  /* TNC */TNC 方法的设置设置了 TNC 优化方法的一些参数，如最大迭代次数、最大函数调用次数、精度等。
  int rc, maxCGit = ndim, maxnfeval = cfg.nfeval_max, nfeval, niter;
  double eta = -1.0, stepmx = -1.0, accuracy =  cfg.tol, fmin = pixon.line.size, 
    ftol = cfg.tol, xtol = cfg.tol, pgtol = cfg.tol, rescale = -1.0;
  
  /* NLopt */使用 NLopt 库初始化了一个优化器 opt0，选择了 BOBYQA 算法。同时初始化了一些用于存储参数和梯度的向量。
  nlopt::opt opt0(nlopt::LN_BOBYQA, ndim);
  vector<double> x(ndim), g(ndim), x_old(ndim);
  vector<double>low(ndim), up(ndim);

  /* bounds and initial values */设置了参数的边界和初始值。对于传输函数的参数，设置了上下界和初始值；对于背景参数，根据 cfg.fix_bg 的值设置为固定值或可变范围；对于连续重建参数，设置了上下界和初始值。
  /* transfer function */
  for(i=0; i<npixel; i++)
  {
    low[i] = -100.0;
    up[i] =   10.0;
    x[i] = log(1.0/(npixel * pixon.dt));
  }
  /* background */
  if(cfg.fix_bg)
  {
    low[npixel] = up[npixel] = x[npixel] = cfg.bg;
  }
  else 
  {
    low[npixel] = -1.0;
    up[npixel] = 1.0;
    x[npixel] = 0.0;
  }
  /* continuum reconstruction */
  for(i=npixel+1; i<ndim; i++)
  {
    low[i] = -10.0;
    up[i] =   10.0;
    x[i] = 0.0;
  }
  
  /* initial optimization *//*初始优化*/
  %使用了两种优化算法，一种是基于NLOPT库的算法，另一种是基于TNC库的算法。
  opt0.set_min_objective(func_nlopt_cont_drw, args); %设置了NLOPT库中的优化器（opt0）的目标函数。函数 func_nlopt_cont_drw 被用作优化目标，而 args 是传递给目标函数的参数。
  opt0.set_lower_bounds(low);                        %设置了NLOPT库中的优化器的变量下界，即优化变量的最小值。low 是包含每个变量下界的向量。
  opt0.set_upper_bounds(up);                         %设置了NLOPT库中的优化器的变量上界，即优化变量的最大值。up 是包含每个变量上界的向量。
  opt0.set_maxeval(1000);                            %设置了NLOPT库中的优化器的最大评估次数，即允许算法进行的最大函数调用次数。
  opt0.set_ftol_abs(cfg.tol);                        %设置了NLOPT库中的优化器的绝对函数值容差。
  opt0.set_xtol_abs(cfg.tol);                        %设置了NLOPT库中的优化器的绝对变量值容差。
  opt0.optimize(x, f);                               %使用NLOPT库中的优化器进行一次优化，将结果存储在向量 x 中，最终函数值存储在 f 中。

  rc = tnc(ndim, x.data(), &f, g.data(), func_tnc_cont_drw, args, low.data(), up.data(), 
      NULL, NULL, TNC_MSG_INFO|TNC_MSG_EXIT,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);               %使用TNC库中的优化器进行一次优化。这里使用了 TNC 算法，传递了一些参数，如目标函数 func_tnc_cont_drw，梯度信息 g，变量的下界和上界等
  %保存当前迭代中一些重要的状态信息，并输出这些信息。
  f_old = f;
  num_old = pixon.compute_pixon_number();
  pixon.compute_rm_pixon(x.data());
  chisq_old = pixon.compute_chisquare(x.data());
  memcpy(x_old.data(), x.data(), ndim*sizeof(double));
  cout<<f_old<<"  "<<num_old<<"  "<<chisq_old<<endl;

  /* then pixel-dependent pixon size */迭代循环，其中包含了对像素相关的"Pixon"大小的调整。
  iter = 0;                                      %初始化迭代计数器为0。
  do                                             %do-while循环
  {
    iter++;                                      %每次循环迭代，迭代计数器递增。
    cout<<"===================iter:"<<iter<<"==================="<<endl;%输出当前迭代的信息，用于调试或记录。

    flag = pixon.update_pixon_map();             %调用 pixon 对象的 update_pixon_map 方法，可能是更新了"Pixon"相关的地图，返回值存储在 flag 中。
    if(!flag)
      break;                                     %如果更新失败，即 flag 为false，则跳出循环。

    num = pixon.compute_pixon_number();          %计算并存储当前"Pixon"的数量。
    
    for(i=0; i<ndim; i++)                        %遍历优化变量的每一个维度，确保其值在设定的下界 low 和上界 up 之间。
    {
      if(x[i] < low[i])
        x[i] = low[i];
      else if(x[i] > up[i])
        x[i] = up[i];
    }

    opt0.optimize(x, f);                       %使用NLOPT库中的优化器进行一次优化，将结果存储在向量 x 中，最终函数值存储在 f 中。
    rc = tnc(ndim, x.data(), &f, g.data(), func_tnc_cont_drw, args, low.data(), up.data(), 
      NULL, NULL, TNC_MSG_INFO|TNC_MSG_EXIT,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);         %使用TNC库中的优化器进行一次优化，类似于上述的NLOPT的操作。
    
    pixon.compute_rm_pixon(x.data());          %根据优化结果，计算并更新"Pixon"的一些相关数据。
    chisq = pixon.compute_chisquare(x.data()); %计算并存储当前优化结果的卡方值。
    cout<<f<<"  "<<num<<"  "<<chisq<<endl;     %输出当前迭代的函数值、Pixon数量和卡方值。

    if(f <= pixon.line.size)                   %如果函数值小于等于某个条件（pixon.line.size），则执行一些操作后跳出循环。
    {
      memcpy(x_old.data(), x.data(), ndim*sizeof(double));  %复制当前优化变量 x 的值到 x_old 中，用于下一次迭代比较。
      break;
    }

    num_old = num;
    f_old = f;
    chisq_old = chisq;
    memcpy(x_old.data(), x.data(), ndim*sizeof(double));
  }while(pixon.pfft.get_ipxion_min() >= pixon_map_low_bound); 

  cout<<"bg: "<<x_old[npixel]<<endl;        %输出数组 x_old 中索引为 npixel 的元素的值，并附加 "bg: " 字符串。这是在控制台打印调试信息。
  pixon.compute_rm_pixon(x_old.data());     %根据数组 x_old 中的数据，调用 pixon 对象的 compute_rm_pixon 方法，可能是计算并更新与"Pixon"相关的一些数据。
    %文件输出部分：
    %通过文件输出，将一些计算结果写入文件中。这里使用了 ofstream 对象和循环来遍历数据并写入文件。
    %resp_drw.txt 文件：保存了一些与响应函数相关的数据，包括时间、图像值以及 exp(x_old[i])。
    %line_drw.txt 文件：保存了与线性数据相关的信息，包括时间、pixon.itline[i]*pixon.line.norm 和 pixon.itline[i] - line.flux[i]。
    %line_drw_full.txt 文件：包含了一些与线性数据相关的完整信息，包括时间和 pixon.rmline[i]*pixon.line.norm。
    %cont_drw.txt 文件：保存了与连续数据相关的信息，包括时间、pixon.cont.flux[i]*pixon.cont.norm 以及 pixon.cont.error[i]*pixon.cont.norm。
    %pixon_map_drw.txt 文件：包含了一些与Pixon映射相关的信息，包括时间和 pixon.pfft.pixon_sizes[pixon.pixon_map[i]]。
  ofstream fout;
  string fname;
  fname = "data/resp_drw.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<npixel; i++)
  {
    fout<<pixon.dt*(i-ipositive_tau)<<"  "<<pixon.image[i]<<"  "<<exp(x_old[i])<<endl;
  }
  fout.close();

  fname = "data/line_drw.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<pixon.line.size; i++)
  {
    fout<<pixon.line.time[i]<<"  "<<pixon.itline[i]*pixon.line.norm<<"   "<<pixon.itline[i] - line.flux[i]<<endl;
  }
  fout.close();

  fname = "data/line_drw_full.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<pixon.cont.size; i++)
  {
    fout<<pixon.cont.time[i]<<"  "<<pixon.rmline[i]*pixon.line.norm<<endl;
  }
  fout.close();
  
  fname = "data/cont_drw.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<pixon.cont.size; i++)
  {
    fout<<pixon.cont.time[i]<<" "<<pixon.cont.flux[i]*pixon.cont.norm<<"  "<<pixon.cont.error[i]*pixon.cont.norm<<endl;
  }
  fout.close();
  
  fname = "data/pixon_map_drw.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<npixel; i++)
  {
    fout<<(i-ipositive_tau)*pixon.dt<<"  "<<pixon.pfft.pixon_sizes[pixon.pixon_map[i]]<<endl;
  }
  fout.close();
  memcpy(pimg, x_old.data(), ndim*sizeof(double));   %使用 memcpy 函数将数组 x_old 中的数据复制到 pimg 数组中。这可能是为了保存当前迭代的状态信息。
}

void run_drw_uniform(Data& cont_data, Data& cont_recon, Data& line, double *pimg, int npixel, 
                  int& npixon_size, int ipositive_tau, double sigmad, double taud, double syserr, Config& cfg)
{
  %打印信息：
  cout<<"************************************************************"<<endl;
  cout<<"Start run_drw_uniform..."<<endl;
  cout<<"npixon_size:"<<npixon_size<<endl;
  int i, iter;
  bool flag;
  PixonDRW pixon(cont_data, cont_recon, line, npixel, npixon_size, sigmad, taud, syserr, ipositive_tau, cfg.sensitivity);  %创建 PixonDRW 类的对象 pixon，并传递一些数据和参数。
  void *args = (void *)&pixon;  %声明一个指向 pixon 对象的指针。
  double f, f_old, num, num_old, chisq, chisq_old, df, dnum;

  int ndim = npixel + 1 + cont_recon.size + 1;  /* include one parameter for background */  %算变量 ndim，它表示问题的维度
  /* TNC */
  int rc, maxCGit = ndim, maxnfeval = cfg.nfeval_max, nfeval, niter;
  double eta = -1.0, stepmx = -1.0, accuracy =  cfg.tol, fmin = pixon.line.size, 
    ftol = cfg.tol, xtol = cfg.tol, pgtol = cfg.tol, rescale = -1.0;    %设定 TNC 方法的一些参数，这些参数在后续调用 TNC 优化算法时会用到。
  
  /* NLopt */
  nlopt::opt opt0(nlopt::LN_BOBYQA, ndim);     %使用 NLopt 库创建一个优化器对象 opt0，选择了 BOBYQA 算法，问题的维度为 ndim。
  vector<double> x(ndim), g(ndim), x_old(ndim);   %创建用于存储优化参数的向量：
  vector<double>low(ndim), up(ndim);              %创建用于设置上下界的向量：

  /* bounds and initial values *//*边界和初始值*/ /*传递函数*/
  /* transfer function */   
  %设置变量的上下界和初始值：
  %针对转移函数，设置 npixel 个像素的上下界和初始值。
  %针对背景参数，如果 cfg.fix_bg 为真，则上下界和初始值都设为 cfg.bg；否则，设置上下界为 -1.0 到 1.0，初始值为 0.0。
  %针对连续重建参数，设置上下界为 -10.0 到 10.0，初始值为 0.0。
  for(i=0; i<npixel; i++)
  {
    low[i] = -100.0;
    up[i] =   10.0;
    x[i] = log(1.0/(npixel * pixon.dt));
  }
  /* background */
  if(cfg.fix_bg)
  {
    low[npixel] = up[npixel] = x[npixel] = cfg.bg;
  }
  else 
  {
    low[npixel] = -1.0;
    up[npixel] = 1.0;
    x[npixel] = 0.0;
  }
  /* continuum reconstruction */
  for(i=npixel+1; i<ndim; i++)
  {
    low[i] = -10.0;
    up[i] =   10.0;
    x[i] = 0.0;
  }
%设置 NLopt 优化器的目标函数和约束：
  opt0.set_min_objective(func_nlopt_cont_drw, args);  %设置优化目标为 func_nlopt_cont_drw，并传递 args 指针作为额外参数。
  opt0.set_lower_bounds(low);                         %分别设置下界和上界。
  opt0.set_upper_bounds(up);
  opt0.set_maxeval(1000);                             %设置最大评估次数为 1000。
  opt0.set_ftol_abs(cfg.tol);                         %设置函数值和参数变化的容差
  opt0.set_xtol_abs(cfg.tol);
  %使用了TNC（Truncated Newton Conjugate-Gradient）优化算法进行最小化的过程
  opt0.optimize(x, f);            %这是一次对 x 进行优化的操作，使用了 opt0 对象，优化结果保存在 x 和 f 中。
  rc = tnc(ndim, x.data(), &f, g.data(), func_tnc_cont_drw, args, low.data(), up.data(), 
      NULL, NULL, TNC_MSG_INFO|TNC_MSG_EXIT,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);   %调用 TNC 优化算法的函数。它使用了一系列参数来配置优化过程，包括维度 ndim、初始点 x.data()、目标函数值 f、梯度 g.data()、约束等。优化的终止条件和输出信息也在这里设置。
  
  f_old = f;    %保存当前优化结果的目标函数值。
  num_old = pixon.compute_pixon_number();   %计算并保存当前的 pixon 数量。
  pixon.compute_rm_pixon(x.data());         %基于当前的参数 x 计算并更新 pixon。
  chisq_old = pixon.compute_chisquare(x.data());    %计算并保存当前的卡方值。
  memcpy(x_old.data(), x.data(), ndim*sizeof(double));   %将当前参数 x 复制到 x_old 中，用于后续比较。
  cout<<f_old<<"  "<<num_old<<"  "<<chisq_old<<endl;     %输出当前的目标函数值、pixon 数量和卡方值。

  iter = 0;
  while(npixon_size>pixon_map_low_bound+1)
  {
    iter++;
    cout<<"===================iter:"<<iter<<"==================="<<endl;
    npixon_size--;      %降低 npixon_size。
    cout<<"npixon_size:"<<npixon_size<<",  size: "<<pixon.pfft.pixon_sizes[npixon_size-1]<<endl;

    pixon.reduce_pixon_map_all();    %执行 pixon.reduce_pixon_map_all();：基于当前参数降低 pixon map。
    num = pixon.compute_pixon_number();   %重新计算 pixon 数量和目标函数值，并输出。

    for(i=0; i<ndim; i++)   %循环，对参数数组 x 进行边界约束。确保参数 x[i] 在指定的范围 [low[i], up[i]] 内。遍历参数数组 x 中的每个元素。
    {
      if(x[i] < low[i])
        x[i] = low[i];        %如果参数 x[i] 小于下界 low[i]，则将其设为下界。
      else if(x[i] > up[i])
        x[i] = up[i];         %如果参数 x[i] 大于上界 up[i]，则将其设为上界。
    } 

    opt0.optimize(x, f);    %使用某种优化算法（由 opt0 对象提供）对参数数组 x 进行优化，优化结果保存在 x 和 f 中。
    rc = tnc(ndim, x.data(), &f, g.data(), func_tnc_cont_drw, args, low.data(), up.data(), 
      NULL, NULL, TNC_MSG_INFO|TNC_MSG_EXIT,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);   %调用 TNC 优化算法，对 x 进行优化，更新参数 x 和目标函数值 f。函数的参数设置了优化过程中的一些配置，如最大迭代次数、收敛条件等。
    
    pixon.compute_rm_pixon(x.data());
    chisq = pixon.compute_chisquare(x.data());
    cout<<f<<"  "<<num<<"  "<<chisq<<endl;

    if(f <= fmin)   %如果目标函数值 f 小于等于预设的最小值 fmin，则复制当前参数到 x_old 并跳出循环。
    {
      memcpy(x_old.data(), x.data(), ndim*sizeof(double));
      break;
    }
    
    df = f-f_old;
    dnum = num - num_old;

    if(-df < dnum * (1.0 + cfg.sensitivity/sqrt(2.0*num)))   %判断是否回退到先前的 pixon 大小。如果满足条件，则调用 pixon.increase_pixon_map_all(); 并跳出循环。
    {
      /* pixon size goes back to previous value */
      pixon.increase_pixon_map_all();
      break;
    }

    num_old = num;
    f_old = f;
    chisq_old = chisq;
    memcpy(x_old.data(), x.data(), ndim*sizeof(double));
  }
%这段代码是一系列用于输出数据到文件的循环。
  cout<<"bg: "<<x_old[npixel]<<endl;
  pixon.compute_rm_pixon(x_old.data());
  ofstream fout;
  string fname;   %并将 x_old 对应的 pixon 重新计算并保存到文件。
  fname = "data/resp_drw_uniform.txt_" + to_string(cfg.pixon_basis_type);  %创建文件名并打开文件流。
  fout.open(fname);
  for(i=0; i<npixel; i++)    %这个循环遍历 npixel 个像素，将相关信息写入文件 fout。
  {
    fout<<pixon.dt*(i-ipositive_tau)<<"  "<<pixon.image[i]<<"  "<<exp(x_old[i])<<endl;
  }
  fout.close();    %关闭文件流。

  fname = "data/line_drw_uniform.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<pixon.line.size; i++)
  {
    fout<<pixon.line.time[i]<<"  "<<pixon.itline[i]*pixon.line.norm<<"   "<<pixon.itline[i] - line.flux[i]<<endl;
  }
  fout.close();

  fname = "data/line_drw_uniform_full.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<pixon.cont.size; i++)
  {
    fout<<pixon.cont.time[i]<<"  "<<pixon.rmline[i]*pixon.line.norm<<endl;
  }
  fout.close();

  fname = "data/cont_drw_uniform.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<pixon.cont.size; i++)
  {
    fout<<pixon.cont.time[i]<<" "<<pixon.cont.flux[i]*pixon.cont.norm<<"  "<<pixon.cont.error[i]*pixon.cont.norm<<endl;
  }
  fout.close();
  
  fname = "data/pixon_map_drw_uniform.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<npixel; i++)
  {
    fout<<(i-ipositive_tau)*pixon.dt<<"  "<<pixon.pfft.pixon_sizes[pixon.pixon_map[i]]<<endl;
  }
  fout.close();

  memcpy(pimg, x_old.data(), ndim*sizeof(double));  %将 x_old 的数据复制到 pimg 中。
}

/* set continuum free and use pixons to model continuum, pixel-dependent pixon sizes for RM *// /设置连续体自由，并使用像素来模拟连续体，像素依赖于RM的像素大小
void run_pixon(Data& cont_data, Data& cont_recon, Data& line, double *pimg, int npixel, 
                    int& npixon_size, int ipositive_tau, Config& cfg)      %这个函数接受一些数据结构和配置参数，其中 Data 和 Config 是自定义的类型，可能包含有关数据和配置的信息。函数的目的是运行一个名为 pixon 的对象
{
  cout<<"************************************************************"<<endl;
  cout<<"Start run_pixon..."<<endl;
  cout<<"npixon_size:"<<npixon_size<<endl;
  %这里声明了一些变量，例如布尔型变量 flag 和整数型变量 i、iter，还初始化了一个名为 pixon 的 PixonCont 类型的对象。
  bool flag;
  int i, iter;
  int npixon_size_cont = 10;
  PixonCont pixon(cont_data, cont_recon, line, npixel, npixon_size, npixon_size_cont, ipositive_tau, cfg.sensitivity);
  void *args = (void *)&pixon;
  double f, f_old, num, num_old, chisq, chisq_old, df, dnum;

  /* TNC */这里声明了一些与 TNC（Truncated Newton Conjugate-Gradient）优化算法相关的参数，用于后续的优化过程。
  int rc, maxCGit = cont_recon.size, maxnfeval = cfg.nfeval_max, nfeval, niter;
  double eta = -1.0, stepmx = -1.0, accuracy =  cfg.tol, fmin = pixon.cont_data.size, 
    ftol = cfg.tol, xtol = cfg.tol, pgtol = cfg.tol, rescale = -1.0;
  
  /* NLopt */这里使用 NLopt 库声明了一个优化器对象 opt0，以及一些与优化相关的向量和参数。
  nlopt::opt opt0(nlopt::LN_BOBYQA, cont_recon.size);
  vector<double> x_cont(cont_recon.size), g_cont(cont_recon.size), x_old_cont(cont_recon.size);
  vector<double> low_cont(cont_recon.size), up_cont(cont_recon.size);

  /* bounds and initial values */这里设置了优化问题的边界和初始值。
  for(i=0; i<cont_recon.size; i++)
  {
    low_cont[i] = fmax(0.0, cont_recon.flux[i] - 5.0 * cont_recon.error[i]);
    up_cont[i] =            cont_recon.flux[i] + 5.0 * cont_recon.error[i];
    x_cont[i] = cont_recon.flux[i];
  }
  %这里设置了 NLopt 优化器的一些属性，包括优化目标函数、边界、最大评估次数等。
  opt0.set_min_objective(func_nlopt_cont, args);
  opt0.set_lower_bounds(low_cont);
  opt0.set_upper_bounds(up_cont);
  opt0.set_maxeval(1000);
  opt0.set_ftol_abs(cfg.tol);
  opt0.set_xtol_abs(cfg.tol);
  %这里调用了 NLopt 的优化过程和 TNC 的优化过程，对一些参数进行了优化，最终结果存储在变量 x_cont 和 f 中。
  opt0.optimize(x_cont, f);
  rc = tnc(cont_recon.size, x_cont.data(), &f, g_cont.data(), func_tnc_cont, args, 
      low_cont.data(), up_cont.data(), NULL, NULL, TNC_MSG_INFO|TNC_MSG_EXIT,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);
  
  f_old = f;
  num_old = pixon.compute_pixon_number_cont();
  pixon.compute_cont(x_cont.data());
  chisq_old = pixon.compute_chisquare_cont(x_cont.data());
  memcpy(x_old_cont.data(), x_cont.data(), cont_recon.size*sizeof(double));
  cout<<f_old<<"  "<<num_old<<"  "<<chisq_old<<endl;
 
  while(npixon_size_cont>2)    %一个 while 循环，其中的代码将不断执行直到 npixon_size_cont 小于等于 2。
  {
    npixon_size_cont--;        %逐步减小 npixon_size_cont。
    cout<<"npixon_size_cont:"<<npixon_size_cont<<",  size: "<<pixon.pfft_cont.pixon_sizes[npixon_size_cont-1]<<endl;  %输出当前的 npixon_size_cont 和对应的大小。
    
    pixon.reduce_ipixon_cont();   %调用 pixon 对象的 reduce_ipixon_cont 方法。
    num = pixon.compute_pixon_number_cont();
    
    for(i=0; i<cont_recon.size; i++)   %对 x_cont 进行一些范围限制。
    {
      if(x_cont[i] < low_cont[i])
        x_cont[i] = low_cont[i];
      else if(x_cont[i] > up_cont[i])
        x_cont[i] = up_cont[i];
    }

    opt0.optimize(x_cont, f);   %使用某种优化方法（可能是 TNC）来更新 x_cont 和 f。
    rc = tnc(cont_recon.size, x_cont.data(), &f, g_cont.data(), func_tnc_cont, args, 
      low_cont.data(), up_cont.data(), NULL, NULL, TNC_MSG_INFO|TNC_MSG_EXIT,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);
    
    pixon.compute_cont(x_cont.data());
    chisq = pixon.compute_chisquare_cont(x_cont.data());
    cout<<f<<"  "<<num<<"  "<<chisq<<endl;

    if(f <= fmin)       %如果 f 小于等于 fmin，则将 x_cont 复制到 x_old_cont 并跳出循环。
    {
      memcpy(x_old_cont.data(), x_cont.data(), cont_recon.size*sizeof(double));
      break;
    }
    
    df = f-f_old;      %否则，计算一些差值（df 和 dnum），
    dnum = num - num_old;

    if(-df < dnum * (1.0 + 1.0/sqrt(2.0*num)))   %根据一定条件决定是否增加 ipixon。
    {
      pixon.increase_ipixon_cont();
      break;
    }

    num_old = num;
    f_old = f;
    chisq_old = chisq;
    memcpy(x_old_cont.data(), x_cont.data(), cont_recon.size*sizeof(double));  %将最终的 x_old_cont 传递给 pixon 对象进行计算，
  }
  
  pixon.compute_cont(x_old_cont.data());   %然后将结果写入文件 "data/cont_recon_pixon.txt"。
  ofstream fp;
  fp.open("data/cont_recon_pixon.txt");
  for(i=0; i<cont_recon.size; i++)
  {
    fp<<pixon.cont.time[i]<<" "<<pixon.image_cont[i]*pixon.cont.norm<<endl;
  }
  fp.close();

  cout<<"Start to RM"<<endl;
  %这段代码执行了一个复杂的参数优化过程，其中包括传递函数参数、背景参数和连续谱重建参数的优化。
  /* then continuum and line reverberation */
  pixon.cont.set_data(pixon.image_cont);   %将 pixon 对象中的 image_cont 数据设置为 cont 对象的数据。
  /* TNC */计算一些与优化相关的参数：
  int ndim = npixel + 1 + pixon.cont.size;  %ndim 表示总的优化参数数量，包括 npixel 个传递函数参数、一个背景参数和 pixon.cont.size 个连续谱重建参数。
  maxCGit = ndim;
  fmin = pixon.line.size + pixon.cont_data.size;

  /* NLopt */  %创建一个 NLopt 优化对象 opt1，采用 BOBYQA 算法（nlopt::LN_BOBYQA）进行优化，参数维度为 ndim。
  nlopt::opt opt1(nlopt::LN_BOBYQA, ndim);
  vector<double> x(ndim), g(ndim), x_old(ndim);  %初始化一些向量和数组，包括 x（优化参数向量）、g（梯度向量）、x_old（保存上一次迭代的优化参数）、low 和 up（参数的下限和上限）。
  vector<double> low(ndim), up(ndim);

  /* bounds and initial values */
  /* tranfer function *//*边界和初始值*/ /*传递函数*/
  for(i=0; i<npixel; i++)   %为传递函数参数设置初始值和边界：
  {
    low[i] = -100.0;
    up[i] =  10.0;
    x[i] = log(1.0/(npixel * pixon.dt));    %对于前 npixel 个参数，下限为 -100.0，上限为 10.0，初始值为通过对数转换计算得到。
  }
  /* background */
  if(cfg.fix_bg)   %背景参数的处理取决于 cfg.fix_bg 的值，如果为真，则上下限和初始值都设置为 cfg.bg；否则，上下限为 -1.0 和 1.0，初始值为 0.0。
  {
    low[npixel] = up[npixel] = x[npixel] = cfg.bg;
  }
  else 
  {
    low[npixel] = -1.0;
    up[npixel] = 1.0;
    x[npixel] = 0.0;
  }
  /* continuum reconstruction */    %为连续谱重建参数设置初始值和边界：
  for(i=0; i<pixon.cont.size; i++)  %对于每个参数，下限为 fmax(0.0, cont_recon.flux[i] - 5.0 * cont_recon.error[i])，上限为 cont_recon.flux[i] + 5.0 * cont_recon.error[i]，初始值为 pixon.cont.flux[i]。
  {
    low[i+npixel+1] = fmax(0.0, cont_recon.flux[i] - 5.0 * cont_recon.error[i]);
    up[i+npixel+1] =            cont_recon.flux[i] + 5.0 * cont_recon.error[i];
    x[i+npixel+1] = pixon.cont.flux[i];
  }
  
  opt1.set_min_objective(func_nlopt_cont_rm, args);   %设置 NLopt 优化对象的目标函数、参数边界和一些其他参数。
  opt1.set_lower_bounds(low);
  opt1.set_upper_bounds(up);
  opt1.set_maxeval(1000);
  opt1.set_ftol_abs(cfg.tol);
  opt1.set_xtol_abs(cfg.tol);
  
  for(i=0; i<ndim; i++)   %对初始化的参数进行检查，确保它们在指定的范围内。
  {
    if(x[i] < low[i])
      x[i] = low[i];
    else if(x[i] > up[i])
      x[i] = up[i];
  }

  opt1.optimize(x, f);   %调用 NLopt 进行优化，将结果保存在 x 和 f 中。
  rc = tnc(ndim, x.data(), &f, g.data(), func_tnc_cont_rm, args, low.data(), up.data(), 
      NULL, NULL, TNC_MSG_INFO|TNC_MSG_EXIT,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);     %调用 TNC（Truncated Newton Conjugate-Gradient）优化算法进行进一步的优化，并更新 x、f 和梯度 g。
  %将优化后的结果用于计算一些值，包括保存上一次迭代的参数、计算 Pixon 的数量、计算反演 Pixon 参数、计算卡方值，并输出结果。  
  f_old = f;
  num_old = pixon.compute_pixon_number();
  pixon.compute_rm_pixon(x.data());
  chisq_old = pixon.compute_chisquare(x.data());
  memcpy(x_old.data(), x.data(), ndim*sizeof(double));
  cout<<f_old<<"  "<<num_old<<"  "<<chisq_old<<endl;
  
  /* then pixel-dependent pixon size *//*然后像素依赖的像素大小*/
  iter = 0;    %iter是一个迭代次数的计数器，用于跟踪迭代的次数。
  do           %代码包含一个do-while循环，循环条件是pixon.pfft.get_ipxion_min() >= pixon_map_low_bound
  {
    iter++;
    cout<<"===================iter:"<<iter<<"==================="<<endl;
    
    flag = pixon.update_pixon_map();     %更新pixon映射。
    if(!flag)
      break;

    num = pixon.compute_pixon_number();   %计算pixon的数量。
    
    for(i=0; i<ndim; i++)                 %对变量数组x进行范围检查，确保其值在一定范围内。
    {
      if(x[i] < low[i])
        x[i] = low[i];
      else if(x[i] > up[i])
        x[i] = up[i];
    }

    opt1.optimize(x, f);         %使用某种优化算法（看起来像是TNC算法）对目标函数opt1进行优化，得到最小值和对应的参数x。
    rc = tnc(ndim, x.data(), &f, g.data(), func_tnc_cont_rm, args, low.data(), up.data(), 
      NULL, NULL, TNC_MSG_INFO|TNC_MSG_EXIT,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);      %计算一些与优化结果相关的值，如残差等。
    %输出一些信息，如目标函数值、pixon数量、卡方值等。
    pixon.compute_rm_pixon(x.data());
    chisq = pixon.compute_chisquare(x.data());
    cout<<f<<"  "<<num<<"  "<<chisq<<endl;

    if(f <= fmin)    %如果目标函数值小于等于fmin，则退出循环。
    {
      memcpy(x_old.data(), x.data(), ndim*sizeof(double));
      break;
    }
               %否则，保存一些旧的值用于下一次迭代。
    num_old = num;
    f_old = f;
    chisq_old = chisq;
    memcpy(x_old.data(), x.data(), ndim*sizeof(double));
  }while(pixon.pfft.get_ipxion_min() >= pixon_map_low_bound); 
  %循环结束后，输出一些最终的结果到文件中，包括背景值、pixon的时间-强度数据等。
  cout<<"bg: "<<x_old[npixel]<<endl;
  pixon.compute_rm_pixon(x_old.data());
  ofstream fout;
  string fname;
  fname = "data/resp_pixon.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<npixel; i++)
  {
    fout<<pixon.dt*(i-ipositive_tau)<<"  "<<pixon.image[i]<<"  "<<exp(x_old[i])<<endl;
  }
  fout.close();

  fname = "data/line_pixon.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<pixon.line.size; i++)
  {
    fout<<pixon.line.time[i]<<"  "<<pixon.itline[i]*pixon.line.norm<<"   "<<pixon.itline[i] - line.flux[i]<<endl;
  }
  fout.close();

  fname = "data/line_pixon_full.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<pixon.cont.size; i++)
  {
    fout<<pixon.cont.time[i]<<"  "<<pixon.rmline[i]*pixon.line.norm<<endl;
  }
  fout.close();
  
  fname = "data/cont_pixon.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<pixon.cont.size; i++)
  {
    fout<<pixon.cont.time[i]<<" "<<pixon.image_cont[i]*pixon.cont.norm<<endl;
  }
  fout.close();
  
  fname = "data/pixon_map_pixon.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<npixel; i++)
  {
    fout<<(i-ipositive_tau)*pixon.dt<<"  "<<pixon.pfft.pixon_sizes[pixon.pixon_map[i]]<<endl;
  }
  fout.close();

  memcpy(pimg, x_old.data(), ndim*sizeof(double));    %memcpy函数，其目的是将一个内存区域的内容复制到另一个内存区域。
}                                                     %将从 x_old.data() 开始的一段内存数据（通过计算的字节数为 ndim*sizeof(double)）复制到目标内存区域 pimg 中。

    %这段代码主要实现了一个 Pixon 重建的过程，结合了 TNC 和 NLopt 两个优化器，用于最小化一个目标函数。
/* set continuum free and use pixons to model continuum, uniform pixon sizes for RM *//*设置连续体自由，并使用像素来模拟连续体，统一像素大小为RM */l
void run_pixon_uniform(Data& cont_data, Data& cont_recon, Data& line, double *pimg, 
                            int npixel, int& npixon_size, int ipositive_tau, Config& cfg)
{
  cout<<"************************************************************"<<endl;
  cout<<"Start run_pixon_uniform..."<<endl;
  cout<<"npixon_size:"<<npixon_size<<endl;
  int i, iter;
  int npixon_size_cont = 10;       %声明并初始化整数变量 npixon_size_cont，赋值为10。
  PixonCont pixon(cont_data, cont_recon, line, npixel, npixon_size, npixon_size_cont, ipositive_tau, cfg.sensitivity);  %创建了一个名为 pixon 的 PixonCont 类的实例，传递了多个参数给构造函数。
  void *args = (void *)&pixon;     %将 pixon 实例的地址转换为 void 指针，并将其存储在 args 变量中。
  double f, f_old, num, num_old, chisq, chisq_old, df, dnum;     %声明多个双精度浮点型变量，用于存储函数运算过程中的一些值。

  /* TNC */
  int rc, maxCGit = cont_recon.size, maxnfeval = cfg.nfeval_max, nfeval, niter;   %声明整数变量和赋初值，其中 rc 用于存储函数运算的返回代码，maxCGit 和 maxnfeval 是迭代和函数调用的最大次数。
  double eta = -1.0, stepmx = -1.0, accuracy =  cfg.tol, fmin = pixon.cont_data.size, 
    ftol = cfg.tol, xtol = cfg.tol, pgtol = cfg.tol, rescale = -1.0;              %double 类型的变量的声明和初始化，用于配置 TNC（Truncated Newton Conjugate-Gradient） 和 NLopt 优化器的参数。
  
  /* NLopt */
  nlopt::opt opt0(nlopt::LN_BOBYQA, cont_recon.size);                            %创建一个 NLopt 中的优化器实例 opt0，使用 BOBYQA 算法。
  vector<double> x_cont(cont_recon.size), g_cont(cont_recon.size), x_old_cont(cont_recon.size);
  vector<double> low_cont(cont_recon.size), up_cont(cont_recon.size);            %创建一系列 vector<double> 类型的变量，用于存储优化器所需的数据（包括优化变量的初始值、梯度等）。

  /* bounds and initial values */
  for(i=0; i<cont_recon.size; i++)       %使用循环为优化问题设置边界和初始值。
  {
    low_cont[i] = fmax(0.0, cont_recon.flux[i] - 5.0 * cont_recon.error[i]);
    up_cont[i] =            cont_recon.flux[i] + 5.0 * cont_recon.error[i];
    x_cont[i] = cont_recon.flux[i];
  }
  
  opt0.set_min_objective(func_nlopt_cont, args);    %设置优化器的目标函数，即 func_nlopt_cont，并传递额外的参数 args。
  opt0.set_lower_bounds(low_cont);                  %设置优化器的下限、上限、最大评估次数等参数。
  opt0.set_upper_bounds(up_cont);
  opt0.set_maxeval(1000);
  opt0.set_ftol_abs(cfg.tol);
  opt0.set_xtol_abs(cfg.tol);
  
  opt0.optimize(x_cont, f);             %调用 NLopt 优化器进行优化，结果存储在 x_cont 和 f 中。
  rc = tnc(cont_recon.size, x_cont.data(), &f, g_cont.data(), func_tnc_cont, args, 
      low_cont.data(), up_cont.data(), NULL, NULL, TNC_MSG_INFO|TNC_MSG_EXIT,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);      %调用 TNC 优化器进行优化，结果存储在 x_cont、f 和 g_cont 中。
   %这段代码是一个循环的优化过程，主要用于反映物理现象或实验结果的数据拟合。
  f_old = f;
  num_old = pixon.compute_pixon_number_cont();   %计算并存储当前pixon的数量。
  pixon.compute_cont(x_cont.data());     %基于给定的数据x_cont，计算并更新pixon的相关内容。
  chisq_old = pixon.compute_chisquare_cont(x_cont.data());    %计算并存储当前pixon的卡方统计量。
  memcpy(x_old_cont.data(), x_cont.data(), cont_recon.size*sizeof(double));   %将当前数据备份到x_old_cont中。
  cout<<f_old<<"  "<<num_old<<"  "<<chisq_old<<endl;
 
  while(npixon_size_cont>2)    %进入一个循环，条件是npixon_size_cont大于2。
  {
    npixon_size_cont--;
    cout<<"npixon_size_cont:"<<npixon_size_cont<<",  size: "<<pixon.pfft_cont.pixon_sizes[npixon_size_cont-1]<<endl;
    
    pixon.reduce_ipixon_cont();    %减少pixon的数量。
    num = pixon.compute_pixon_number_cont();    %重新计算pixon的数量。
    
    for(i=0; i<cont_recon.size; i++)     %对数据进行一些范围约束，确保在规定范围内。
    {
      if(x_cont[i] < low_cont[i])
        x_cont[i] = low_cont[i];
      else if(x_cont[i] > up_cont[i])
        x_cont[i] = up_cont[i];
    }

    opt0.optimize(x_cont, f);    %使用某种优化算法对目标函数进行优化，更新x_cont和f。
    rc = tnc(cont_recon.size, x_cont.data(), &f, g_cont.data(), func_tnc_cont, args, 
      low_cont.data(), up_cont.data(), NULL, NULL, TNC_MSG_INFO|TNC_MSG_EXIT,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);    %调用TNC优化库进行进一步的优化。
     
    pixon.compute_cont(x_cont.data());    %根据优化后的参数重新计算pixon的相关内容。
    chisq = pixon.compute_chisquare_cont(x_cont.data());   %计算新的卡方统计量。
    cout<<f<<"  "<<num<<"  "<<chisq<<endl;

    if(f <= fmin)
    {
      memcpy(x_old_cont.data(), x_cont.data(), cont_recon.size*sizeof(double));
      break;
    }
    
    df = f-f_old;
    dnum = num - num_old;

    if(-df < dnum * (1.0 + 1.0/sqrt(2.0*num)))
    {
      pixon.increase_ipixon_cont();
      break;
    }

    num_old = num;
    f_old = f;
    chisq_old = chisq;
    memcpy(x_old_cont.data(), x_cont.data(), cont_recon.size*sizeof(double));
  }
  
  pixon.compute_cont(x_old_cont.data());    %基于最终参数重新计算pixon的相关内容。
  ofstream fp;
  fp.open("data/cont_recon_pixon_uniform.txt");    %打开一个文件用于保存结果。
  for(i=0; i<cont_recon.size; i++)    %将最终结果写入文件。
  {
    fp<<pixon.cont.time[i]<<" "<<pixon.image_cont[i]*pixon.cont.norm<<endl;
  }
  fp.close();

  %主要是进行非线性优化操作，使用了NLopt库d
  cout<<"Start to RM"<<endl;
  /* then continuum and line reverberation */
  pixon.cont.set_data(pixon.image_cont);   %将pixon对象中的image_cont数据设置给cont对象。
  int ndim = npixel + 1 + pixon.cont.size;
  maxCGit = ndim;
  fmin = pixon.line.size + pixon.cont_data.size;

  /* NLopt */创建NLopt优化器对象
  nlopt::opt opt1(nlopt::LN_BOBYQA, ndim);   %选择了BOBYQA算法。
  vector<double> x(ndim), g(ndim), x_old(ndim);   %初始化一些变量
  vector<double> low(ndim), up(ndim);     %以及lower和up

  /* bounds and initial values */
  /* transfer function *//*边界和初始值*//*传递函数*/      %为变量x、low、up设置初始值和边界条件：
  for(i=0; i<npixel; i++)   %对于transfer function中的npixel个变量，设置初始值和边界。
  {
    low[i] = -100.0;
    up[i] =  10.0;
    x[i] = log(1.0/(npixel * pixon.dt));
  }
  /* background */
  if(cfg.fix_bg)      %对于background，根据条件设置初始值和边界。
  {
    low[npixel] = up[npixel] = x[npixel] = cfg.bg;
  }
  else 
  {
    low[npixel] = -1.0;
    up[npixel] = 1.0;
    x[npixel] = 0.0;
  }
  /* continuuum reconstruction */
  for(i=0; i<pixon.cont.size; i++)    %对于continuum reconstruction，设置初始值和边界。
  {
    low[i+npixel+1] = fmax(0.0, cont_recon.flux[i] - 5.0 * cont_recon.error[i]);
    up[i+npixel+1] =            cont_recon.flux[i] + 5.0 * cont_recon.error[i];
    x[i+npixel+1] = pixon.cont.flux[i];
  }
  %使用NLopt设置优化目标函数和相关参数：
  opt1.set_min_objective(func_nlopt_cont_rm, args);   %设置最小化目标函数。
  opt1.set_lower_bounds(low);                         %设置下限和上限。
  opt1.set_upper_bounds(up);
  opt1.set_maxeval(1000);                             %设置最大评估次数。
  opt1.set_ftol_abs(cfg.tol);                         %设置终止容差。
  opt1.set_xtol_abs(cfg.tol);
  
  for(i=0; i<ndim; i++)     %将初始值x调整到边界内。
  {
    if(x[i] < low[i])
      x[i] = low[i];
    else if(x[i] > up[i])
      x[i] = up[i];
  }

  opt1.optimize(x, f);     %使用NLopt进行优化：
  rc = tnc(ndim, x.data(), &f, g.data(), func_tnc_cont_rm, args, low.data(), up.data(), 
      NULL, NULL, TNC_MSG_INFO|TNC_MSG_EXIT,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);    %使用TNC算法进行优化：
    
  f_old = f;    %记录一些变量的旧值：f_old、num_old、chisq_old，并更新pixon对象。
  num_old = pixon.compute_pixon_number();
  pixon.compute_rm_pixon(x.data());
  chisq_old = pixon.compute_chisquare(x.data());
  memcpy(x_old.data(), x.data(), ndim*sizeof(double));
  cout<<f_old<<"  "<<num_old<<"  "<<chisq_old<<endl;
  
  iter = 0;
  while(npixon_size>pixon_map_low_bound+1)     %进入一个循环，条件是npixon_size大于pixon_map_low_bound + 1
  {
    iter++;
    cout<<"===================iter:"<<iter<<"==================="<<endl;
    npixon_size--;
    cout<<"npixon_size:"<<npixon_size<<",  size: "<<pixon.pfft.pixon_sizes[npixon_size-1]<<endl;

    pixon.reduce_pixon_map_all();       %调用 reduce_pixon_map_all() 方法
    num = pixon.compute_pixon_number(); %计算 pixon 的数量

    for(i=0; i<ndim; i++)      %对一些数组进行限制操作，确保数组中的值在一定范围内。
    {
      if(x[i] < low[i])
        x[i] = low[i];
      else if(x[i] > up[i])
        x[i] = up[i];
    }

    opt1.optimize(x, f);      %调用 optimize() 方法，可能是进行某种优化
    rc = tnc(ndim, x.data(), &f, g.data(), func_tnc_cont_rm, args, low.data(), up.data(), 
      NULL, NULL, TNC_MSG_INFO|TNC_MSG_EXIT,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);     %调用 tnc() 方法，可能是使用 TNC（Truncated Newton Conjugate-Gradient）算法进行优化。
    
    pixon.compute_rm_pixon(x.data());    %调用一些方法计算一些值，并输出相关信息。
    chisq = pixon.compute_chisquare(x.data());
    cout<<f<<"  "<<num<<"  "<<chisq<<endl;

    if(f <= fmin)
    {
      memcpy(x_old.data(), x.data(), ndim*sizeof(double));      %使用 memcpy 将当前的优化变量 x 的值复制到 x_old，即保存当前的最优解。
      break;
    }
    
    df = f-f_old;
    dnum = num - num_old;

    if(-df < dnum * (1.0 + cfg.sensitivity/sqrt(2.0*num)))
    {
      /* pixon size goes back to previous value */    %将 pixon 的大小恢复到先前的值，可能是通过调用 increase_pixon_map_all() 方法实现的。
      pixon.increase_pixon_map_all();
      break;
    }

    num_old = num;
    f_old = f;
    chisq_old = chisq;
    memcpy(x_old.data(), x.data(), ndim*sizeof(double));    %使用 memcpy 将当前的优化变量 x 的值复制到 x_old，即更新最优解。
  }
  
  cout<<"bg: "<<x_old[npixel]<<endl;
  pixon.compute_rm_pixon(x_old.data());
  ofstream fout;    %打开一个文件输出流对象 ofstream fout; 用于写入文件。
  string fname;     %打开文件并写入数据到 "data/resp_pixon_uniform.txt_" 文件中。
  fname = "data/resp_pixon_uniform.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<npixel; i++)
  {
    fout<<pixon.dt*(i-ipositive_tau)<<"  "<<pixon.image[i]<<"   "<<exp(x_old[i])<<endl;
  }
  fout.close();

  fname = "data/line_pixon_uniform.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<pixon.line.size; i++)
  {
    fout<<pixon.line.time[i]<<"  "<<pixon.itline[i]*pixon.line.norm<<"   "<<pixon.itline[i] - line.flux[i]<<endl;
  }
  fout.close();

  fname = "data/line_pixon_uniform_full.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<pixon.cont.size; i++)
  {
    fout<<pixon.cont.time[i]<<"  "<<pixon.rmline[i]*pixon.line.norm<<endl;
  }
  fout.close();
  
  fname = "data/cont_pixon_uniform.txt_" + to_string(cfg.pixon_basis_type);
  fp.open(fname);
  for(i=0; i<pixon.cont.size; i++)
  {
    fp<<pixon.cont.time[i]<<" "<<pixon.image_cont[i]*pixon.cont.norm<<endl;
  }
  fp.close();

  fname = "data/pixon_map_pixon_uniform.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<npixel; i++)
  {
    fout<<(i-ipositive_tau)*pixon.dt<<"  "<<pixon.pfft.pixon_sizes[pixon.pixon_map[i]]<<endl;
  }
  fout.close();

  memcpy(pimg, x_old.data(), ndim*sizeof(double));
}


%这段代码是一个优化算法的实现，使用了两种不同的优化器（NLopt 和 TNC），并在每次迭代中计算了一些相关的数值，包括目标函数值、pixon 数量和卡方值。
/* set continuum fixed from a drw reconstruction and use pixel dependent pixon sizes for RM *//*设置连续体固定从绘制重建和使用像素依赖的像素大小的RM */
void run_contfix(Data& cont, Data& line, double *pimg, int npixel, int& npixon_size, int ipositive_tau, Config& cfg)
{
  cout<<"************************************************************"<<endl;
  cout<<"Start run_contfix..."<<endl;
  cout<<"npixon_size:"<<npixon_size<<endl;
  int i, iter;    %整型变量
  Pixon pixon(cont, line, npixel, npixon_size, ipositive_tau, cfg.sensitivity);
  void *args = (void *)&pixon;
  bool flag;
  double f, f_old, num, num_old, chisq, chisq_old, df, dnum;    %双精度浮点型变量
  
  int ndim = npixel + 1;  /* include one parameter for background */
  /* TNC */   %配置 TNC 优化算法的参数，包括最大迭代次数 maxCGit、最大函数调用次数 maxnfeval、容许的误差 accuracy 等。
  int rc, maxCGit = ndim, maxnfeval = cfg.nfeval_max, nfeval, niter;
  double eta = -1.0, stepmx = -1.0, accuracy =  cfg.tol, fmin = pixon.line.size, 
    ftol = cfg.tol, xtol = cfg.tol, pgtol = cfg.tol, rescale = -1.0;
  
  /* NLopt */   %创建 NLopt 优化器对象 opt0，使用 BOBYQA 算法，维度为 ndim。
  nlopt::opt opt0(nlopt::LN_BOBYQA, ndim);
  vector<double> x(ndim), g(ndim), x_old(ndim);
  vector<double>low(ndim), up(ndim);

  /* bounds and initial values */    %设定变量的边界和初始值，其中包括传递函数的部分和背景的部分。
  /* transfer function */
  for(i=0; i<npixel; i++)
  {
    low[i] = -100.0;
    up[i] =  10.0;
    x[i] = log(1.0/(npixel * pixon.dt));
  }
  /* background */
  if(cfg.fix_bg)
  {
    low[npixel] = up[npixel] = x[npixel] = cfg.bg;     %将背景参数的边界和初始值设为 cfg.bg
  }
  else 
  {
    low[npixel] = -1.0;
    up[npixel] = 1.0;
    x[npixel] = 0.0;
  }

  opt0.set_min_objective(func_nlopt, args);     %使用 NLopt 进行优化，调用 optimize 方法，并传递优化变量 x 和目标函数的指针 func_nlopt。这里采用的是 BOBYQA 算法。
  opt0.set_lower_bounds(low);
  opt0.set_upper_bounds(up);
  opt0.set_maxeval(1000);
  opt0.set_ftol_abs(cfg.tol);
  opt0.set_xtol_abs(cfg.tol);
  
  opt0.optimize(x, f);
  rc = tnc(ndim, x.data(), &f, g.data(), func_tnc, args, low.data(), up.data(), 
      NULL, NULL, TNC_MSG_INFO|TNC_MSG_EXIT,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);    %使用 TNC 进行优化，调用 tnc 函数，传递优化变量 x、目标函数的指针 func_tnc 以及一些其他参数。
  
  f_old = f;
  num_old = pixon.compute_pixon_number();   %将优化后的目标函数值 f 赋值给 f_old，计算当前的 pixon 数量并存储在 num_old 中。
  pixon.compute_rm_pixon(x.data());         %调用 compute_rm_pixon 方法，可能用于计算和更新 pixon 的一些值。
  chisq_old = pixon.compute_chisquare(x.data());     %计算当前的卡方值，并将其存储在 chisq_old 中。
  memcpy(x_old.data(), x.data(), ndim*sizeof(double));   %使用 memcpy 将当前的优化变量 x 复制到 x_old，即保存当前的最优解。
  cout<<f_old<<"  "<<num_old<<"  "<<chisq_old<<endl;

  /* then pixel-dependent pixon size *//*然后像素依赖的像素大小*/
  iter = 0;
  do
  {
    iter++;
    cout<<"===================iter:"<<iter<<"==================="<<endl;
    
    flag = pixon.update_pixon_map();     %调用 pixon.update_pixon_map() 更新一些地图。
    if(!flag)     %果变量 flag 的逻辑取反结果为真（即 !flag 为真），则执行 break;
      break;

    num = pixon.compute_pixon_number();   %计算 pixon 的数量并存储在 num 中。
    
    for(i=0; i<ndim; i++)    %对数组 x 进行范围限制。
    {
      if(x[i] < low[i])
        x[i] = low[i];
      else if(x[i] > up[i])
        x[i] = up[i];
    }

    opt0.optimize(x, f);    %调用 opt0.optimize(x, f) 进行优化。
    rc = tnc(ndim, x.data(), &f, g.data(), func_tnc, args, low.data(), up.data(), 
      NULL, NULL, TNC_MSG_INFO|TNC_MSG_EXIT,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);     %使用 TNC 算法进行进一步的优化，结果存储在 rc 中。
    
    pixon.compute_rm_pixon(x.data());
    chisq = pixon.compute_chisquare(x.data());
    cout<<f<<"  "<<num<<"  "<<chisq<<endl;

    if(f <= pixon.line.size)
    {
      memcpy(x_old.data(), x.data(), ndim*sizeof(double));
      break;
    }

    df = f-f_old;
    dnum = num - num_old;   

    num_old = num;
    f_old = f;
    chisq_old = chisq;
    memcpy(x_old.data(), x.data(), ndim*sizeof(double));
  }while(pixon.pfft.get_ipxion_min() >= pixon_map_low_bound); 

  cout<<"bg: "<<x_old[npixel]<<endl;
  
  pixon.compute_rm_pixon(x_old.data());
  ofstream fout;   %这行代码声明了一个名为 fout 的 ofstream 对象，该对象用于将数据写入文件。这只是一个声明，如果要使用它进行实际的文件写入操作，还需要在后续的代码中打开文件并使用 fout 对象进行写入。
  string fname;    %打开名为 "data/resp_contfix.txt_" + to_string(cfg.pixon_basis_type) 的文件，写入一些数据。
  fname = "data/resp_contfix.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<npixel; i++)
  {
    fout<<pixon.dt*(i-ipositive_tau)<<"  "<<pixon.image[i]<<"   "<<exp(x_old[i])<<endl;
  }
  fout.close();
  
  fname = "data/line_contfix.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<pixon.line.size; i++)
  {
    fout<<pixon.line.time[i]<<"  "<<pixon.itline[i]*pixon.line.norm<<"   "<<pixon.itline[i] - line.flux[i]<<endl;
  }
  fout.close();

  fname = "data/line_contfix_full.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<pixon.cont.size; i++)
  {
    fout<<pixon.cont.time[i]<<"  "<<pixon.rmline[i]*pixon.line.norm<<endl;
  }
  fout.close();

  fname = "data/pixon_map_contfix.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<npixel; i++)
  {
    fout<<(i-ipositive_tau)*pixon.dt<<"  "<<pixon.pfft.pixon_sizes[pixon.pixon_map[i]]<<endl;
  }
  fout.close();
  
  memcpy(pimg, x_old.data(), ndim*sizeof(double));
}

/* set continuum fixed from a drw reconstruction and use uniform pixon sizes for RM */
void run_contfix_uniform(Data& cont, Data& line, double *pimg, int npixel, int& npixon_size, int ipositive_tau, Config& cfg)
{
  cout<<"************************************************************"<<endl;
  cout<<"Start run_contfix_uniform..."<<endl;
  cout<<"npixon_size:"<<npixon_size<<endl;
  int i;
  Pixon pixon(cont, line, npixel, npixon_size, ipositive_tau, cfg.sensitivity);
  void *args = (void *)&pixon;
  double f, f_old, num_old, num, df, dnum, chisq, chisq_old;
  int iter;
 
  int ndim = npixel + 1;
  /* TNC */
  int rc, maxCGit = ndim, maxnfeval = cfg.nfeval_max, nfeval, niter;
  double eta = -1.0, stepmx = -1.0, accuracy = cfg.tol, fmin = pixon.line.size, 
    ftol = cfg.tol, xtol = cfg.tol, pgtol = cfg.tol, rescale = -1.0;

  /* NLopt */
  nlopt::opt opt0(nlopt::LN_BOBYQA, ndim);
  vector<double> x(ndim), g(ndim), x_old(ndim);
  vector<double> low(ndim), up(ndim);
  
  /* bounds and initial values */
  /* transfer function */
  for(i=0; i<npixel; i++)
  {
    low[i] = -100.0;
    up[i] =  10.0;
    x[i] = log(1.0/(npixel * pixon.dt));
  }
  /* background */
  if(cfg.fix_bg)
  {
    low[npixel] = up[npixel] = x[npixel] = cfg.bg;
  }
  else 
  {
    low[npixel] = -1.0;
    up[npixel] = 1.0;
    x[npixel] = 0.0;
  }

  /* NLopt settings */
  opt0.set_min_objective(func_nlopt, args);
  opt0.set_lower_bounds(low);
  opt0.set_upper_bounds(up);
  opt0.set_maxeval(1000);
  opt0.set_ftol_abs(cfg.tol);
  opt0.set_xtol_abs(cfg.tol);
   
  opt0.optimize(x, f);
  rc = tnc(ndim, x.data(), &f, g.data(), func_tnc, args, low.data(), up.data(), 
      NULL, NULL, TNC_MSG_INFO|TNC_MSG_EXIT,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);
  
  f_old = f;
  num_old = pixon.compute_pixon_number();
  pixon.compute_rm_pixon(x.data());
  chisq_old = pixon.compute_chisquare(x.data());
  memcpy(x_old.data(), x.data(), ndim*sizeof(double));
  cout<<f_old<<"  "<<num_old<<"  "<<chisq_old<<endl;

  iter = 0;
  while(npixon_size>pixon_map_low_bound+1)
  {
    iter++;
    cout<<"===================iter:"<<iter<<"==================="<<endl;
    npixon_size--;
    cout<<"npixon_size:"<<npixon_size<<",  size: "<<pixon.pfft.pixon_sizes[npixon_size-1]<<endl;

    pixon.reduce_pixon_map_all();
    num = pixon.compute_pixon_number();
    
    for(i=0; i<ndim; i++)
    {
      if(x[i] < low[i])
        x[i] = low[i];
      else if(x[i] > up[i])
        x[i] = up[i];
    }

    opt0.optimize(x, f);
    rc = tnc(ndim, x.data(), &f, g.data(), func_tnc, args, low.data(), up.data(), 
      NULL, NULL, TNC_MSG_INFO|TNC_MSG_EXIT,
      maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol,
      rescale, &nfeval, &niter, NULL);
    
    pixon.compute_rm_pixon(x.data());
    chisq = pixon.compute_chisquare(x.data());
    cout<<f<<"  "<<num<<"  "<<chisq<<endl;

    if(f <= pixon.line.size)
    {
      memcpy(x_old.data(), x.data(), ndim*sizeof(double));
      break;
    }
    
    df = f-f_old;
    dnum = num - num_old;

    if(-df < dnum * (1.0 + cfg.sensitivity/sqrt(2.0*num)))
    {
      /* pixon size goes back to previous value */
      pixon.increase_pixon_map_all();
      break;
    }

    num_old = num;
    f_old = f;
    chisq_old = chisq;
    memcpy(x_old.data(), x.data(), ndim*sizeof(double));
  }
  
  cout<<"bg: "<<x_old[npixel]<<endl;
  
  pixon.compute_rm_pixon(x_old.data());
  ofstream fout;
  string fname;
  fname = "data/resp_contfix_uniform.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<npixel; i++)
  {
    fout<<pixon.dt*(i-ipositive_tau)<<"   "<<pixon.image[i]<<"  "<<exp(x_old[i])<<endl;
  }
  fout.close();
  
  fname = "data/line_contfix_uniform.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<pixon.line.size; i++)
  {
    fout<<pixon.line.time[i]<<"  "<<pixon.itline[i]*pixon.line.norm<<"   "<<pixon.itline[i] - line.flux[i]<<endl;
  }
  fout.close();

  fname = "data/line_contfix_uniform_full.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<pixon.cont.size; i++)
  {
    fout<<pixon.cont.time[i]<<"  "<<pixon.rmline[i]*pixon.line.norm<<endl;
  }
  fout.close();
  
  fname = "data/pixon_map_contfix_uniform.txt_" + to_string(cfg.pixon_basis_type);
  fout.open(fname);
  for(i=0; i<npixel; i++)
  {
    fout<<(i-ipositive_tau)*pixon.dt<<"  "<<pixon.pfft.pixon_sizes[pixon.pixon_map[i]]<<endl;
  }
  fout.close();

  memcpy(pimg, x_old.data(), ndim*sizeof(double));

}
