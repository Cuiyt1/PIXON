/*
 *  PIXON
 *  A Pixon-based method for reconstructing velocity-delay map in reverberation mapping.
 * 
 *  Yan-Rong Li, liyanrong@mail.ihep.ac.cn
 * 
 */
#include "utilities.hpp"
#include "pixon_cont.hpp"
%初始化多个指针为NULL。
PixonCont::PixonCont()
{
  image_cont = pseudo_image_cont = NULL;
  residual_cont = NULL;
  grad_chisq_cont = NULL;
  grad_mem_cont = NULL;
  Kpixon = NULL;
}
%通过使用提供的参数调用其构造函数继承基类Pixon。为数组residual_cont、image_cont等分配动态内存。使用初始化列表初始化成员变量
PixonCont::PixonCont(
  Data& cont_data_in, Data& cont_in, Data& line_data_in, int npixel_in,  
  int npixon_in, int npixon_cont_in, int ipositive_in, double sensitivity_in
  )
  :Pixon(cont_in, line_data_in, npixel_in, npixon_in, ipositive_in, sensitivity_in),
   cont_data(cont_data_in),
   pfft_cont(cont_in.size, npixon_cont_in),
   rmfft_pixon(cont_in.size, dt, fmax(npixel-ipositive_in, ipositive_in)),
   ipixon_cont(npixon_cont_in-1)
{
  residual_cont = new double[cont_data_in.size];
  image_cont = new double[cont_in.size];
  pseudo_image_cont = new double[cont_in.size];
  grad_chisq_cont = new double[cont_in.size];
  grad_mem_cont = new double[cont_in.size];
  Kpixon = new double[2*cont_in.size];
}
%在删除动态分配的数组之前检查npixel是否大于0，以避免未初始化内存的问题。
PixonCont::~PixonCont()
{
  if(npixel > 0)
  {
    delete[] residual_cont;
    delete[] image_cont;
    delete[] pseudo_image_cont;
    delete[] grad_chisq_cont;
    delete[] grad_mem_cont;
    delete[] Kpixon;
  }
}
%接受一个数组x作为输入。
%将值从x复制到pseudo_image_cont。
%使用pfft_cont将pseudo_image_cont与pixons进行卷积，结果存储在inage_cont中。
%确保inage_cont中的值为正数。如果一个值小于或等于0.0，则设置为EPs。
%将cont对象中的数据重置为inage_cont中的值。
%通过从实际数据中减去插值值来计算残差，并将其存储在residual_cont中。
{
  int i;
  double t;
  /* convolve with pixons for continuum */
  for(i=0; i<cont.size; i++)
  {
    pseudo_image_cont[i] = x[i];
  }
  pfft_cont.convolve(pseudo_image_cont, ipixon_cont, image_cont);
  
  /* enforce positive image */
  for(i=0; i<cont.size; i++)
  {
    if(image_cont[i] <= 0.0)
      image_cont[i] = EPS;
  }
  /* reset Data cont */
  cont.set_data(image_cont);

  for(i=0; i<cont_data.size; i++)
  {
    t = cont_data.time[i];
    residual_cont[i] = Pixon::interp_cont(t) - cont_data.flux[i];
  }

  return;
}

/* compute rm amd pixon convolutions */
void PixonCont::compute_rm_pixon(const double *x)
{
  int i;
  double t;  %声明变量 i 和 t
             %数组 x 中的前 npixel+1 个元素用于传递函数和背景，其后是连续谱的参数。
  /* first npixel+1 entities are for transfer functions and bg, followed by continuum parameters */
  compute_cont(x + npixel + 1);          %调用 compute_cont 函数，传递从 npixel + 1 开始到数组末尾的子集。
  rmfft.set_data(image_cont, cont.size); %使用 image_cont 和 cont 的大小设置 rmfft 的数据。
  Pixon::compute_rm_pixon(x);            %调用基类 Pixon 中的 compute_rm_pixon 函数，传递整个数组 x
}

double PixonCont::compute_chisquare_cont(const double *x)
{
  int i;                                %声明一个循环变量 i。
  
  chisq_cont = 0;                       %将 chisq_cont 初始化为0。
  for(i=0; i<cont_data.size; i++)       %遍历 cont_data 中的数据，计算对连续谱的卡方贡献（chisq_cont）。
  {
    chisq_cont += (residual_cont[i] * residual_cont[i])/(cont_data.error[i] * cont_data.error[i]);
  }
  
  return chisq_cont;
}

double PixonCont::compute_chisquare(const double *x)
{
  /* calculate chi square */                              %计算整个模型的卡方，通过将线的卡方（chisq_line）和连续谱的卡方（chisq_cont）相加得到。
  chisq_line = Pixon::compute_chisquare(x);               %调用基类 Pixon 中的 compute_chisquare 函数计算线的卡方。
  chisq_cont = compute_chisquare_cont(x + npixel + 1);    %调用 compute_chisquare_cont 函数计算连续谱的卡方。
  chisq = chisq_line + chisq_cont;
  return chisq;
}

double PixonCont::compute_mem(const double *x)            %通过将线的MEM（mem_line）和连续谱的MEM（mem_cont）相加得到总的MEM。
{
  mem_line = Pixon::compute_mem(x);                       %调用基类 Pixon 中的 compute_mem 函数计算线的MEM。
  mem_cont = compute_mem_cont(x + npixel + 1);            %调用 compute_mem_cont 函数计算连续谱的MEM。
  mem =  mem_line + mem_cont;
  return mem;
}

double PixonCont::compute_mem_cont(const double *x)       %调用 compute_mem_cont 函数计算连续谱的MEM。
{
  double Itot, num, alpha;
  int i;

  Itot = 0.0;
  for(i=0; i<cont.size; i++)                              %通过对 image_cont 中的值求和计算连续谱的总强度（Itot）。
  {
    Itot += image_cont[i];
  }

  num = compute_pixon_number_cont();                      %使用函数 compute_pixon_number_cont 计算连续谱中的pixon数。
  alpha = log(num)/log(cont.size);                        %使用对数计算参数 alpha。

  mem_cont = 0.0;                                         %通过对 image_cont 数组进行迭代计算连续谱的MEM。
  for(i=0; i<cont.size; i++)
  {
    mem_cont += (image_cont[i]/Itot) * log(image_cont[i]/Itot);
  }
  
  mem_cont *= 2.0*alpha;
  return mem_cont;
}

/* compute pixon number of continuum */
double PixonCont::compute_pixon_number_cont()             %计算连续谱中的pixon数。
{
  int i;
  double num, psize;
    
  num = 0.0;                                              %遍历 cont.size，获取pixon大小并用其累加到pixon数的贡献中。
  for(i=0; i<cont.size; i++)
  {
    psize = pfft_cont.pixon_sizes[ipixon_cont];
    num += pixon_norm(psize);
  }
  return num;
}

/* compute total pixon number */                         %计算总的pixon数。
double PixonCont::compute_pixon_number()
{
  return compute_pixon_number_cont() + Pixon::compute_pixon_number();
}

void PixonCont::compute_chisquare_grad(const double *x)
{
  Pixon::compute_chisquare_grad(x);       /* derivative of chisq_line with respect to transfer function计算关于传输函数的chisq_line导数*/   
  compute_chisquare_grad_cont(x+npixel+1);  /* derivative of chisq_cont with respect to continuum计算关于连续区域的chisq_cont导数*/

  /* derivative of chisq_line with respect to continuum */                  %计算关于连续区域的chisq_line导数
  int i, j;
  double psize, grad_in, grad_out, K, t;
  
  rmfft_pixon.set_resp_real(image, npixel, ipositive);                     %设置了一个实数响应，涉及一些图像处理的操作，传递了图像、像素数目和正像素数目作为参数。
  psize = pfft_cont.pixon_sizes[ipixon_cont]; /* uniform pixon size for continuum */%计算continuum 中 pixon 的统一大小，该大小存储在 pixon_sizes 数组的 ipixon_cont 索引处。
  for(i=0; i<cont.size; i++)                                               %外层循环遍历 cont.size 次
  {
    for(j=0; j<cont.size; j++)                                             %内层循环遍历 cont.size 次。
    {
      resp_pixon[j] = pixon_function(j, i, psize);                        %调用 pixon_function 函数计算了一些响应值，并将其存储在 resp_pixon 数组中。
    }
    rmfft_pixon.set_data(resp_pixon, cont.size);                          %通过 rmfft_pixon.set_data 将这些响应值传递给某种数据结构，
    rmfft_pixon.convolve_simple(conv_pixon);                              %通过 rmfft_pixon.convolve_simple 执行了卷积操作，将结果存储在 conv_pixon 中。

    grad_out = 0.0;
    for(j=0; j<line.size; j++)                                            %内层循环遍历 line.size 次，计算了一些梯度值，并将其累加到 grad_out 中。
    {
      t = line.time[j];
      grad_in = interp_pixon(t);
      grad_out += grad_in * residual[j]/line.error[j]/line.error[j];
    }
    grad_chisq_cont[i] += grad_out * 2.0; /* chisq = chisq_cont + chisq_line */  %将计算得到的梯度值 grad_out 乘以2，然后加到 grad_chisq_cont[i] 中。
  }
}
/* Kpixon = K((tj-ti)/psize) */                                                  %实现了一个插值函数 interp_Kpixon，用于计算 Kpixon 数组中某个时刻 t 的插值值。
double PixonCont::interp_Kpixon(double t)
{
  int it;

  it = t/dt + cont.size;                                                         %声明了一个整数变量 it，用于表示时间 t 对应的数组索引。通过计算 t/dt + cont.size 得到 it 的值。
  if(it < 0)
    return Kpixon[0];                                                            %如果 it 小于0，则返回 Kpixon 数组的第一个元素。
  else if(it >= 2*cont.size -1)
    return Kpixon[2*cont.size-1];                                                %如果 it 大于等于 2*cont.size - 1，则返回 Kpixon 数组的最后一个元素。这是为了处理边界情况，确保不越界。

  return Kpixon[it] + (Kpixon[it+1] - Kpixon[it])/dt * (t - (it-cont.size)*dt);  %如果 it 位于数组范围内，通过线性插值计算了 Kpixon 在时间 t 处的插值值。使用线性插值公式，通过两个相邻点的斜率和距离来估算目标点的值。
}

%这个函数用于计算与连续值有关的卡方偏导数。它初始化一些变量，如psize、grad_in、K等。它使用pixon_function计算Kpixon数组。然后，它迭代连续体大小(contt . size)，并根据插值和其他计算计算grad_chisq_cont中每个元素的梯度。
void PixonCont::compute_chisquare_grad_cont(const double *x)                     %计算关于连续性的 chisquare 的梯度
{
  int i, j, jt, jrange1, jrange2;                                                %定义变量，包括循环中的索引（i，j，jt等），时间（tj），pixon 大小（psize），梯度（grad_in，K等）等。
  double tj;
  double psize, grad_in, K, jt_real;
/* derivative of chisq_cont with respect to continuum */                         %计算chisq_cont对连续体的导数                       
  
  /* uniform pixon size */                                                    
  psize = pfft_cont.pixon_sizes[ipixon_cont];                                    %获取 PixonCont 类中的 pfft_cont 对象的 pixon 大小。
  for(j=0; j<cont.size*2; j++)
  {
    Kpixon[j] = pixon_function(j-cont.size, 0, psize);  /* correspoding to time from -n*dt to +n*dt */  %使用 pixon_function 函数计算并存储 Kpixon 数组中的值，该函数接受参数 (j-cont.size, 0, psize)。
  }
  for(i=0; i<cont.size; i++)
  {
    jrange1 = fmin(fmax(0, i - pixon_size_factor * psize), cont_data.size-1);    %计算 jrange1 的值，限制在 0 到 cont_data.size-1 之间。
    jrange2 = fmin(cont_data.size-1, i + pixon_size_factor * psize);

    grad_in = 0.0;                                                               %初始化 grad_in 为 0。
    for(j=jrange1; j<=jrange2; j++)                                              %遍历一个范围从 jrange1 到 jrange2 的循环。
    {
      tj = cont_data.time[j];      
      K = interp_Kpixon(tj - cont.time[i]);
      grad_in += K * residual_cont[j]/cont_data.error[j]/cont_data.error[j];
    }
    grad_chisq_cont[i] = 2.0 * grad_in;
  }
}
%这个函数从基类Pixon调用compute_men_grad，然后用修改后的形参调用compute_men_grad_cont，计算内存的梯度，调用了 Pixon 类的 compute_mem_grad 函数以及自身的 compute_mem_grad_cont 函数。
void PixonCont::compute_mem_grad(const double *x)
{
  Pixon::compute_mem_grad(x);
  compute_mem_grad_cont(x+npixel+1);
}
%这个函数计算记忆项的连续统部分的梯度。它将Itot计算为image_cont元素的和，并计算num和alpha。然后迭代连续体大小，并基于pixon_function、image_cont和其他参数计算梯度。
void PixonCont::compute_mem_grad_cont(const double *x)
{
  double Itot, num, alpha, grad_in, psize, K;
  int i, j, jrange1, jrange2;
  Itot = 0.0;
  for(i=0; i<cont.size; i++)
  {
    Itot += image_cont[i];                           %遍历 image_cont 数组并累加其中的值。
  }
  num = compute_pixon_number_cont();
  alpha = log(num)/log(cont.size);
  
  /* uniform pixon size */
  psize = pfft_cont.pixon_sizes[ipixon_cont];
  for(i=0; i<cont.size; i++)
  {       
    jrange1 = fmax(0, i - pixon_size_factor * psize);
    jrange2 = fmin(cont.size-1, i + pixon_size_factor*psize);
    grad_in = 0.0;
    for(j=jrange1; j<=jrange2; j++)
    {
      K = pixon_function(j, i, psize);
      grad_in += (1.0 + log(image_cont[j]/Itot)) * K;
    }
    grad_mem_cont[i] = 2.0 * alpha * grad_in / Itot;
  }
}
%这个函数将ippixon_cont的值减少1，并从pfft_cont对象中调用reduce_pixon_min。似乎是对Pixon的某种调整。
void PixonCont::reduce_ipixon_cont()
{
  int i;
  pfft_cont.reduce_pixon_min();
  ipixon_cont--;
}
%这个函数将ippixon_cont的值增加1，并从pfft_cont对象调用incree_pixon_min。似乎是对Pixon的某种调整。
void PixonCont::increase_ipixon_cont()
{
  int i;
  pfft_cont.increase_pixon_min();
  ipixon_cont++;
}
%这段代码涉及到使用 nlopt 和 tnc 这两个库进行数值优化。总体来说，这些函数的目的是为了在给定参数下，计算某个目标函数（卡方和内存的总和），并提供梯度信息以便进行数值优化。
%这是一个用于 nlopt 优化的函数。
%参数包括一个向量 x，一个梯度向量 grad，以及一个指向 PixonCont 类型的数据结构的指针 f_data。
%在函数内部，通过将 f_data 转换为 PixonCont 对象 pixon，调用 compute_cont 函数计算一些值，然后判断是否需要计算梯度。
%如果需要，就分别调用 compute_chisquare_grad_cont 和 compute_mem_grad_cont 计算卡方和内存的梯度。
%最后，将梯度值存入 grad 向量，计算并返回卡方和内存的总和。
/* function for nlopt */
double func_nlopt_cont(const vector<double> &x, vector<double> &grad, void *f_data)
{
  PixonCont *pixon = (PixonCont *)f_data;
  double chisq, mem;

  pixon->compute_cont(x.data());
  if (!grad.empty()) 
  {
    int i;
    
    pixon->compute_chisquare_grad_cont(x.data());
    pixon->compute_mem_grad_cont(x.data());
    
    for(i=0; i<(int)grad.size(); i++)
      grad[i] = pixon->grad_chisq_cont[i] + pixon->grad_mem_cont[i];
  }
  chisq = pixon->compute_chisquare_cont(x.data());
  mem = pixon->compute_mem_cont(x.data());
  return chisq + mem;
}

%这是一个用于 tnc 优化的函数。
%参数包括一个数组 x，一个指向 double 的指针 f，一个数组 g，以及一个指向 PixonCont 类型的数据结构的指针 state。
%在函数内部，通过将 state 转换为 PixonCont 对象 pixon，
%调用 compute_cont 函数计算一些值，然后计算卡方和内存的梯度。
%最后，将卡方和内存的总和存入 f 指针，将梯度值存入 g 数组，返回 0。
/* function for tnc */
int func_tnc_cont(double x[], double *f, double g[], void *state)
{
  PixonCont *pixon = (PixonCont *)state;
  int i;
  double chisq, mem;

  pixon->compute_cont(x);
  pixon->compute_chisquare_grad_cont(x);
  pixon->compute_mem_grad_cont(x);
  
  chisq = pixon->compute_chisquare_cont(x);
  mem = pixon->compute_mem_cont(x);

  *f = chisq + mem;

  for(i=0; i<pixon->cont.size; i++)
    g[i] = pixon->grad_chisq_cont[i] + pixon->grad_mem_cont[i];

  return 0;
}
%这是另一个用于 nlopt 优化的函数，与第一个函数类似，但是使用了 compute_rm_pixon 函数进行计算。
%同样，根据是否需要计算梯度，分别调用 compute_chisquare_grad 和 compute_mem_grad 计算卡方和内存的梯度。
%最后，将梯度值存入 grad 向量，计算并返回卡方和内存的总和。
/* function for nlopt */
double func_nlopt_cont_rm(const vector<double> &x, vector<double> &grad, void *f_data)
{
  PixonCont *pixon = (PixonCont *)f_data;
  double chisq, mem;

  pixon->compute_rm_pixon(x.data());
  if (!grad.empty()) 
  {
    int i;
    
    pixon->compute_chisquare_grad(x.data());
    pixon->compute_mem_grad(x.data());
    
    for(i=0; i<pixon->npixel+1; i++)
      grad[i] = pixon->grad_chisq[i] + pixon->grad_mem[i];

    for(i=pixon->npixel+1; i<(int)grad.size(); i++)
      grad[i] = pixon->grad_chisq_cont[i-pixon->npixel-1] + pixon->grad_mem_cont[i-pixon->npixel-1];
  }
  chisq = pixon->compute_chisquare(x.data());
  mem = pixon->compute_mem(x.data());
  return chisq + mem;
}

%这段代码是一个用于 TNC 优化算法的函数。
%这是一个 TNC 优化算法的函数，接受参数包括一个数组 x，一个指向 double 的指针 f，一个数组 g，以及一个指向 PixonCont 类型的数据结构的指针 state。
%将传入的 state 转换为 PixonCont 对象的指针，这里使用了类型转换 (PixonCont *)。
%调用 PixonCont 对象的 compute_rm_pixon 函数，计算一些与参数 x 相关的值。
%调用 PixonCont 对象的 compute_chisquare_grad 函数，计算卡方的梯度。
%调用 PixonCont 对象的 compute_mem_grad 函数，计算内存的梯度。
%调用 PixonCont 对象的 compute_chisquare 函数，计算卡方。
%将卡方和内存的梯度值存入 g 数组的前 npixel+1 个位置。
%将卡方和内存在 cont 部分的梯度值存入 g 数组的剩余位置。
%函数返回 0，表示成功执行。这是 TNC 优化算法的要求，返回值为 0 表示成功完成。
/* function for tnc */
int func_tnc_cont_rm(double x[], double *f, double g[], void *state)
{
  PixonCont *pixon = (PixonCont *)state;
  int i;
  double chisq, mem;

  pixon->compute_rm_pixon(x);
  pixon->compute_chisquare_grad(x);
  pixon->compute_mem_grad(x);
  
  chisq = pixon->compute_chisquare(x);
  mem = pixon->compute_mem(x);

  *f = chisq + mem;

  for(i=0; i<pixon->npixel+1; i++)
    g[i] = pixon->grad_chisq[i] + pixon->grad_mem[i];

  for(i=pixon->npixel+1; i<pixon->cont.size + pixon->npixel+1; i++)
    g[i] = pixon->grad_chisq_cont[i - pixon->npixel - 1] + pixon->grad_mem_cont[i - pixon->npixel - 1];

  return 0;
}
