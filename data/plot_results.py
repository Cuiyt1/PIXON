import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages
import configparser as cfgpars
from os.path import basename

#
# load params
# 
with open("param_input") as f:
  file_content = '[dump]\n' + f.read()
  
config = cfgpars.ConfigParser(delimiters='=', allow_no_value=True)
config.read_string(file_content)

for key in config["dump"].keys():
  print(key, config["dump"][key])

plt.rc("text", usetex=True)
plt.rc('font', family='serif', size=15)

cont = np.loadtxt(basename(config["dump"]["fcont"]))
line = np.loadtxt(basename(config["dump"]["fline"]))
offset = np.mean(line[:, 1]) - np.std(line[:, 1]) - (np.mean(cont[:, 1]) - np.std(cont[:, 1]))
tlim1 = np.min((cont[0, 0], line[0, 0]))
tlim2 = np.max((cont[-1,0], line[-1, 0]))
tspan=tlim2-tlim1
#resp_input = np.loadtxt("resp_input.txt")

fnames = ["_contfix", "_pixon", "_drw"]
postfix = config["dump"]["pixon_basis_type"]

fig = plt.figure(figsize=(15, 15))

for i, fn in enumerate(fnames):

  ax = fig.add_axes((0.1, 0.7-i*0.25, 0.5, 0.25))
  ax.errorbar(cont[:, 0], cont[:, 1], yerr = cont[:, 2], ls='none', marker='o', markersize=3, zorder=0)
  ax.errorbar(line[:, 0], line[:, 1]-0.7*offset, yerr = line[:, 2], ls='none', marker='o', markersize=3, zorder=0)
  if i==0:
   cont_rec = np.loadtxt("cont_recon_drw.txt")
   cont_rec_uniform = cont_rec
   line_rec = np.loadtxt("line_contfix_full.txt_"+postfix)
   line_rec_uniform = np.loadtxt("line_contfix_uniform_full.txt_"+postfix)
   
  elif i==1:
   cont_rec = np.loadtxt("cont_pixon.txt_"+postfix)
   cont_rec_uniform = np.loadtxt("cont_pixon_uniform.txt_"+postfix)
   line_rec = np.loadtxt("line_pixon_full.txt_"+postfix)
   line_rec_uniform = np.loadtxt("line_pixon_uniform_full.txt_"+postfix)
   
  else:
   cont_rec = np.loadtxt("cont_drw.txt_"+postfix)
   cont_rec_uniform = np.loadtxt("cont_drw_uniform.txt_"+postfix)
   line_rec = np.loadtxt("line_drw_full.txt_"+postfix)
   line_rec_uniform = np.loadtxt("line_drw_uniform_full.txt_"+postfix)
  
  ax.plot(cont_rec[:, 0], cont_rec[:, 1], lw=1, color='r',)
  ax.plot(cont_rec_uniform[:, 0], cont_rec_uniform[:, 1],  lw=1, color='b')
  
  idx = np.where( (line_rec[:, 0] > line[0, 0] - 10) & (line_rec[:, 0] < line[-1, 0] + 10))
  ax.plot(line_rec[idx[0], 0], line_rec[idx[0], 1]-0.7*offset, lw=1, color='r')
  ax.plot(line_rec_uniform[idx[0], 0], line_rec_uniform[idx[0], 1]-0.7*offset,  lw=1, color='b')
  
  ax.set_xlim(tlim1-0.01*tspan, tlim2+0.01*tspan)
   
  ax = fig.add_axes((0.64, 0.7-i*0.25, 0.3, 0.25))
  resp = np.loadtxt("resp"+fn+".txt_"+postfix)
  resp_uniform = np.loadtxt("resp"+fn+"_uniform.txt_"+postfix)
  #ax.plot(resp_input[:, 0], resp_input[:, 1], lw=2, color='k', label='Truth')
  ax.plot(resp[:, 0], resp[:, 1], lw=1, label='pixel', color='r')
  ax.plot(resp_uniform[:, 0], resp_uniform[:, 1], lw=1, label='uniform', color='b')
  ax.legend(frameon=False)

plt.show()

fname = basename(config["dump"]["fline"])
fname.replace(".txt", "")
fname = fname + ".pdf"
fig.savefig(fname, bbox_inches='tight')

