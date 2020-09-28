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

con = np.loadtxt(basename(config["dump"]["fcon"]))
line = np.loadtxt(basename(config["dump"]["fline"]))
offset = np.mean(line[:, 1]) - np.std(line[:, 1]) - (np.mean(con[:, 1]) - np.std(con[:, 1]))
tlim1 = np.min((con[0, 0], line[0, 0]))
tlim2 = np.max((con[-1,0], line[-1, 0]))
tspan=tlim2-tlim1
resp_input = np.loadtxt("resp_input.txt")

fnames = ["", "_cont", "_drw"]
postfix = config["dump"]["pixon_type"]

fig = plt.figure(figsize=(15, 15))

for i, fn in enumerate(fnames):

  ax = fig.add_axes((0.1, 0.7-i*0.25, 0.5, 0.25))
  ax.errorbar(con[:, 0], con[:, 1], yerr = con[:, 2], ls='none', marker='o', markersize=3, zorder=0)
  ax.errorbar(line[:, 0], line[:, 1]-0.7*offset, yerr = line[:, 2], ls='none', marker='o', markersize=3, zorder=0)
  if i==0:
   con_rec = np.loadtxt("con_recon.txt")
   con_rec_uniform = con_rec
   line_rec = np.loadtxt("line_rec_full.txt_"+postfix)
   line_rec_uniform = np.loadtxt("line_rec_full_uniform.txt_"+postfix)
   
  elif i==1:
   con_rec = np.loadtxt("con_pixon_rm.txt_"+postfix)
   con_rec_uniform = np.loadtxt("con_pixon_rm_uniform.txt_"+postfix)
   line_rec = np.loadtxt("line_rec_full_cont.txt_"+postfix)
   line_rec_uniform = np.loadtxt("line_rec_full_cont_uniform.txt_"+postfix)
   
  else:
   con_rec = np.loadtxt("con_drw_rm.txt_"+postfix)
   con_rec_uniform = np.loadtxt("con_drw_rm_uniform.txt_"+postfix)
   line_rec = np.loadtxt("line_rec_full_drw.txt_"+postfix)
   line_rec_uniform = np.loadtxt("line_rec_full_drw_uniform.txt_"+postfix)
  
  ax.plot(con_rec[:, 0], con_rec[:, 1], lw=1, color='r',)
  ax.plot(con_rec_uniform[:, 0], con_rec_uniform[:, 1],  lw=1, color='b')
  
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

