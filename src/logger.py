from datetime import datetime
import os, sys
os.chdir(sys.path[0])

def get_timestamp(for_file=False):
  # Returns a timestamp
  # File name friendly format if for_file
  if for_file:
    return "-".join(str(datetime.now()).split(".")[0].split(":")).replace(" ", "_")
  else:
    return str(datetime.now())

class Logger:

  def __init__(self, fpath: str, title: str, include_time_stamp=False):
    dirs = "".join(fpath.split('/')[:-1])
    if not os.path.exists(dirs):
      os.makedirs(dirs)

    if include_time_stamp:
      fpath += get_timestamp(True)
    self.fpath = fpath
    
    with open(self.fpath, "w+", encoding="utf-8") as logfile:
      logfile.write("")
    
    self.log(title + "\n")
  
  def log(self, *tolog):

    time = get_timestamp()
    with open(self.fpath, "a") as logfile:
      tolog = " ".join([str(x) for x in tolog])
      n_spaces = len(time)
      logs = tolog.split("\n")
      logs[0] = time + "\t\t" + logs[0]
      for i in range(1, len(logs)):
        logs[i] = n_spaces * " " + "\t\t" + logs[i]
      tolog = "\n".join(logs)
      logfile.write(tolog+"\n")
      print(tolog)
