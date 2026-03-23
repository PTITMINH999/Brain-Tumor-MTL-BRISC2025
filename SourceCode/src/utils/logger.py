from pathlib import Path 
import sys
import os 

class Logger(object):
    def __init__(self,filename="results/training_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename,"w",encoding='utf-8')
    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        