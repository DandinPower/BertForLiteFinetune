import torch 
import sys 
import os
import psutil
import time

class IOCounter:
    def __init__(self):
        self.ReadData = []
        self.WriteData = []
        self.process = psutil.Process(os.getpid())
        
    def AddRead(self):
        self.ReadData.append(self.process.io_counters()[2])

    def AddWrite(self):
        self.WriteData.append(self.process.io_counters()[3])

    def GetRead(self,a,b):
        value = self.ReadData[b] - self.ReadData[a]
        self.ShowBytes(value)

    def GetWrite(self,a,b):
        value = self.WriteData[b] - self.WriteData[a]
        self.ShowBytes(value)
    
    def ShowBytes(self,byte):
        print(f'{byte}bytes, {byte/1024}kbs, {byte/(1024*1024)}mbs')
    
    def Clear(self):
        self.ReadData.clear()
        self.WriteData.clear()

class MemoryCounter:
    def __init__(self):
        self.data = []
        self.process = psutil.Process(os.getpid())
    
    def Add(self):
        self.data.append(self.process.memory_info()[0])

    def GetData(self,a,b):
        value = self.data[b] - self.data[a]
        self.ShowBytes(value)
        
    def ShowBytes(self,byte):
        print(f'{byte}bytes, {byte/1024}kbs, {byte/(1024*1024)}mbs')
    
    def Clear(self):
        self.data.clear()