
# coding: utf-8

# Strated this code cause it seemed there was some clock drift apparent even after aligning spike trains to state0 DIO (i.e. spike_times{i} = spike_times - state0_dio{i} + (alignEvent{i} - state0_bpod{i}). Drift appeard in the form of a peak with a constant time offset of a few ms over consecutive trials. That was observed before a bug with subtracting the first element of state0_dio from the entire vector, but not of the corresponding spike times - and hasn't been obvserved after, so I'm ignoring it for now.

# In[ ]:


import os
import re
import sys

import numpy as np
import matplotlib as mp
import pandas as pd
import scipy as sp
import scipy.stats as spt
import scipy.io as sio
import matplotlib.pyplot as plt

from tasks import dual2afc
from traum import traum as tr
#from traum.mdaio import readmda
#from traum import readDIO
#from traum.readTrodesExtractedDataFile import readTrodesExtractedDataFile as readdat

print(sys.version)


# ## Behavior

# In[ ]:


subjName = 'M14'
sessName = 'M14_20171117_175330'

pathBhv = os.path.join('datasets/dual2afc_ds1/','bhv',subjName,'M14_Dual2AFC_Nov17_2017_Session1.mat')
dataBhv = dual2afc.parser(sio.loadmat(pathBhv, squeeze_me=True))


# In[ ]:


obj = tr.traum(dataBhv)


# ## Neural

# In[ ]:


pathNeur = os.path.join('datasets/dual2afc_ds1/','neur',subjName,sessName)

obj.readNeur(pathNeur)


# ## Sync

# In[ ]:


pathDIO = os.path.join('datasets/dual2afc_ds1/','neur',subjName,sessName,'dio')
obj.readDio(pathDIO)

obj.sync()


# ## Positive control

# In[6]:


obj.neur = obj.neur.append({'cluster': 0, 'dataset': 'water_', 'spikes': obj.dio['time'][(obj.dio['state']==20) | (obj.dio['state']==21)].values},ignore_index=True)
# ## Traum

# In[ ]:


#ndxType = np.full(len(obj.bhv.parsedData),0)

corr_ndxType = np.full(len(obj.bhv.parsedData),0)
corr_ndxType[obj.bhv.parsedData['isCorrect']] = 1
corr_ndxType[obj.bhv.parsedData['isIncorr']] = 2
corr_colors = ['xkcd:grass green', 'xkcd:scarlet']

cho_ndxType = np.full(len(obj.bhv.parsedData),0)
cho_ndxType[obj.bhv.parsedData['isChoiceLeft']] = 1
cho_ndxType[obj.bhv.parsedData['isChoiceRight']] = 2
cho_colors =['xkcd:golden yellow', 'xkcd:dark sky blue']

stim_ndxType = np.full(len(obj.bhv.parsedData),0)
setStim = list(set(obj.bhv.parsedData['OdorFracA']))
for iStim in range(len(setStim)):
    stim_ndxType[obj.bhv.parsedData['OdorFracA'] == setStim[iStim]] = iStim+1
stim_colors = ['xkcd:bright blue', 'xkcd:deep sky blue', 'xkcd:bright sky blue', 'xkcd:aqua', 'xkcd:aquamarine', 'xkcd:shamrock green', 'xkcd:kermit green']

corr_trialMask = (corr_ndxType, corr_colors)
cho_trialMask = (cho_ndxType, cho_colors)
stim_trialMask = (stim_ndxType, stim_colors)

trialMask = (stim_trialMask,cho_trialMask,corr_trialMask)


# In[ ]:


listAlign = ('tsCin','tsStimOn','tsStimOff','tsChoice','tsRwd','tsErrTone')


# In[ ]:


window = [-1,2]
bins = np.arange(window[0],window[1],.005)
conv = spt.norm.pdf(np.arange(-30,30),0,10)
conv = conv/sum(conv)
pathFigs = os.path.join('/Users','thiago','Pictures','Traum',subjName)


# In[10]:


for iUnit in range(len(obj.neur)):
    figTitle = sessName + '_unit' + str(iUnit).zfill(len(str(len(trialMask))))
    hf, ha = plt.subplots(len(trialMask)*2,len(listAlign),figsize=(20,10))

    for iMask in range(len(trialMask)):
        for iAlign in range(len(listAlign)) :
            h = (ha[iMask*2,iAlign], ha[iMask*2+1,iAlign])# + '\t' + str() + str(iAlign) + '\n')
            obj.raspeth(listAlign[iAlign],iUnit,trialMask[iMask],h,bins=bins,conv=conv)

            if (iMask == 0) :
                ha[iMask*2,iAlign].set_title(listAlign[iAlign])

            if (iAlign == 0) :
                if (iMask == 0) :
                    ha[iMask*2,iAlign].set_ylabel('Trial #')
                    ha[iMask*2+1,iAlign].set_ylabel('Firing rate')
                elif (iMask == len(list(trialMask))) :
                    ha[iMask*2+1,iAlign].set_xlabel('Time (s)')

    #plt.suptitle(figTitle,fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(pathFigs,figTitle + '_bin5ms_sd10bins.pdf'))#,dpi=150,orientation='landscape',papertype='letter',format='eps')
    plt.close(hf)

