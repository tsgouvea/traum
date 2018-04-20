import os

import pandas as pd
import numpy as np

from mdaio import readmda

class traum:

    def __init__(self,dataBhv):
        self.bhv = dataBhv.parsedData
        self.bhv['tsState0'] = self.bhv['tsState0']-self.bhv['tsState0'][0]
        self.neur = pd.DataFrame({'spikes': [], 'dataset':[], 'cluster':[]})#, 'waveform_mean':[], 'waveform_std':[], 'nTrode':[], 'cluster':[]})

    def readDio(self,pathDio,fs=30000):
        listDio = os.listdir(pathDio)
        listDio.sort()
        df_singleChanges = pd.DataFrame({'channel': [], 'state': [], 'time': []})
        from readTrodesExtractedDataFile import readTrodesExtractedDataFile as readDat
        for iDio in range(len(listDio)):
            x = readDat(os.path.join(pathDio,listDio[iDio]))
            df_temp = pd.DataFrame(x['data'])
            df_temp['channel'] = iDio
            df_singleChanges = df_singleChanges.append(df_temp)

        timeSet = sorted(set(df_singleChanges['time']))
        t0 = timeSet[0]
        timeSet.remove(t0)
        time = [t0-t0]
        stateBin = ['0'*len(set(df_singleChanges['channel']))]
        state = [int(stateBin[-1], 2)]

        for t in timeSet:
            df_t = df_singleChanges[df_singleChanges['time'] == t]
            sbin = list(stateBin[-1][::-1])
            listStates = df_t['state']
            for c in df_t['channel']:
                sbin[int(c)] = str(int(listStates[c==df_t['channel']]))
            time.append(t-t0)
            stateBin.append(''.join(sbin)[::-1])
            state.append(int(stateBin[-1], 2))

        self.dio = pd.DataFrame({'time':time, 'state':state, 'stateBin':stateBin})
        self.dio['time'] = self.dio['time']/fs
        self.sync()

    def sync(self):

        def trim(lon,sho):
            delta = len(lon)-len(sho)
            rhos = np.full(delta+1,np.nan)
            for i in range(len(rhos)):
                rhos[i] = np.corrcoef(np.diff(sho),np.diff(lon[i:len(sho)+i]))[0,1]
            assert(np.max(rhos)>.99)
            return(np.argmax(rhos), delta, rhos)

        tsDio = self.dio['time'][self.dio['state']==0]
        tsBhv = self.bhv['tsState0']-self.bhv['tsState0'][0]

        if len(tsBhv) > len(tsDio):
            #print('Off sync - len(tsBhv) > len(tsDio)')
            i, delta, rhos = trim(tsBhv,tsDio)
            self.bhv = self.bhv[i:len(tsDio)+i]
        elif len(tsDio) > len(tsBhv):
            #print('Off sync - len(tsDio) > len(tsBhv)')
            i, delta, rhos = trim(tsDio,tsBhv)
            self.dio = self.dio[(self.dio['time'] < tsDio.iloc[i+len(tsBhv)]) & (self.dio['time'] >= tsDio.iloc[i])]
        self.aligEvent = list(self.bhv.columns[[n.startswith('ts') for n in self.bhv.columns]])
        self.aligEvent.remove('tsState0')
        #self.tsDio = self.dio['time'][self.dio['state']==0]
        #self.tsBhv = self.bhv['tsState0']-self.bhv['tsState0'][0]

    def readNeur(self,pathNeur,prefix='ms3',filename='firings.curated.mda',fs=30000):

        listNt = np.array(os.listdir(pathNeur))
        listNt = listNt[[n.startswith(prefix) for n in listNt]]
        assert(len(listNt)>0)
        for nt in listNt:
            mdaCurated = readmda(os.path.join(pathNeur,nt,filename))
            setClust = list(set(mdaCurated[2,:]))
            setClust.sort()
            df_spikes = [[]]*len(setClust)
            df_cluster = np.empty(len(setClust),int)
            for i in range(len(setClust)):
                ndx = mdaCurated[2,:] == setClust[i]
                df_spikes[i] = mdaCurated[1,ndx]
                df_cluster[i] = int(setClust[i])
            self.neur = self.neur.append(pd.DataFrame({'spikes': np.array(df_spikes)/fs, 'cluster':df_cluster, 'dataset': nt}),ignore_index=True)

    def raspeth(self,alignment,iUnit,trialMask,panes,window=(-1,2)):
        listAlign = np.array(self.dio['time'][self.dio['state']==0][trialMask]) + np.array(self.bhv[alignment][trialMask])
        spikes = [[]]*len(listAlign)

        for iTrial in range(len(listAlign)):
            temp = self.neur['spikes'][iUnit] - listAlign[iTrial]
            spikes[iTrial] = temp[(temp>=window[0]) & (temp<=window[1])]

        ha_raster, ha_peth = panes
        ha_raster.eventplot(spikes)

        counts, edges = np.histogram([item for sublist in spikes for item in sublist], bins='auto', density=False)
        ha_peth.plot(edges[:-1]+(edges[1]-edges[0])/2,counts)
        return counts,edges
            #self.neur = self.neur[i:-delta]
