import os

import pandas as pd
import numpy as np

from traum.mdaio import readmda

class traum:

    def __init__(self,dataBhv='none'):
        self.neur = pd.DataFrame({'spikes': [], 'dataset':[], 'cluster':[]})#, 'waveform_mean':[], 'waveform_std':[], 'nTrode':[], 'cluster':[]})
        if dataBhv=='none':
            pass
        else:
            self.readBhv(dataBhv)

    def readBhv(self,dataBhv):
        self.bhv = dataBhv
        self.bhv.parsedData['tsState0'] = self.bhv.parsedData['tsState0']-self.bhv.parsedData['tsState0'][0]

    def readDio(self,pathDio,fs=30000):
        from .readTrodesExtractedDataFile3 import readTrodesExtractedDataFile3 as readDat

        listDio = np.array(os.listdir(pathDio))
        listDio = listDio[['Din' in n for n in listDio]]
        listDio.sort()
        df_singleChanges = pd.DataFrame({'channel': [], 'state': [], 'time': []})

        for iDio in range(len(listDio)):
            x = readDat(os.path.join(pathDio,listDio[iDio]))
            df_temp = pd.DataFrame(x['data'])
            df_temp['channel'] = iDio
            df_singleChanges = df_singleChanges.append(df_temp,sort=True)

        df_singleChanges.channel = df_singleChanges.channel.astype(int)
        df_singleChanges.state = df_singleChanges.state.astype(int)
        timeSet = sorted(set(df_singleChanges['time']))
        time = []
        stateBin = []
        state = []

        for t in timeSet:
            df_t = df_singleChanges[df_singleChanges['time'] == t]
            sbin = list(stateBin[-1]) if stateBin else list([['X']*len(set(df_singleChanges['channel']))][0])
            for c in df_t['channel']:
                sbin[c] = df_t[df_t['channel']==c]['state'].values.astype(str).item()
            assert('X' not in sbin), "Initial state of some channels not defined: " + ''.join(sbin)
            time.append(t)
            stateBin.append(''.join(sbin))
            state.append(int(stateBin[-1], 2))

        self.dio = pd.DataFrame({'time':np.array(time)/fs, 'state':state, 'stateBin':stateBin, 'iTrial':np.cumsum(np.array(state)==0)-1})
        self.dio = self.dio.set_index('iTrial')

    def readNeur(self,pathNeur,prefix='ms3',filename='firings.curated.mda',fs=30000):
        listNt = np.array(os.listdir(pathNeur))
        listNt = listNt[[n.startswith(prefix) for n in listNt]]
        listNt.sort()
        assert(len(listNt)>0)
        for nt in listNt:
            if not os.path.isfile(os.path.join(pathNeur,nt,filename)):
                continue
            mdaCurated = readmda(os.path.join(pathNeur,nt,filename))
            setClust = list(set(mdaCurated[2,:]))
            setClust.sort()
            df_spikes = [[]]*len(setClust)
            df_cluster = np.empty(len(setClust),np.int)
            for i in range(len(setClust)):
                ndx = mdaCurated[2,:] == setClust[i]
                df_spikes[i] = mdaCurated[1,ndx]
                df_cluster[i] = setClust[i]

            df_spikes = [df_spikes[0]/fs] if len(setClust)==1 else np.array(df_spikes)/fs

            self.neur = self.neur.append(pd.DataFrame({'spikes': df_spikes, 'cluster':df_cluster, 'dataset': nt}),ignore_index=True,sort=False)
        self.neur.cluster = self.neur.cluster.astype(int)

    def sync(self):

        def trim(lon,sho):
            delta = len(lon)-len(sho)
            rhos = np.full(delta+1,np.nan)
            for i in range(len(rhos)):
                rhos[i] = np.corrcoef(np.diff(sho),np.diff(lon[i:len(sho)+i]))[0,1]
            assert(np.max(rhos)>.99)
            return(np.argmax(rhos), delta, rhos)

        tsDio = self.dio['time'][self.dio['state']==1]
        tsBhv = self.bhv.parsedData['tsState0']-self.bhv.parsedData['tsState0'][0]

        if len(tsBhv) > len(tsDio):
            i, delta, rhos = trim(tsBhv,tsDio)
            self.bhv.parsedData = self.bhv.parsedData[i:len(tsDio)+i]
        elif len(tsDio) > len(tsBhv):
            i, delta, rhos = trim(tsDio,tsBhv)
            self.dio = self.dio[(self.dio['time'] < tsDio.iloc[i+len(tsBhv)]) & (self.dio['time'] >= tsDio.iloc[i])]
        else:
            i, delta, rhos = trim(tsDio,tsBhv)
        print('at traum.sync():\nCorrelation coefficients for all alignments:')
        print(rhos)
        self.aligEvent = list(self.bhv.parsedData.columns[[n.startswith('ts') for n in self.bhv.parsedData.columns]])
        self.aligEvent.remove('tsState0')

    def raspeth(self,alignment,iUnit,trialMask,panes,window=(-1,2),bins='rice',conv='None'):
        ha_raster, ha_peth = panes
        ndxType, colors = trialMask # ndxType {0, 1, 2, ...} is trial type, trials discarded where 0 // len(colors) = len(set(ndx[ndx>0]))
        setType = list(set(ndxType[ndxType>0]))
        offset = 0
        listAlign = np.array(self.dio['time'][self.dio['state']==1]) + np.array(self.bhv.parsedData[alignment])
        if type(bins)==int:
            bins = np.linspace(window[0],window[1],bins)
        for iType in setType:
            ndx = (ndxType == iType) & (np.logical_not(np.isnan(listAlign)))
            if any(ndx):
                spikes = [[]]*sum(ndx)
                i = 0
                for iTrial in np.nonzero(ndx)[0]:
                    temp = self.neur['spikes'][iUnit] - listAlign[iTrial]
                    spikes[i] = temp[(temp>=window[0]) & (temp<=window[1])]
                    i += 1

                ha_raster.eventplot([[]]*offset + spikes,colors=colors[iType-1])
                offset += sum(ndx)

                counts, edges = np.histogram([item for sublist in spikes for item in sublist], bins=bins, density=False)
                if (type(conv)!=str) :# or (conv != 'None') or (conv != 'none'):
                    counts = np.convolve(counts,conv,mode='same')
                ha_peth.plot(edges[:-1]+(edges[1]-edges[0])/2,counts/(edges[1]-edges[0])/sum(ndx),color=colors[iType-1])
        ha_raster.set_xlim(window)
        ha_peth.set_xlim(window)
        """ha_peth.set_ylabel('Firing rate')
        ha_peth.set_xlabel('Time (s)')
        ha_raster.set_ylabel('Trial #')
        ha_raster.set_title(alignment)"""
        #return ha_raster, ha_peth
        return spikes
