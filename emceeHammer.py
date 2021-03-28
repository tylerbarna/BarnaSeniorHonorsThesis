from useful_functions import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import emcee
import corner
from multiprocessing import Pool
##Need to figure out why this stuff is needed
tess_2020bpi = pd.read_csv('JhaData/TESS_SN2020bpi.csv')[::2]
tess_2020bpi['mjd_0'] = tess_2020bpi['mjd'] - tess_2020bpi['mjd'].min()
fluxNorm = 0.4*np.max(tess_2020bpi['flux'])
# fluxNorm = 0.4*np.max(tess_2020bpi['flux'])
# tess_2020bpi_norm = tess_2020bpi
# tess_2020bpi_norm.flux = tess_2020bpi.flux/fluxNorm
# tess_2020bpi_norm.e_flux = tess_2020bpi.e_flux/fluxNorm
# tess_2020bpi_a = pd.read_csv('JhaData/TESS_SN2020bpi_updated.csv')[::2]
# tess_2020bpi_a['mjd_0'] = tess_2020bpi_a['mjd'] - tess_2020bpi['mjd'].min()
# tess_2020bpi_a_norm = tess_2020bpi_a
# tess_2020bpi_a_norm.flux = tess_2020bpi_a.flux/fluxNorm
# tess_2020bpi_a_norm.e_flux = tess_2020bpi_a.e_flux/fluxNorm'

def normLC(lcDF,fluxNorm=fluxNorm):
    ## normalizes lc to arbitrary value or default of
    ## 40% of original lightcurve
    normFrame = lcDF.copy()
    normFrame['mjd_0'] = normFrame['mjd'] - tess_2020bpi['mjd'].min()
    normFrame['flux'] = lcDF['flux']/fluxNorm
    normFrame['e_flux'] = lcDF['e_flux']/fluxNorm
    if 'raw_flux' in normFrame.columns:
        normFrame['raw_flux'] = lcDF['raw_flux']/fluxNorm
        normFrame['e_raw_flux'] = lcDF['e_raw_flux']/fluxNorm
#     if 'e_flux_tuple' in normFrame.columns:
#         normFrame['e_flux_tuple'] = lcDF['e_flux_tuple']/fluxNorm
    return normFrame
def get_fullparam(theta, thetaKeys):
    params = {'t0':7, 
              'a':1,
              'sigma':0.0,
              'power':1.8,
              'y0':0,
              'mean':10.65,
              'std':1,
              'gFactor':0.0,
              'bkg mod':0}
    params.update(dict(zip(thetaKeys,theta)))
    return params

def lc_model(theta,thetaKeys, data, ztfData=None,curveModel='standard',cutoff=16.75):
    thetaDict = get_fullparam(theta,thetaKeys)
    
    if curveModel =='standard':
        var = (data['e_flux'].to_numpy()**2 + thetaDict['sigma']**2)
        model = np.array([0 if t <= thetaDict['t0'] else
                 thetaDict['a'] * (t - thetaDict['t0'])**thetaDict['power'] 
                 for t in data['mjd_0'].to_numpy()])
    if curveModel =='standardPlus':
        var = (data['e_flux'].to_numpy()**2 + thetaDict['sigma']**2)
        model = np.array([thetaDict['y0'] if t <= thetaDict['t0'] else
                 thetaDict['a'] * (t - thetaDict['t0'])**thetaDict['power'] + thetaDict['y0']
                 for t in data['mjd_0'].to_numpy()])
        
    elif curveModel =='powerFix1':
        var = (data['e_flux'].to_numpy()**2 + thetaDict['sigma']**2)
        model = np.array([0 if t <= thetaDict['t0'] else
                 thetaDict['a']**thetaDict['power']  * (t - thetaDict['t0'])**thetaDict['power'] 
                 for t in data['mjd_0'].to_numpy()])
        
    elif curveModel =='powerFix2':
        var = (data['e_flux'].to_numpy()**2 + thetaDict['sigma']**2)
        model = np.array([0 if t <= thetaDict['t0'] else
                 thetaDict['a']**(1/thetaDict['power'])  * (t - thetaDict['t0'])**thetaDict['power'] 
                 for t in data['mjd_0'].to_numpy()])
        
    elif curveModel =='gaussian':
        var = (data['e_flux'].to_numpy()**2 + thetaDict['sigma']**2)
        model = np.array([0 if t <= thetaDict['t0'] else
                 (thetaDict['a'] * (t - thetaDict['t0'])**thetaDict['power']) +                 
                 (thetaDict['gFactor']* np.exp(-0.5*((t-thetaDict['mean'])/thetaDict['std'])**2))
                 for t in data['mjd_0'].to_numpy()]) 
        
    elif curveModel =='decoupled':
        var = (data['e_flux'].to_numpy()**2 + thetaDict['sigma']**2)
        model = np.array([thetaDict['y0'] if t <= thetaDict['t0'] else
                          ((thetaDict['a']*((t - thetaDict['t0'])/
                                            (cutoff-thetaDict['t0']))**thetaDict['power']) 
                           +thetaDict['y0'])
                          for t in data['mjd_0'].to_numpy()])
    elif curveModel =='dcRaw':
        var = (data['e_raw_flux'].to_numpy()**2 + thetaDict['sigma']**2)
        model = np.array([thetaDict['y0'] if t <= thetaDict['t0'] else
                          ((thetaDict['a']*((t - thetaDict['t0'])/
                                            (cutoff-thetaDict['t0']))**thetaDict['power']) 
                           +thetaDict['y0'])
                          for t in data['mjd_0'].to_numpy()])
        
    elif curveModel =='dcGauss':
        var = (data['e_flux'].to_numpy()**2 + thetaDict['sigma']**2)
        model = np.array([thetaDict['y0'] if t <= thetaDict['t0'] else
                          ((thetaDict['a']*((t - thetaDict['t0'])/
                                            (cutoff-thetaDict['t0']))**thetaDict['power']) 
                           +thetaDict['y0']) +
                          (thetaDict['gFactor']* np.exp(-0.5*((t-thetaDict['mean'])/thetaDict['std'])**2))
                          for t in data['mjd_0'].to_numpy()]) 
    
    else:
        raise KeyError('Must Provide Valid Model')
    return model, var

# def gaussian(theta, thetaKeys,data): 
#     thetaDict = dict(zip(thetaKeys,theta))#get_fullparam(theta,thetaKeys)
#     #curveFrac = (1/(thetaDict['std']*np.sqrt(2*np.pi)))
#     #curveExp = -0.5*((data['mjd_0']-thetaDict['mean'])/thetaDict['std'])**2
#     #curve = thetaDict['gFactor']* np.exp(curveExp)
#     curve = np.array([ 0 if t <= thetaDict['t0'] else
#                       (thetaDict['gFactor']* np.exp(-0.5*((t-thetaDict['mean'])/thetaDict['std'])**2))
#                       for t in data['mjd_0']
#                     ])
#     return curve 

def log_prior(theta, thetaKeys):
    
    nparam = len(theta)
    thetaDict = dict(zip(thetaKeys,theta))#get_fullparam(theta, thetaKeys)
    
    logpr = 0.

    if thetaDict['sigma'] < 0:
        return -np.inf

    if thetaDict['t0'] < 0:
        return -np.inf
    
#     elif thetaDict['t0'] > 7: ##should converge without this
#         STD = 1/ (2* np.sqrt(2*np.log(2))) ## not sure about this
#         arbitraryTestScale = 1e0
#         Mean = 5
#         ## Ideally, I should pull this out and make a generalized gaussian prior function
#         ## for use in other priors 
#         logpr += -arbitraryTestScale * 0.5 * (np.log(2 * np.pi * STD) + 
#                             ((Mean - thetaDict['std'])**2 / STD) )
    elif thetaDict['t0'] > 16:
        return -np.inf
    
    elif thetaDict['t0'] >9: ##more just for ztf so it doesn't look so odd
        logpr += -(thetaDict['t0'])**(1/4)
    
    if thetaDict['a'] <= 0.01:
        return -np.inf
    
    elif thetaDict['a'] > 10:
        return -np.inf
    ## don't seem to be needed
#     if thetaDict['y0'] > .5:
#         return -np.inf
    
    if thetaDict['y0'] < -1.5:
        return -np.inf
    
#     if (thetaDict['mean']-2*thetaDict['std']*np.sqrt(2*np.log(2))) <= thetaDict['t0']:
#         return -np.inf

    if thetaDict['mean'] < thetaDict['t0']:
        return -np.inf
    
    if thetaDict['mean'] > thetaDict['t0']+10:
        return -np.inf
    
#     elif (thetaDict['mean']-2*thetaDict['sigma']*np.sqrt(2*np.log(2))) > thetaDict['t0'] + 4:
#         return -np.inf ##maybe make this a gaussian prior as well
    
    if thetaDict['std'] < 1:
        return -np.inf
    
    elif thetaDict['std'] > 4:
        return -np.inf
    
#     else:
#         widthSTD = 1#
#         widthMean = 6 ##pick good parameters later
#         ## Ideally, I should pull this out and make a generalized gaussian prior function
#         ## for use in other priors 
#         #arbitraryTestScale * 0.5 * (np.log(2 * np.pi * widthSTD) ## unneeded
#         logpr += -0.5*((widthMean - thetaDict['std']) / widthSTD)**2
    
    if thetaDict['gFactor'] < 0: 
        return -np.inf
    
    elif thetaDict['gFactor'] > 1: 
        return -np.inf
    
    if thetaDict['power'] <= 0: 
        return -np.inf
    
    elif thetaDict['power'] >= 5:
        return -np.inf
    return logpr

def log_likelihood(theta, thetaKeys, data, curveModel,cutoff=16.75):
    #return 0
    #thetaDict = dict(zip(thetaKeys,theta))#get_fullparam(theta, thetaKeys)
    #print(thetaDict)
    model,var = lc_model(theta,thetaKeys,data, curveModel=curveModel,cutoff=cutoff)
    if curveModel == 'dcRaw':
        flux = data['raw_flux'].to_numpy()
    else:
        flux = data['flux'].to_numpy()
    logl = -0.5 * (np.sum(np.log(2 * np.pi * var) + 
                            ((flux - model)**2 / var) ))
        
    return logl

    
def log_posterior(theta, thetaKeys, data, curveModel,cutoff=16.75,debug=False):
    thetaDict = get_fullparam(theta, thetaKeys)
    logpr = log_prior(thetaDict.values(), thetaDict.keys())
    
    if logpr == -np.inf or debug:
        return logpr
    else:
        return logpr + log_likelihood(thetaDict.values(), thetaDict.keys(), data, curveModel=curveModel,cutoff=cutoff)

def doMCMC(data, guess, scale, cutoff=16.75,
           nwalkers=100, nburn=1500, nsteps=3000, 
           curveModel='decoupled',debug=False,
           dataType='TESS',savePlots=None,fileNameExtras='',plotExt='.png'):
    '''
    Takes data which contains mjd and flux data
    and performs an mcmc fit on it
    '''
    ndim = len(guess)
    assert ndim == len(scale)

    starting_guesses = np.swapaxes(list({
        k:(np.random.randn(nwalkers)*scale[k]+v)
        for k,v in guess.items()}.values()),0,1)
#     print(starting_guesses)
    print('sampling...')
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, threads=-1, 
                                    args=[list(guess.keys()),data,curveModel,cutoff,debug])
    sampler.run_mcmc(starting_guesses, nsteps,progress="notebook")
    print('done')
    
    
    samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))
    sampler.reset()
    tlabels = list(guess.keys())
    figcorner = corner.corner(samples, labels=tlabels[0:ndim],
                    show_titles=True, title_fmt=".6f", verbose=True,
                    title_kwargs={"fontsize": 10}, label_kwargs={"fontsize": 12})
    if savePlots:
        dirString = np.str('./plots/lcFit/'+np.str(dataType)+'/corner/')
        mkdir(dirString)
        fileName = np.str('mod-'+np.str(curveModel)+'-nparam-'+np.str(len(guess)+1)+'-nwalk-'+np.str(nwalkers)+'-nstep-'+np.str(nsteps))
        figcorner.savefig(dirString+fileName+plotExt)
        

    return samples

def hammerTime(data, guess, scale, cutoff=16.75, 
                    nwalkers=100, nburn=1500, nsteps=3000,
                    curveModel='decoupled',numModels=None,
                   MC=True, debug=False,
               plotPal=('#003C86','#7CFFFA','#008169'),
               dataType='TESS',savePlots=None,fileNameExtras='',plotExt='.pdf'):
    '''
    Fits model to TESS or ZTF data using emcee.  Plots model on top of data and residuals an
    optionally saves the plot and fits/parameters.
    
    data: Pandas dataFrame with 'flux', 'e_flux', and 'mjd_0' columns. Some models
    require 'raw_flux' and 'e_raw_flux' as well, but most do not.
    
    guess: dictionary with all parameters that one wants to test. Any parameters used in the chosen
    model that aren't included will be held constant. However, adding terms not used in the model will
    result in unconstrained behavior
    
    '''
    ## pretty sloppy implementation, could make this a part of the function directly
    
     
    if debug:
        print('Debug Mode is active; log_likelihood will always return 0')
    ## Fix this causing dependency on data being included in file (actually fine if you don't need a plot title)
#     if data.mjd_0.min() <=3:##== tess_2020bpi.mjd_0[0] or data.mjd_0[0] == tess_2020bpi_a.mjd_0[0]:
#         title = 'TESS'
#     elif data.mjd_0.min() > 3: ##== ztf_2020bpi.mjd_0[0]:
#         title = 'ZTF'
#     else:
#         raise ValueError('Must Provide data for TESS or ZTF')

    if MC:    
        fits = doMCMC(data[data.mjd_0 <= cutoff], guess, scale, cutoff, nwalkers, nburn, nsteps,curveModel,debug,dataType,savePlots,fileNameExtras,plotExt)
    elif not MC:
        fits = np.array([list(guess.values())])
    fitPD = pd.DataFrame(fits)
    fitPD.columns = guess.keys()
    
    ##Plotting Models
    fig,ax = plt.subplots(figsize=(8,8))
    plt.xlabel("mjd-"+str(round(tess_2020bpi.mjd.min())));
    ax.set_ylabel("Normalized Flux");
    ax.set_xlim(left=0,right=cutoff*1.05)
    ax.set_ylim(top=data[data.mjd_0 <= cutoff].flux.to_numpy().max()*1.1) ##maybe unneeded
    
    ## Plotting data and setting up to plot random samples
    tRange = np.linspace(0,data.mjd_0.max(),data.mjd_0.max()*48)
    dummyPD = pd.DataFrame()
    dummyPD['mjd_0'] = tRange
    dummyPD['e_flux'] = np.zeros(np.size(tRange))
    dummyPD['e_raw_flux'] = np.zeros(np.size(tRange))
    
    if dataType == 'ZTF':
        ax.scatter(data[data.mjd_0 < cutoff].mjd_0, 
        data[data.mjd_0 < cutoff].flux, alpha=0.25, color='black',zorder=2, label=dataType)
        plt.xlabel("mjd-"+str(round(tess_2020bpi.mjd.min())));
        dummyPD['flux'] = data['flux']
    elif curveModel == 'dcRaw':
        ax.scatter(data[data.mjd_0 < cutoff].mjd_0, 
        data[data.mjd_0 < cutoff].raw_flux, alpha=0.5, color=plotPal[0], label=dataType)
        dummyPD['flux'] = data['raw_flux']
    else:
        ax.scatter(data[data.mjd_0 < cutoff].mjd_0, 
        data[data.mjd_0 < cutoff].flux, alpha=0.5, color=plotPal[0], label=dataType)
        dummyPD['flux'] = data['flux']
    ## Note: Should add error bars to plot (requires changing it to ax.errorbars)
    
    ## Plotting random models
    ## Note: assert that numModels must be <= nwalkers*nsteps (not particularly important)
    if not numModels:
        numModels = round(0.001*nwalkers*nsteps)
        
    indices = list(dict.fromkeys([np.random.randint(low=0,high=len(fits[:,0])) ##makes it so you don't overplot any duplicates (mostly for when you're just plotting your guess)
                                  for i in range(0,numModels)]))
    if len(indices) > 1:
        for i in indices:
            model, var = lc_model(fits[i],guess.keys(),dummyPD[dummyPD.mjd_0 < cutoff],curveModel=curveModel,cutoff=cutoff)
            alfPogForm = 0.6 - 0.1*(numModels/(numModels+250))
            ## above could be bad in the case of a large sample with few models selected
            ax.plot(dummyPD[dummyPD.mjd_0 < cutoff].mjd_0,model, alpha=alfPogForm, linewidth=3, color=plotPal[1],zorder=1)
            ## line width not necessarily representative of actual width of 
            ## distribution? Maybe a concern
            if i == indices[-1]:
                if numModels >= nwalkers*nsteps:
                    ax.lines[-1].set_label('Model Distribution (N='+str(numModels)+')')
                elif numModels < nwalkers*nsteps:
                    ax.lines[-1].set_label('Random Model Samples (N='+str(numModels)+')')
                    

        
    ## Plotting Median Model
    theta = [np.median(fits[:,i]) for i in range(0,np.shape(fits)[1])]
    model, var = lc_model(theta,guess.keys(),dummyPD[dummyPD.mjd_0 < cutoff],curveModel=curveModel,cutoff=cutoff)
    ax.plot(dummyPD[dummyPD.mjd_0 < cutoff].mjd_0,model, linewidth=2,
            label='Median Fit',color=plotPal[2])
    modelDat, varDat = lc_model(theta,guess.keys(),data,curveModel=curveModel,cutoff=cutoff)
    ## Making into a PD for future use
    modelPD = data.copy()
    modelPD['modelFlux'] = modelDat
    
    ##Plotting 16th and 84th Percentile Models
    theta16, theta84 = [np.quantile(fits[:,i],0.16) for i in range(0,np.shape(fits)[1])], [np.quantile(fits[:,i],0.84) for i in range(0,np.shape(fits)[1])]
    model16, var16 = lc_model(theta16,guess.keys(),dummyPD[dummyPD.mjd_0 < cutoff],curveModel=curveModel,cutoff=cutoff)
    ax.plot(dummyPD[dummyPD.mjd_0 < cutoff].mjd_0,model16, linewidth=2,
            label='16th and 84th Percentile',color=plotPal[2],linestyle='--')
    model84, var84 = lc_model(theta84,guess.keys(),dummyPD[dummyPD.mjd_0 < cutoff],curveModel=curveModel,cutoff=cutoff)
    ax.plot(dummyPD[dummyPD.mjd_0 < cutoff].mjd_0,model84, linewidth=2,
            color=plotPal[2],alpha=0.75,linestyle='--')
    
    ##Plotting Residuals
    if curveModel == 'dcRaw':
        modelPD['modelResidual'] = data.raw_flux - modelPD['modelFlux']
    else:
        modelPD['modelResidual'] = data.flux - modelPD['modelFlux']
    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("bottom",size="25%",pad=0.03)
    ax.figure.add_axes(ax2)
    ax2.scatter(modelPD[modelPD.mjd_0 < cutoff].mjd_0, modelPD[modelPD.mjd_0 < cutoff].modelResidual,
             alpha=1, color=plotPal[0],s=3)
    ax2.grid()
    ax2.set_xlim(left=0,right=cutoff*1.05)
    
    ax.legend();
    #ax.set_title(title);
    if savePlots:
        ## Saves fit Plot
        fileName = np.str('mod-'+np.str(curveModel)+'-nparam-'+np.str(len(guess)+1)+'-nwalk-'+np.str(nwalkers)+'-nstep-'+np.str(nsteps))
        dirString = np.str('./plots/lcFit/'+np.str(dataType)+'/fitModel/')
        mkdir(dirString)
        fig.savefig(dirString+fileName+plotExt)
        ## Saves PD with the model flux included
        dirString = np.str('./fitPD/')
        mkdir(dirString)
        fileName = np.str('mod-'+np.str(curveModel)+'-cutoff-'+np.str(cutoff)+'-nparam-'+np.str(len(guess)+1)+'-nwalk-'+np.str(nwalkers)+'-nstep-'+np.str(nsteps)+'.csv')
        modelPD.to_csv(dirString+fileName,index=False)
        ## Saves all the sampled parameters 
        dirString = np.str('./fitPD/params/')
        mkdir(dirString)
        fileName = np.str('params-mod-'+np.str(curveModel)+'-cutoff-'+np.str(cutoff)+'-nparam-'+np.str(len(guess)+1)+'-nwalk-'+np.str(nwalkers)+'-nstep-'+np.str(nsteps)+'.csv')
        fitPD.to_csv(dirString+fileName,index=False)
    return fitPD, modelPD