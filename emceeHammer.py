from useful_functions import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import emcee
import corner
from multiprocessing import Pool
##Need to figure out why this stuff is needed
tess_2020bpi = pd.read_csv('JhaData/TESS_SN2020bpi.csv')[::2]
tess_2020bpi['mjd_0'] = tess_2020bpi['mjd'] - tess_2020bpi['mjd'].min()
# fluxNorm = 0.4*np.max(tess_2020bpi['flux'])
# tess_2020bpi_norm = tess_2020bpi
# tess_2020bpi_norm.flux = tess_2020bpi.flux/fluxNorm
# tess_2020bpi_norm.e_flux = tess_2020bpi.e_flux/fluxNorm
# tess_2020bpi_a = pd.read_csv('JhaData/TESS_SN2020bpi_updated.csv')[::2]
# tess_2020bpi_a['mjd_0'] = tess_2020bpi_a['mjd'] - tess_2020bpi['mjd'].min()
# tess_2020bpi_a_norm = tess_2020bpi_a
# tess_2020bpi_a_norm.flux = tess_2020bpi_a.flux/fluxNorm
# tess_2020bpi_a_norm.e_flux = tess_2020bpi_a.e_flux/fluxNorm

def get_fullparam(theta, thetaKeys):
    params = {'t0':7, 
              'a':0.0025,
              'sigma':0,
              'power':1.4,
              'y0':0,
              'mean':10.65,
              'std':0.05,
              'gFactor':0.001}
    params.update(dict(zip(thetaKeys,theta)))
    return params

def lc_model(theta,thetaKeys, data, curveModel='standard'):
    thetaDict = get_fullparam(theta,thetaKeys)
    
    if curveModel =='standard':
        var = (data['e_flux']**2 + thetaDict['sigma']**2) ##normalize the flux and fix
        model = np.array([0 if t <= thetaDict['t0'] else
                 thetaDict['a'] * (t - thetaDict['t0'])**thetaDict['power'] 
                 for t in data['mjd_0']])
        
    elif curveModel =='powerFix1':
        var = (data['e_flux']**2 + thetaDict['sigma']**2)
        model = np.array([0 if t <= thetaDict['t0'] else
                 thetaDict['a']**thetaDict['power']  * (t - thetaDict['t0'])**thetaDict['power'] 
                 for t in data['mjd_0']])
        
    elif curveModel =='powerFix2':
        var = (data['e_flux']**2 + thetaDict['sigma']**2)
        model = np.array([0 if t <= thetaDict['t0'] else
                 thetaDict['a']**(1/thetaDict['power'])  * (t - thetaDict['t0'])**thetaDict['power'] 
                 for t in data['mjd_0']])
        
    elif curveModel =='gaussian':
        var = (data['e_flux']**2 + thetaDict['sigma']**2 + thetaDict['std'])
        model = np.array([0 if t <= thetaDict['t0'] else
                 (thetaDict['a'] * (t - thetaDict['t0'])**thetaDict['power'])
                 for t in data['mjd_0']]) + gaussian(theta,thetaKeys,data)
        
    elif curveModel =='decoupled':
        var = (data['e_flux']**2 + thetaDict['sigma']**2)
        model = np.array([thetaDict['y0'] if t <= thetaDict['t0'] else
                 ((thetaDict['a']*((t - thetaDict['t0'])/
                                   (np.max(data['mjd_0'])-thetaDict['t0']))**thetaDict['power']) 
                  +thetaDict['y0'])
                          for t in data['mjd_0']])
        
    elif curveModel =='dcGauss':
        var = (data['e_flux']**2 + thetaDict['sigma']**2 + thetaDict['std'])
        model = np.array([thetaDict['y0'] if t <= thetaDict['t0'] else
                          ((thetaDict['a']*((t - thetaDict['t0'])/
                                            (np.max(data['mjd_0'])-thetaDict['t0']))**thetaDict['power']) 
                           +thetaDict['y0'])
                          for t in data['mjd_0']]) + gaussian(theta,thetaKeys,data)
    else:
        raise KeyError('Must Provide Valid Model')
    return model, var

def gaussian(theta, thetaKeys,data):
    thetaDict = get_fullparam(theta,thetaKeys)
    curveFrac = (1/(thetaDict['std']*np.sqrt(2*np.pi)))
    curveExp = -0.5*((data['mjd_0']-thetaDict['mean'])/thetaDict['std'])**2
    curve = thetaDict['gFactor']*curveFrac * np.exp(curveExp)
    return curve 

def log_prior(theta, thetaKeys):
    
    nparam = len(theta)
    thetaDict = get_fullparam(theta, thetaKeys)
    
    logpr = 0.

    if thetaDict['sigma'] < 0:
        return -np.inf

    if thetaDict['t0'] < 0:
        return -np.inf
    
    if thetaDict['t0'] > 10:
        return -(100)**np.float(thetaDict['t0'])
    
    if thetaDict['a'] <= 0:
        return -np.inf
    
    if thetaDict['y0'] > .5:
        return -np.inf
    
    if thetaDict['y0'] < -0.5:
        return -np.inf
    
    if thetaDict['mean'] <= thetaDict['t0']:
        return -np.inf
    
    if thetaDict['mean'] > thetaDict['t0']+7:
        return -np.inf
    
    if thetaDict['mean'] > 12:
        return -np.inf
    
    if thetaDict['std'] <= 0:
        return -np.inf
    
#     if thetaDict['std'] > 5:
#         return -np.inf
    
    if thetaDict['gFactor'] <= 0:
        return -np.inf
    
    if thetaDict['gFactor'] > 1:
        return -np.inf
    
    if thetaDict['power'] <= 1:
        return -np.inf
    
    if thetaDict['power'] >= 2.5:
        return -np.inf
    return logpr

def log_likelihood(theta, thetaKeys, data, curveModel='standard'):
    
    thetaDict = get_fullparam(theta, thetaKeys)
    #print(thetaDict)
    model,var = lc_model(thetaDict.values(),thetaDict.keys(),data, curveModel)
    logl = -0.5 * (np.sum(np.log(2 * np.pi * var) + 
                            ((data.flux - model)**2 / var) ))
        
    return logl

    
def log_posterior(theta, thetaKeys, data, curveModel='standard'):
    
    logpr = log_prior(theta, thetaKeys)
    
    if logpr == -np.inf:
        return logpr
    else:
        return logpr + log_likelihood(theta, thetaKeys, data, curveModel)

def doMCMC(data, guess, scale, 
           nwalkers=100, nburn=1500, nsteps=3000, 
           curveModel='standard',savePlots=True):
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
                                    args=[list(guess.keys()),data,curveModel])
    sampler.run_mcmc(starting_guesses, nsteps,progress="notebook")
    print('done')
    
    
    samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))
    sampler.reset()
    tlabels = list(guess.keys())
    figcorner = corner.corner(samples, labels=tlabels[0:ndim],
                    show_titles=True, title_fmt=".6f", verbose=True,
                    title_kwargs={"fontsize": 11}, label_kwargs={"fontsize": 14})
    ## add functionality for saving corner plot here

    return samples

def hammerTime(data, guess, scale, cutoff=16.75, 
                    nwalkers=100, nburn=1500, nsteps=3000,
                    curveModel='standard',numModels=None,
                   MC=True, plotPal=('#003C86','#7CFFFA','#008169'),savePlots=False):
    ## pretty sloppy implementation, could make this a part of the function directly
    ## assert nwalkers must be twice number of parameters and fix if not
    
    ##Fix this causing dependency on data being included in file 
    if data.mjd_0[0] <=3:##== tess_2020bpi.mjd_0[0] or data.mjd_0[0] == tess_2020bpi_a.mjd_0[0]:
        title = 'TESS'
    elif data.mjd_0[0] > 3 and data.mjd_0[0] < 10: ##== ztf_2020bpi.mjd_0[0]:
        title = 'ZTF'
    else:
        raise ValueError('Must Provide data for TESS or ZTF')
    if savePlots:
        RootDirString = np.str('./plots/'+title+'/')
        mkdir(dirString)
    if MC:    
        fits = doMCMC(data[data.mjd_0 <= cutoff], guess, scale, nwalkers, nburn, nsteps,curveModel,savePlots)
    elif not MC:
        fits = np.array([list(guess.values())])

    
    fig,ax = plt.subplots(figsize=(8,8))

    ax.scatter(data[data.mjd_0 <= cutoff].mjd_0, 
    data[data.mjd_0 <= cutoff].flux,alpha=0.25,color=plotPal[0],label=title)
    ## Add error bars to plot 

    tRange = np.linspace(0,cutoff,cutoff*48)
    dummyPD = pd.DataFrame()
    dummyPD['mjd_0'] = tRange
    dummyPD['e_flux'] = np.zeros(np.size(tRange))
    dummyPD['flux'] = data[data.mjd_0 <= cutoff]['flux']
    
    ## assert that numModels must be <= nwalkers*nsteps
    if numModels ==None: ## ehh try to improve this
        ## desired behavior is to plot all fits if ==None and also not plot
        ## at all in the case of there only being one guess and no mcmc call
        ## current implementation doesn't guarantee all fits will be plotted
        ## probably unecessary, just plot like 1% of all models or so
        numModels = round(0.01*nwalkers*nsteps)
    indices = list(dict.fromkeys([np.random.randint(low=0,high=len(fits[:,0])) ##makes it so you don't overplot any duplicates (mostly for when you're just plotting your guess)
                                  for i in range(0,numModels)]))
    if len(indices) > 1:
        for i in indices:
            model, var = lc_model(fits[i],guess.keys(),dummyPD,curveModel)
            alfPogForm = 0.25 - 0.25*(numModels/(numModels+250))
            ## above could be bad in the case of a large sample with few models selected
            ax.plot(tRange,model, alpha=alfPogForm, linewidth=3, color=plotPal[1])
            ## line width not necessarily representative of actual width of 
            ## distribution? Maybe a concern
            if i == indices[-1]:
                if numModels >= nwalkers*nsteps:
                    ax.lines[-1].set_label('Model Distribution (N='+str(numModels)+')')
                elif numModels < nwalkers*nsteps:
                    ax.lines[-1].set_label('Random Model Samples (N='+str(numModels)+')')
        
    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("bottom",size="25%",pad=0.03)
    ax.figure.add_axes(ax2)
    #print(fits[0])
    theta = [np.median(fits[:,i]) for i in range(0,np.shape(fits)[1])]
    #print(theta)
    model, var = lc_model(theta,guess.keys(),data[data.mjd_0 <= cutoff],curveModel)
    ax.plot(data[data.mjd_0 <= cutoff].mjd_0,model, linewidth=2,
            label='Median Fit',color=plotPal[2])
    residual = data[data.mjd_0 <= cutoff].flux - model
    #print(model)
    ax2.scatter(data[data.mjd_0 <= cutoff].mjd_0, residual,
             alpha=1, color=plotPal[0],s=1)
    ax2.grid()
    plt.xlabel("mjd-"+str(round(tess_2020bpi.mjd.min())));
    ax.set_ylabel("flux");
    ax.legend()
    ax.set_title(title);
    ## getting weird behavior if a walker goes significantly off the mark in that it will plot the y axis relative to that max rather than fluxNorm
    ## Update: was an issue with fluxNorm value, not this
    ## add functionality for saving plot here
    return [fits, fits[indices]]