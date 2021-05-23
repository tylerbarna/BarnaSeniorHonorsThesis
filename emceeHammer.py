from useful_functions import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import emcee
import corner
from multiprocessing import Pool
##Nuisance Stuff
tess_2020bpi = pd.read_csv('./JhaData/TESS_SN2020bpi.csv')[::2] ## old file is doubled sampling
tess_2020bpi['mjd_0'] = tess_2020bpi['mjd'] - tess_2020bpi['mjd'].min()
ztf_2020bpi = pd.read_csv('./JhaData/ztf_SN2020bpi.csv')
ztf_2020bpi['mjd_0'] = ztf_2020bpi['mjd'] - tess_2020bpi['mjd'].min()


def normLC(lcDF):
    '''
    normalizes lc to arbitrary value or default of
    around 40% of original lightcurve sent by Michael.
    There's a small tweak to it so the TESS lightcurve is roughly ~1
    at mjd_0=17, but this shouldn't affect the fit other than the 
    scaling constant very slightly.
    I don't believe there would be any significant effect if 
    one were to use a different normalization constant, in this case
    you would alter the fluxNorm variable and use this function as usual
    
    Arguments:
    - lcDF: pandas DataFrame with the columns I describe in the documentation notebook
    '''
    normFrame = lcDF.copy()
    normFrame['mjd_0'] = normFrame['mjd'] - tess_2020bpi['mjd'].min()
    fluxNorm = 1.2*np.max(tess_2020bpi['flux'].rolling(24).median())
    normFrame['flux'] = lcDF['flux']/fluxNorm
    normFrame['e_flux'] = lcDF['e_flux']/fluxNorm
    if 'raw_flux' in normFrame.columns:
        normFrame['raw_flux'] = lcDF['raw_flux']/fluxNorm
        normFrame['e_raw_flux'] = lcDF['e_raw_flux']/fluxNorm
#     if 'e_flux_tuple' in normFrame.columns: ##need to get working
#         normFrame['e_flux_tuple'] = [np.array(np.abs(lcDF['e_flux_tuple'].to_numpy()[ind][0])/fluxNorm,
#                                               np.abs(lcDF['e_flux_tuple'].to_numpy()[ind][1])/fluxNorm)
#                                      for ind in range(len(lcDF['e_flux_tuple']))]
    return normFrame
def get_fullparam(theta, thetaKeys):
    '''
    Function used at various points in other functions so emcee fit can be run while
    evaluating only the parameters listed in the guess/scale dictionaries and setting the others
    to predefined constant values. Two arguments are passed to the function because emcee
    cannot natively pass dictionaries when varying parameters as far as I can tell.
    Note: the values in the params dictionary are the default values for each parameter;
    the parameters that are varying will have their values updated when this function is called.
    If changing these default values, be careful that they don't cause conflicts with priors.
    
    Arguments:
    - theta: an array containing the values of each parameter to be varied by emcee
    - thetaKeys: the key values for each of these parameters to construct a dictionary
    '''
    params = {'t0':7, 
              'a':1,
              'sigma':0.0,
              'power':1.8,
              'y0':0,
              'mean':10.65,
              'std':1,
              'gFactor':0.0,
              'bkg mod':0,
              'flux scale':1,
              'flux shift':0,
              'cutoffParam':17}
    params.update(dict(zip(thetaKeys,theta)))
    return params

def lc_model(theta,thetaKeys, data, ztfData=None,curveModel='dcRaw',cutoff=17):
    '''
    A function that includes a number of different models, though the only one particularly
    relevant to the final analysis is dcRaw. Function evaluates a given model for the 
    parameters provided and then returns the flux and variance.
    
    Arguments:
    - theta: an array containing the values of each parameter to be varied by emcee
    - thetaKeys: the key values for each of these parameters to construct a dictionary
    - data: the lightcurve data that is being fit; assumes the DataFrame is formatted
    as outlined in the documentation file
    - ztfData: part of a model that isn't currently functional that attempts to fit the TESS
    data to the ZTF lightcurve and also model the lightcurve at the same time; can be safely
    ignored for most models (default: None)
    - curveModel: string that corresponds to desired model (default: dcRaw)
    - cutoff: upper limit for the data emcee uses to fit the lightcurve provided as 
    number of days after the start of TESS Sector 21 (default: 17)
    '''
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
    elif curveModel =='dcCutoff': ## not functional
        print('Warning: Model may not behave as intended')
        var = (data['e_flux'].to_numpy()**2 + thetaDict['sigma']**2)
        model = np.array([thetaDict['y0'] if t <= thetaDict['t0'] else
                          ((thetaDict['a']*((t - thetaDict['t0'])/
                                            (thetaDict['cutoffParam']-thetaDict['t0']))**thetaDict['power']) 
                           +thetaDict['y0'])
                          for t in data[data.mjd_0 < thetaDict['cutoffParam']]['mjd_0'].to_numpy()])
    elif curveModel =='dcRaw':
        var = (data['e_raw_flux'].to_numpy()**2 + thetaDict['sigma']**2)
        model = np.array([thetaDict['y0'] if t <= thetaDict['t0'] else
                          ((thetaDict['a']*((t - thetaDict['t0'])/
                                            (cutoff-thetaDict['t0']))**thetaDict['power']) 
                           +thetaDict['y0'])
                          for t in data['mjd_0'].to_numpy()])
    elif curveModel =='dcRaw2020bpi': ## (Mostly) Not Functional
        print('Warning: Model may not behave as intended')
        var = (data['e_raw_flux'].to_numpy()**2 + thetaDict['sigma']**2)
        model = np.array([thetaDict['y0'] if t <= thetaDict['t0'] else
                          ((thetaDict['a']*((t - thetaDict['t0'])/
                                            (cutoff-thetaDict['t0']))**thetaDict['power']) 
                           +thetaDict['y0'])
                          for t in data['mjd'].to_numpy()])
        
    elif curveModel =='dcForAll': ##not functional
        print('Warning: Model may not behave as intended')
        var = ((thetaDict['flux scale']*data['e_raw_flux'].to_numpy())**2 + ztfData['e_flux'].to_numpy()**2+thetaDict['sigma']**2) ## is ztf error needed?
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
    '''
    Checks that none of the parameters violate any prior conditions. If a prior is violated,
    function returns a value of negative infinity. Soft penalties can also be added that 
    are dependent on how different the parameter is from the expected value
    
    Arguments:
    - theta: an array containing the values of each parameter to be varied by emcee
    - thetaKeys: the key values for each of these parameters to construct a dictionary
    nparam = len(theta)
    thetaDict = dict(zip(thetaKeys,theta))#get_fullparam(theta, thetaKeys)
    '''
    thetaDict = get_fullparam(theta, thetaKeys)
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
#     elif thetaDict['t0'] > 16:
#         return -np.inf
    
#     elif thetaDict['t0'] >11: ##more just for ztf so it doesn't look so odd
#         logpr += -(thetaDict['t0'])**(1/4)
    
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
    
    if thetaDict['cutoffParam'] < 10:
        return -np.inf
    
    elif thetaDict['gFactor'] > 1: 
        return -np.inf
    
    if thetaDict['power'] <= 0: 
        return -np.inf
    
    elif thetaDict['power'] >= 5:
        return -np.inf
    if thetaDict['flux scale'] < 0:
        return -np.inf
    return logpr

def log_likelihood(theta, thetaKeys, data, curveModel,cutoff=17,ztfData=None):
    '''
    Calculates the likelihood of the model with the given parameter values. Some complication
    is added to the function by checking for a few specfic curve models, but this shouldn't
    affect standard behavior
    
     Arguments:
    - theta: an array containing the values of each parameter to be varied by emcee
    - thetaKeys: the key values for each of these parameters to construct a dictionary
    - data: the lightcurve data that is being fit; assumes the DataFrame is formatted
    as outlined in the documentation file
    - curveModel: string that corresponds to desired model (default: dcRaw)
    - cutoff: upper limit for the data emcee uses to fit the lightcurve provided as 
    number of days after the start of TESS Sector 21 (default: 17)
    - ztfData: part of a model that isn't currently functional that attempts to fit the TESS
    data to the ZTF lightcurve and also model the lightcurve at the same time; can be safely
    ignored for most models (default: None)
    '''
    #return 0
    thetaDict = get_fullparam(theta, thetaKeys)
    #print(thetaDict)
    model,var = lc_model(theta,thetaKeys,data[data.mjd_0 < cutoff],ztfData=ztfData, curveModel=curveModel,cutoff=cutoff)
    if curveModel == 'dcRaw':
        flux = data['raw_flux'].to_numpy()
    elif curveModel == 'dcForAll' and ztfData: ## not functional
        thetaDict = get_fullparam(theta, thetaKeys)
        flux = thetaDict['flux scale']* data['raw_flux'].to_numpy() + thetaDict['flux shift']
    elif curveModel == 'dcCutoff': ## Not functional
        thetaDict = get_fullparam(theta, thetaKeys)
        flux = data[data.mjd_0 < thetaDict['cutoffParam']]['flux'].to_numpy()
    else:
        flux = data['flux'].to_numpy()
               
    logl = -0.5 * (np.sum(np.log(2 * np.pi * var) + 
                            ((flux - model)**2 / var) ))
        
    return logl

    
def log_posterior(theta, thetaKeys, data, curveModel,cutoff=17,debug=False,ztfData=None):
    '''
    Combines the log likelihood and log prior functions; a parameter that violates a 
    prior will always return the value of log prior (negative infinity)
    
    Arguments:
    - theta: an array containing the values of each parameter to be varied by emcee
    - thetaKeys: the key values for each of these parameters to construct a dictionary
    - data: the lightcurve data that is being fit; assumes the DataFrame is formatted
    as outlined in the documentation file
    - curveModel: string that corresponds to desired model (default: dcRaw)
    - cutoff: upper limit for the data emcee uses to fit the lightcurve provided as 
    number of days after the start of TESS Sector 21 (default: 17)
    - debug: when set to True, this makes it so the log likelihood is never returned, 
    only the log prior. This is useful for investigating issues with prior 
    conditions (default: False)
    - ztfData: part of a model that isn't currently functional that attempts to fit the TESS
    data to the ZTF lightcurve and also model the lightcurve at the same time; can be safely
    ignored for most models (default: None)
    '''
    thetaDict = get_fullparam(theta, thetaKeys)
    logpr = log_prior(thetaDict.values(), thetaDict.keys())
    
    if logpr == -np.inf or debug:
        return logpr
    else:
        return logpr + log_likelihood(thetaDict.values(), thetaDict.keys(), data, curveModel=curveModel,cutoff=cutoff,ztfData=ztfData)

def doMCMC(data, guess, scale, cutoff=17, ztfData=None,
           nwalkers=100, nburn=1000, nsteps=3000, 
           curveModel='dcRaw',debug=False,
           dataType='TESS',savePlots=None,
           fileNameExtras='',plotExt='.pdf'):
    '''
    Takes data which contains mjd and flux data and performs an mcmc fit on it;
    also makes a corner plot of the selected parameters. Some of the starting guesses
    axis shaping can appear a bit obtuse, but this is so there can be a dynamic 
    number of parameters. This behaves as intended with the TESS data.
    
    Arguments:
    - data: the lightcurve data that is being fit; assumes the DataFrame is formatted
    as outlined in the documentation file
    - guess: a dictionary with the starting guess for the parameters emcee should vary
    - scale: a dictionary with the scale lengths for the parameters emcee should vary;
    make sure these agree with the guess argument
    - cutoff: upper limit for the data emcee uses to fit the lightcurve provided as 
    number of days after the start of TESS Sector 21 (default: 17)
    - ztfData: part of a model that isn't currently functional that attempts to fit the TESS
    data to the ZTF lightcurve and also model the lightcurve at the same time; can be safely
    ignored for most models (default: None)
    - nwalkers: number of walkers emcee should use; must be at least double 
    the number of parameters that are being varied. Increasing the number of walkers 
    will exonentially increase the runtime. Usually about 100 is enough (default: 100)
    - nburn: number of random walks emcee should do for the burn-in run; these steps are 
    discarded and the randomized parameters are then used as the starting values for the 
    main run (default: 1000)
    - nsteps: number of total steps to run for the emcee fit; the number of steps in the main
    run is equal to nsteps minus nburn (default: 3000)
    - curveModel: string that corresponds to desired model (default: dcRaw)
    - debug: when set to True, emcee will only consider the prior conditons when 
    evaluating the quality of a fit. This is useful for investigating issues with prior 
    conditions (default: False)
    - dataType: string that mostly exists so the plots can be saved to a folder that
    matches the data being used and so the plots have accurate labels (default: 'TESS')
    - savePlots: when set to True, saves the corner plot of the parameters (default: None)
    - fileNameExtras: string appended to the end of file names so one can manually add 
    unique names to plots and csv files (default: '')
    - plotExt: string used to specify the format to use when saving the 
    corner plot (default: '.pdf')
    '''
    ndim = len(guess)
    assert ndim == len(scale)

    starting_guesses = np.swapaxes(list({
        k:(np.random.randn(nwalkers)*scale[k]+v)
        for k,v in guess.items()}.values()),0,1)
#     print(starting_guesses)
    print('sampling...')
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, threads=-1, 
                                    args=[list(guess.keys()),data,curveModel,cutoff,debug,ztfData])
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
        fileName = np.str('mod-'+np.str(curveModel)+'-nparam-'+np.str(len(guess)+1)+'-nwalk-'+np.str(nwalkers)+'-nstep-'+np.str(nsteps)+np.str(fileNameExtras))
        figcorner.savefig(dirString+fileName+plotExt)
        

    return samples

def hammerTime(data, guess, scale, cutoff=17, ztfData=None,
                    nwalkers=100, nburn=1000, nsteps=3000,
                    curveModel='dcRaw',numModels=None,
                   MC=True, debug=False,
               plotPal=('#003C86','#7CFFFA','#008169'), 
               dataType='TESS',savePlots=None,fileNameExtras='',plotExt='.pdf'):
    '''
    Fits model to TESS or ZTF data using emcee.  Plots model on top of data and residuals an
    optionally saves the plot and fits/parameters.
    
     Arguments:
    - data: the lightcurve data that is being fit; assumes the DataFrame is formatted
    as outlined in the documentation file
    - guess: a dictionary with the starting guess for the parameters emcee should vary
    - scale: a dictionary with the scale lengths for the parameters emcee should vary;
    make sure these agree with the guess argument
    - cutoff: upper limit for the data emcee uses to fit the lightcurve provided as 
    number of days after the start of TESS Sector 21 (default: 17)
    - ztfData: part of a model that isn't currently functional that attempts to fit the TESS
    data to the ZTF lightcurve and also model the lightcurve at the same time; can be safely
    ignored for most models (default: None)
    - nwalkers: number of walkers emcee should use; must be at least double 
    the number of parameters that are being varied. Increasing the number of walkers 
    will exonentially increase the runtime. Usually about 100 is enough (default: 100)
    - nburn: number of random walks emcee should do for the burn-in run; these steps are 
    discarded and the randomized parameters are then used as the starting values for the 
    main run (default: 1000)
    - nsteps: number of total steps to run for the emcee fit; the number of steps in the main
    run is equal to nsteps minus nburn (default: 3000)
    - curveModel: string that corresponds to desired model (default: dcRaw)
    - numModels: allows for one to specify the number of random models to plot. If set to None,
    plot will randomly select .1% of the parameter combinations found by emcee (default: None)
    - MC: pseudo-debug flag that will only run the emcee fit if set to True. When set to False,
    the guess argument is taken as the best fit (default: True)
    - debug: when set to True, emcee will only consider the prior conditons when 
    evaluating the quality of a fit. This is useful for investigating issues with prior 
    conditions (default: False)
    - plotPal: color palette for plot in the form of a list. The first value is the color of
    the data points and residuals, the second is the color of the random fits, and the third
    is the color of the median fit (default: ('#003C86','#7CFFFA','#008169'))
    - dataType: string that mostly exists so the plots and csv files can be saved 
    to a folder that matches the data being used and so the plots have 
    accurate labels (default: 'TESS')
    - savePlots: when set to True, saves the plots as well as the csv files (default: None)
    - fileNameExtras: string appended to the end of file names so one can manually add 
    unique names to plots and csv files (default: '')
    - plotExt: string used to specify the format to use when saving the 
    corner plot (default: '.pdf')
    
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
        fits = doMCMC(data[data.mjd_0 <= cutoff], guess, scale, ztfData=ztfData, cutoff=cutoff, 
                      nwalkers=nwalkers, nburn=nburn, nsteps=nsteps, curveModel=curveModel, 
                      debug=debug,dataType=dataType,savePlots=savePlots,fileNameExtras=fileNameExtras,plotExt=plotExt)
    elif not MC:
        fits = np.array([list(guess.values())])
    fitPD = pd.DataFrame(fits)
    fitPD.columns = guess.keys()
    
    ##Plotting Models
    fig,ax = plt.subplots(figsize=(8,8))
    plt.xlabel("Modified Julian Date");
    ax.set_ylabel("Normalized Flux");
    #ax.set_xlim(left=tess_2020bpi.mjd.min()-(cutoff)*0.05,right=(cutoff)*1.05+tess_2020bpi.mjd.min())
#     ax.set_ylim(top=data[data.mjd_0 <= cutoff].flux.to_numpy().max()*1.1) ##maybe unneeded
    
    ## Plotting data and setting up to plot random samples
    tRange = np.linspace(0,data.mjd_0.max(),data.mjd_0.max()*48)
    dummyPD = pd.DataFrame()
    dummyPD['mjd_0'] = tRange
    dummyPD['mjd'] = np.linspace(tess_2020bpi['mjd'].min(),(data.mjd.max()),data.mjd_0.max()*48)
    dummyPD['e_flux'] = np.zeros(np.size(tRange))
    dummyPD['e_raw_flux'] = np.zeros(np.size(tRange))
    
    if dataType == 'ZTF':
        ax.scatter(data[data.mjd_0 < cutoff].mjd, 
        data[data.mjd_0 < cutoff].flux, alpha=0.75, color=plotPal[0],zorder=2, label=dataType)
        plt.xlabel("Modified Julian Date");
        dummyPD['flux'] = data['flux']
    elif curveModel == 'dcRaw':
        ax.scatter(data[data.mjd_0 < cutoff].mjd, 
        data[data.mjd_0 < cutoff].raw_flux, #yerr=data[data.mjd_0 < cutoff].e_raw_flux, 
                   alpha=0.5, color=plotPal[0], label=dataType)
        dummyPD['flux'] = data['raw_flux']
    else:
        ax.scatter(data[data.mjd_0 < cutoff].mjd, 
        data[data.mjd_0 < cutoff].flux, #yerr=data[data.mjd_0 < cutoff].e_flux, 
                   alpha=0.5, color=plotPal[0], label=dataType)
        dummyPD['flux'] = data['flux']
    ## Note: Should add error bars to plot (requires changing it to ax.errorbars) (no it doesn't lol)
    
#     if ztfData:## need to get working with e_flux_tuple
# #         offsetArray = ztfData[ztfData.mjd_0 > dummyPD.mjd_0.min()][ztfData.mjd_0 < cutoff].e_flux_tuple.to_numpy() 
# #         offsets =[np.array(np.abs(offsetArray[ind][0]),
# #                            np.abs(offsetArray[ind][1])) 
# #                   for ind in range(len(offsetArray))]
#         ax.errorbar(ztfData[ztfData.mjd_0 < cutoff].mjd_0,
#                       ztfData[ztfData.mjd_0 < cutoff].flux,
#                       yerr=ztfData[ztfData.mjd_0 < cutoff].e_flux,color=plotPal[3],label='ZTF')
    
    ## Plotting random models
    ## Note: assert that numModels must be <= nwalkers*nsteps (not particularly important)
    if not numModels: ## can change this to change number of models to plot
        numModels = round(0.001*nwalkers*(nsteps-nburn))
        
    indices = list(dict.fromkeys([np.random.randint(low=0,high=len(fits[:,0])) ##makes it so you don't overplot any duplicates (mostly for when you're just plotting your guess)
                                  for i in range(0,numModels)]))
    if len(indices) > 1:
        for i in indices:
            model, var = lc_model(fits[i],guess.keys(),dummyPD[dummyPD.mjd_0 < cutoff],curveModel=curveModel,cutoff=cutoff)
            tempDict = get_fullparam(fits[i], guess.keys())
            alfPogForm = 0.6 - 0.1*(numModels/(numModels+250))
            ## above could be bad in the case of a large sample with few models selected
            if curveModel == 'dcCutoff':
                ax.plot(dummyPD[dummyPD.mjd_0 < tempDict['cutoffParam']].mjd,model, alpha=alfPogForm, linewidth=3, color=plotPal[1],zorder=1)
            else:
                ax.plot(dummyPD[dummyPD.mjd_0 < cutoff].mjd,model, alpha=alfPogForm, linewidth=3, color=plotPal[1],zorder=1)
            ## line width not necessarily representative of actual width of 
            ## distribution? Maybe a concern
            if i == indices[-1]:
                if numModels >= nwalkers*nsteps:
                    ax.lines[-1].set_label('Model Distribution (N='+str(numModels)+')')
                elif numModels < nwalkers*nsteps:
                    ax.lines[-1].set_label('Random Model Samples (N='+str(numModels)+')')
                    

        
    ## Plotting Median Model
    theta = [np.median(fits[:,i]) for i in range(0,np.shape(fits)[1])]
    thetaDict = get_fullparam(fits[i], guess.keys())
    model, var = lc_model(theta,guess.keys(),dummyPD[dummyPD.mjd_0 < cutoff],curveModel=curveModel,cutoff=cutoff)
    ax.plot(dummyPD[dummyPD.mjd_0 < cutoff].mjd,model, linewidth=2,
            label='Median Fit',color=plotPal[2])
    modelDat, varDat = lc_model(theta,guess.keys(),data,curveModel=curveModel,cutoff=cutoff)
    ## Making into a PD for future use
    modelPD = data.copy()
    modelPD['modelFlux'] = modelDat
    
    
    ##Plotting Residuals
    if curveModel == 'dcRaw':
        modelPD['modelResidual'] = data.raw_flux - modelPD['modelFlux']
    else:
        modelPD['modelResidual'] = data.flux - modelPD['modelFlux']
    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("bottom",size="25%",pad=0.03)
    ax.figure.add_axes(ax2)
    ax2.scatter(modelPD[modelPD.mjd_0 < cutoff].mjd, modelPD[modelPD.mjd_0 < cutoff].modelResidual,
             alpha=1, color=plotPal[0],s=3)
    ax2.grid()
#     if curveModel == 'dcRaw2020bpi':
#         ax.set_xlim(left=tess_2020bpi.mjd.min()-(cutoff)*0.05,right=(cutoff)*1.05+tess_2020bpi.mjd.min())
#         ax2.set_xlim(left=tess_2020bpi.mjd.min()-(cutoff)*0.05,right=(cutoff)*1.05+tess_2020bpi.mjd.min())
#     else:
    ax.set_xlim(left=tess_2020bpi.mjd.min()-(cutoff)*0.05,right=(cutoff)*1.05+tess_2020bpi.mjd.min())
    ax2.set_xlim(left=tess_2020bpi.mjd.min()-(cutoff)*0.05,right=(cutoff)*1.05+tess_2020bpi.mjd.min())
    ax2.set_xlabel('Modified Julian Date')
    ax.legend();
    #ax.set_title(title);
    if savePlots:
        ## Saves fit Plot
        fileName = np.str('mod-'+np.str(curveModel)+'-nparam-'+np.str(len(guess)+1)+'-nwalk-'+np.str(nwalkers)+'-nstep-'+np.str(nsteps)+np.str(fileNameExtras))
        dirString = np.str('./plots/lcFit/'+np.str(dataType)+'/fitModel/')
        mkdir(dirString)
        fig.savefig(dirString+fileName+plotExt)
        ## Saves PD with the model flux included
        dirString = np.str('./fitPD/'+np.str(dataType)+'/')
        mkdir(dirString)
        fileName = np.str('mod-'+np.str(curveModel)+'-cutoff-'+np.str(cutoff)+'-nparam-'+np.str(len(guess)+1)+'-nwalk-'+np.str(nwalkers)+'-nstep-'+np.str(nsteps)+np.str(fileNameExtras)+'.csv')
        modelPD.to_csv(dirString+fileName,index=False)
        ## Saves all the sampled parameters 
        dirString = np.str('./fitPD/'+np.str(dataType)+'/params/')
        mkdir(dirString)
        fileName = np.str('params-mod-'+np.str(curveModel)+'-cutoff-'+np.str(cutoff)+'-nparam-'+np.str(len(guess)+1)+'-nwalk-'+np.str(nwalkers)+'-nstep-'+np.str(nsteps)+np.str(fileNameExtras)+'.csv')
        fitPD.to_csv(dirString+fileName,index=False)
    plt.show()
    return fitPD, modelPD