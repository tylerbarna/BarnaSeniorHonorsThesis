import numpy as np
import emcee
import corner

def get_fullparam(theta):

    nparam = len(theta)
    
    # 0th parameter is time of explosion t0
    # 1st parameter is a
    # 2nd parameter is sigma 
    # 3rd parameter is power, this is left out for now

    assert (nparam >= 3) and (nparam <= 5), "invalid nparam"

    if nparam == 2:
        t0, a = theta
        sigma = 0.
        # power = 2.
    elif nparam == 3:
        t0, a, sigma = theta
        # power = 2.
    # elif nparam == 4:
    #     t0, a, sigma, power = theta

    return t0, a, sigma#, power

def log_prior(theta):
    
    nparam = len(theta)
    t0, a, sigma = get_fullparam(theta)
    
    logpr = 0.

    if nparam <= 2:
        return logpr

    if sigma <= 0:
        return -np.inf
      
    # Can play around with different priors if you want to    
    # if nparam >= 4:
    #     logpr -= np.log(sigma)

    return logpr

def log_likelihood(theta, data):
    
    t0, a, sigma = get_fullparam(theta)

    power = 2.0
    
    var = (data['flux_err']**2 + sigma**2)
    model = np.heaviside(data['mjd'] - t0, 0) * a * (data['mjd'] - t0)**power
    logl = -0.5 * (np.sum(np.log(2 * np.pi * var) + 
                            ((data['flux'] - model)**2 / var) ))
        
    return logl

    
def log_posterior(theta, data):
    
    logpr = log_prior(theta)
    
    if logpr == -np.inf:
        return logpr
    else:
        return logpr + log_likelihood(theta, data)

def doMCMC(data, guess, scale, nwalkers=100, nburn=1500, nsteps=3000):
    '''
    Takes data which contains mjd and flux data
    and performs an mcmc fit on it
    '''
    ndim = len(guess)
    assert ndim == len(scale)

    starting_guesses = np.random.randn(nwalkers, ndim)*scale + guess

    print('sampling...')
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, threads=1, args=[data])
    sampler.run_mcmc(starting_guesses, nsteps)
    print('done')
    
    
    tlabels = [r"t0", 
           r"a",
           r"sigma",
           # r"power"
           ]
    samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))
    sampler.reset()

    figcorner = corner.corner(samples, labels=tlabels[0:ndim],
                    show_titles=True, title_fmt=".6f", verbose=True,
                    title_kwargs={"fontsize": 11}, label_kwargs={"fontsize": 14})

    return samples