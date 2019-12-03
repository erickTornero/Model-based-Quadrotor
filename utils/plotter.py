import matplotlib.pyplot as plt
import numpy as np
def drawStdPlot( batchResults, title, xlabel, ylabel, startStep = 1, color = 'r', axes = None, scalefactor=1 ) : 
    """
        This Function was substracted from W. Pumacay, 
        https://github.com/wpumacay/DeeprlND-projects/
        Draws an std-plot from a batch of results from various experiments
    
    Args:
        batchResults (list): list of experiments, each element containing results
                             for each expriment (training run) as a list of elements
    """
    # sanity-check: should have at least pass some data
    assert len( batchResults ) > 0, 'ERROR> should have passed at least some data'

    # convert batch to user friendly np.ndarray
    batchResults = np.array( batchResults )

    # if no axes given, create a new one
    if axes is None :
        fig, axes = plt.subplots()

    # each element has _niters elements on it (iters: episodes, steps, etc.)
    _niters = batchResults.shape[1]
    _xx = (np.arange( _niters ) + startStep) * scalefactor

    # grab mean and std over all runs
    _mean = batchResults.mean( axis = 0 )
    _std = batchResults.std( axis = 0 )

    # do the plotting
    axes.plot( _xx, _mean, color = color, linestyle = '-')
    axes.fill_between( _xx, _mean - 2. * _std, _mean + 2. * _std, color = color, alpha = 0.2 )

    axes.set_xlabel( xlabel )
    axes.set_ylabel( ylabel )
    axes.set_title( title )

    return axes