import pandas as pd
from pandas import DataFrame as df
import numpy as np

def lfldaComputeCoherenceScore (resultsFile):
    readLfldaResultsFile (resultsFile)

def readLfldaResultsFile (resultsFile):
    results = open(resultsFile, "r")
    
