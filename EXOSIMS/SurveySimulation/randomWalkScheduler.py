from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import numpy as np


class randomWalkScheduler(SurveySimulation):
    """randomWalkScheduler

    This class implements a random walk scheduler that selects the 
    next target at random from the pool of currently available targets.

    This is useful for mapping out the space of possible mission outcomes
    for a fixed population of planets in order to validate other schedulers.
    """

    def choose_next_target(self,old_sInd,sInds,slewTime):
        """Choose next target at random

        Args:
            old_sInd (integer):
                Index of the previous target star
            sInds (integer array):
                Indices of available targets
            slewTime (float array):
                slew times to all stars (must be indexed by sInds)
                
        Returns:
            sInd (integer):
                Index of next target star
        
        """
    
        sInd = np.random.choice(sInds)
        
        return sInd

    
