# coding: utf-8
# Author: Marc Weber

"""
=========================================================================================
RNA folding tools
=========================================================================================
"""

import pandas as pd
import RNA
from mwTools.general import rolling_window_deque



def compute_folding_energy(seq):
    structure, energy = RNA.fold(seq)
    return pd.Series({'folding_structure':structure, 'folding_energy':energy})


def compute_folding_energy_rolling_window(seq, rnaWindow=80, rnaWindowStep=1):

    rollingWinList = [("".join(list(seq)), x0, x1) for (seq, x0, x1) in
        rolling_window_deque(seq, size=rnaWindow, step=rnaWindowStep, yield_position=True)]
    rollingWinDf = pd.DataFrame(rollingWinList, columns=['window_seq', 'window_start', 'window_end'])
    # Compute folding energy on rolling window
    rollingWinDf = rollingWinDf.join(rollingWinDf['window_seq'].apply(compute_folding_energy))
    rollingWinDf['window_center'] = (rollingWinDf['window_start'] + rollingWinDf['window_end'] - 1)/2.
    rollingWinDf = rollingWinDf.set_index('window_center')
    return rollingWinDf
