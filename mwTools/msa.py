# coding: utf-8
# Author: Marc Weber

"""
=========================================================================================
Multiple sequence alignment
=========================================================================================

We follow the definition of conservation score in @Capra2007 by computing the Jensen-Shannon divergence (JSD) of MSA columns.

I found the source code of the protein residue conservation prediction score by the Capra lab,
[here](http://compbio.cs.princeton.edu/conservation/). Adapted the source code and verified that we get the same results
as in the web tool.

Capra2007: Capra, J. A., & Singh, M. (2007). Predicting functionally important residues from sequence conservation.
Bioinformatics, 23(15), 1875–1882. http://doi.org/10.1093/bioinformatics/btm270 """


import subprocess
from subprocess import CalledProcessError
import shlex
from pathlib import Path
import re
import numpy as np

# This is the BLOSUM62 distribution. It is the default background distribution.
blosum62_bg_dist = dict(zip(
    ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'],
    [float(x) for x in "0.078 0.051 0.041 0.052 0.024 0.034 0.059 0.083 0.025 0.062 0.092 0.056 0.024 0.044 0.043 0.059 0.055 0.014 0.034 0.072".split()]
    ))


amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '-']
PSEUDOCOUNT = .0000001


def weighted_freq_count_pseudocount(col, seq_weights, pc_amount):
    """ Return the weighted frequency count for a column--with pseudocount."""

    # if the weights do not match, use equal weight
    if len(seq_weights) != len(col):
        seq_weights = [1.] * len(col)

    aa_num = 0
    freq_counts = dict(zip(amino_acids, len(amino_acids)*[pc_amount]))

    for aa in amino_acids:
        for j in range(len(col)):
            if col[j] == aa:
                freq_counts[aa] = freq_counts[aa] + 1 * seq_weights[j]

    for aa in amino_acids:
        freq_counts[aa] = freq_counts[aa] / (sum(seq_weights) + len(amino_acids) * pc_amount)

    return freq_counts


def weighted_gap_penalty(col, seq_weights):
    """ Calculate the simple gap penalty multiplier for the column. If the 
    sequences are weighted, the gaps, when penalized, are weighted 
    accordingly. """

    # if the weights do not match, use equal weight
    if len(seq_weights) != len(col):
        seq_weights = [1.] * len(col)

    gap_sum = 0.
    for i in range(len(col)):
        if col[i] == '-':
            gap_sum += seq_weights[i]

    return 1 - (gap_sum / sum(seq_weights))



def msa_Jensen_Shannon_divergence(col, bg_distr, seq_weights, gap_penalty=True):
    """ Return the Jensen-Shannon Divergence for the column with the background
    distribution bg_distr. sim_matrix is ignored. JSD is the default method."""

    fc = weighted_freq_count_pseudocount(col, seq_weights, PSEUDOCOUNT)

    # if background distribution lacks a gap count, remove fc gap count
    aaList = amino_acids
    if '-' not in bg_distr.keys():
        aaList = [aa for aa in aaList if aa != '-']
        fc = {aa:fc[aa] for aa in aaList}

    if fc.keys() != bg_distr.keys():
        raise ValueError("fc and bg_distr amino acids are not the same.")
        
    p = np.array([fc[aa] for aa in aaList])
    q = np.array([bg_distr[aa] for aa in aaList])
    p = p/sum(p)
    q = q/sum(q)
    r = 0.5 * p + 0.5 * q
    d = 0.
    for i in range(len(p)):
        if r[i] != 0.0:
            if p[i] == 0.0:
                d += q[i] * np.log2(q[i]/r[i])
            elif q[i] == 0.0:
                d += p[i] * np.log2(p[i]/r[i])
            else:
                d += p[i] * np.log2(p[i]/r[i]) + q[i] * np.log2(q[i]/r[i])
    d /= 2.

    if gap_penalty:
        return d * weighted_gap_penalty(col, seq_weights)
    else:
        return d


def multiple_sequence_alignment(fastafile, outputDir=None, nThreads=1, tool='t-coffee', toolOptions=None, verbose=True):
    """
    """
    if outputDir is None:
        outputPath = Path(fastafile).resolve().parent
    else:
        outputPath = Path(outputDir)

    print("outputDir", outputDir)
    fastafileOut = str(outputPath / re.sub(r'(.+)\.f[na]a', r'\1.aln', Path(fastafile).name))
    if tool == 't-coffee':
        cmd = 't_coffee -n_core={:d} -seq "{}"'.format(nThreads, fastafile)
    elif tool == 'mafft':
        if not toolOptions:
            toolOptions = '--globalpair'
        cmd = 'mafft {} --clustalout --maxiterate 1000 --thread {:d} "{}" > "{}"'\
              .format(toolOptions, nThreads, fastafile, fastafileOut)
    else:
        cmd = None
    if verbose:
        print("MSA tool:", tool)
        print("fastafileOut:", fastafileOut)
        print("cmd:", cmd)
#     cmd = shlex.split(cmd)
    stderr = subprocess.STDOUT if verbose else subprocess.DEVNULL
    try:
        cmd_output = subprocess.check_output(cmd, stderr=stderr, cwd=str(outputPath), shell=True)
        cmd_output = re.sub(r'\\n','\n',str(cmd_output))
        if verbose: print(cmd_output)
    except CalledProcessError as err:
        print(err.output)
        print(err.stderr)
        if tool == 'mafft':
            print("Alignment had an error, we try the same with the --anysymbol option")
            cmd = 'mafft --anysymbol {} --clustalout --maxiterate 1000 --thread {:d} "{}" > "{}"'\
                  .format(toolOptions, nThreads, fastafile, fastafileOut)
            cmd_output = subprocess.check_output(cmd, stderr=stderr, cwd=str(outputPath), shell=True)
            cmd_output = re.sub(r'\\n','\n',str(cmd_output))
            if verbose: print(cmd_output)
        else:
            raise

    with open(fastafileOut, 'r') as f:
        MSA_result = f.read()

    return MSA_result, fastafileOut
