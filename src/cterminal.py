# coding: utf-8

# Matplotlib default backend using LaTeX
# Note: if importing the cterminal module into another module, there will be
# conflict between the different matplotlib settings.
import matplotlib_options
matplotlib_options.define_matplotlib_backend_options(backend='agg')

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.image as mpimg
import matplotlib.patheffects
from matplotlib.font_manager import FontProperties
from IPython.core.display import display, HTML

import pdfkit
from wand.image import Image as wandImage
from wand.display import display as wandDisplay
from wand.color import Color as wandColor

import seaborn

import numpy as np
from numpy import exp
from numpy import log2
import pandas as pd
import collections
from collections import Counter
import scipy
import copy
import bs4
from functools import reduce
import operator
import json
import argparse
from statsmodels.sandbox.stats.multicomp import multipletests

import subprocess
import shlex
import itertools
import os.path
from pathlib import Path
import re
import gzip
import urllib
import gc
import time

import Bio.SeqIO
import Bio
from Bio.SeqFeature import FeatureLocation, ExactPosition

# mwTools
from mwTools.general import sliding_window_array
from mwTools.general import sliding_window_string

from mwTools.id import extract_refseq_accession_id
from mwTools.id import extract_seq_id
from mwTools.id import extract_mpn_id
from mwTools.bio import pretty_print_mRNA
from mwTools.bio import read_assemblySummary_file
from mwTools.bio import import_genome_gbff_file
from mwTools.bio import extract_compressed_genome_file
from mwTools.bio import extract_codons_list
from mwTools.bio import build_refCodonTable
from mwTools.bio import sort_codon_index
from mwTools.plot import get_divergent_color_map
from mwTools.clustering import clustering2
from mwTools.clustering import parse_cluster_file
from mwTools.clustering import extract_seqIdList_from_cluster




idx = pd.IndexSlice

# print(plt.style.available)
seaborn.set_style("darkgrid")
colorAxis = '0.8'


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


#@profile
def write_dataframe(df, filename, dfFormat='csv'):
    if dfFormat == 'csv':
        df.to_csv(filename)
    elif dfFormat == 'json':
        df.to_json(filename)


colorBackground1 = '0.85'
colorSmallN = colorBackground1
colorHeatmapLine = 'white'
vmax = 3


def get_line_width(figsizeCoeff):
    return figsizeCoeff*1.5


# # Enrichment analysis of a.a. at the C-terminal

cTerminalSize = 20
nTerminalSize = cTerminalSize
family_wise_FDR = 0.05

# Group amino acids by physicochemical properties
aaTable = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'U', 'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V']
# Drop the Selenocysteine (very rare amino acid)
aaTable = ['R', 'K', 'H', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'G', 'P', 'A', 'V', 'I', 'L', 'M', 'F', 'W', 'Y']

pvalueThresholds = [[1.1,""], [1e-2,"**"], [1e-3,"***"], [1e-4,"****"]]
pvalueThresholdMask = 0.001
cmap = get_divergent_color_map(name="RdBu_r")

# ## Prepare sequences

# Length of the 3'UTR in codons to extract for each coding sequence.
_length3UTR = 10


#@profile
def prepare_all_CDS(genomeBio, genome_accession, genome_GBFF_file, verbose=0, addLocus=False):
    
    species_name = genomeBio.annotations['organism']
    species_name = species_name if species_name is not None else ''
    
    CDSList = []
    if verbose >= 1: print("len(genomeBio.features):", len(genomeBio.features))
    for feature in genomeBio.features:
        if feature.type == 'CDS':
            if verbose >= 2:
                print(feature.qualifiers,'\n\n',feature.type,'\n',feature.location,'\n\n',
                      feature.extract(genomeBio),'\n')
            
            refSeqProteinId = feature.qualifiers.get('protein_id')
            refSeqProteinId = refSeqProteinId[0] if refSeqProteinId is not None else None
            product = feature.qualifiers.get('product')
            product = product[0] if product is not None else None
            
            # Test for not valid CDS annotation
            if refSeqProteinId is None:
                if verbose>=1:
                    print("Error prepare_all_CDS: discarded CDS annotation.")
                    print(feature.qualifiers)
                
            else:

                # Extend sequence to the 3' end
                # Note: for the FeatureLocation object, we use Python-like integer positions, contrariwise to the Genbank format
                if feature.location.strand == +1:
                    location3UTR = FeatureLocation(feature.location.end,
                                                   feature.location.end + 3*_length3UTR,
                                                   strand=feature.location.strand)
                elif feature.location.strand == -1:
                    location3UTR = FeatureLocation(feature.location.start - 3*_length3UTR,
                                                   feature.location.start,
                                                   strand=feature.location.strand)
                if verbose >= 2: print(location3UTR)

                DNASeqBio = feature.extract(genomeBio)
                if DNASeqBio is None:
                    if verbose >= 1: print("ERROR: DNASeqBio is null", DNASeqBio, "feature", feature)
                #DNASeqCodons = tuple(extract_codons_list(str(DNASeqBio.seq)))
                DNASeq = str(DNASeqBio.seq)
                if DNASeqBio is None:
                    if verbose >= 1: print("ERROR: DNASeq is null", DNASeq, "feature", feature)
                stopCodon = DNASeq[-3:]
                DNA3UTRSeqBio = location3UTR.extract(genomeBio)
                #DNA3UTRSeqCodons = tuple(extract_codons_list(str(DNA3UTRSeqBio.seq)))
                DNA3UTRSeq = str(DNA3UTRSeqBio.seq)
                if verbose>=2:
                    print('DNASeqBio:\n',DNASeqBio.seq,'\n')
                    print('DNASeqBio 3\'UTR extension:\n',DNA3UTRSeqBio.seq,' length:',len(DNA3UTRSeqBio.seq),'\n')
                    print('DNASeqBio last 9 bps and 3\'UTR:\n', 
                          genomeBio.seq[feature.location.end - 9 : feature.location.end + 3*_length3UTR] \
                          if feature.location.strand == +1 else \
                          genomeBio.seq[feature.location.start - 1 - 3*_length3UTR : feature.location.start + 9 ].reverse_complement()
                         )

                proteinSeq = feature.qualifiers.get('translation')
                proteinSeq = proteinSeq[0] if proteinSeq is not None else None
                codonTableBio = feature.qualifiers.get('transl_table')
                codonTableBio = codonTableBio[0] if codonTableBio is not None else None
                geneName = feature.qualifiers.get('gene')
                geneName = geneName[0] if geneName is not None else None
                locusTag = feature.qualifiers.get('locus_tag')
                locusTag = locusTag[0] if locusTag is not None else None
                proteinGI = feature.qualifiers.get('db_xref')
                proteinGI = int(re.sub(r'GI:','',proteinGI[0])) if proteinGI is not None else None

                if False:
                    CDSobject = [refSeqProteinId, species_name, geneName, locusTag, product,
                                 genome_accession, genome_GBFF_file, feature, DNASeqBio,
                                 stopCodon, DNA3UTRSeqBio, codonTableBio, proteinSeq]
                    CDSobjectNames = ['refSeqProteinId','species_name',
                                      'geneName','locusTag','product','genome_accession','genomeFile',
                                      'featureBio','DNASeqBio','stopCodon','DNA3UTRSeqBio','codonTableBio','proteinSeq']
                else:
                    CDSobject = [refSeqProteinId, proteinGI, species_name, genome_accession,
                                 DNASeq, stopCodon, DNA3UTRSeq, codonTableBio, proteinSeq]
                    CDSobjectNames = ['refSeqProteinId', 'proteinGI', 'species_name', 'genome_accession',
                                      'DNASeq','stopCodon','DNA3UTRSeq','codonTableBio','proteinSeq']
                if addLocus:
                    CDSobject.append(locusTag)
                    CDSobjectNames.append('locusTag')

                CDSList.append(CDSobject)
                if verbose >= 2: print("\n\n")
            
            
    allCDSDf = pd.DataFrame(CDSList, columns=CDSobjectNames)
    allCDSDf.set_index('refSeqProteinId', drop=False, inplace=True)
    # Remove duplicates with the same accession id
    # There is one case for example in Mycoplasma genitalium, WP_011113499.1
    allCDSDf = allCDSDf[~allCDSDf.index.duplicated()]
    
    return allCDSDf



# ## Generate fasta file for protein sequences

def generate_protein_seq_fasta(genomeBio, verbose=False):

    species_name = genomeBio.annotations['organism']
    species_name = species_name if species_name is not None else ''
    
    fastaString = ''
    for feature in genomeBio.features:
        if feature.type == 'CDS':
            refSeqProteinId = feature.qualifiers.get('protein_id')
            if refSeqProteinId is None:
                if verbose:
                    print("Error generate_protein_seq_fasta: discarded CDS annotation.")
                    print(feature.qualifiers)
            else:
                refSeqProteinId = refSeqProteinId[0]
            
                proteinSeq = feature.qualifiers.get('translation')
                if proteinSeq is None:
                    if verbose:
                        print("Error: cannot retrieve protein sequence from the translation qualifier.")
                        print(feature.qualifiers)
                else:
                    proteinSeq = proteinSeq[0]

                    geneName = feature.qualifiers.get('gene')
                    geneName = geneName[0] if geneName is not None else ''
                    product = feature.qualifiers.get('product')
                    product = product[0] if product is not None else ''
                    locusTag = feature.qualifiers.get('locus_tag')
                    locusTag = locusTag[0] if locusTag is not None else ''

                    fastaString += ('>' + refSeqProteinId + ' '
                                    '|' + geneName +
                                    '|' + locusTag +
                                    '|' + product +
                                    '|' + species_name +
                                    '\n' + proteinSeq + '\n')
    
    return fastaString


# ## Import all protein sequences from fasta file

#@profile
def prepare_allSeq(allProteinSeqBio, species_name):
    def format_seqRecord(seq_record):
        refSeqProteinId = extract_refseq_accession_id(seq_record.id)
        return [species_name, refSeqProteinId, str(seq_record.seq), seq_record]

    allProteinSeqDf = pd.DataFrame(list(map(format_seqRecord, allProteinSeqBio)), columns=['species_name','refSeqProteinId','proteinSeq','seqBio'])
    allProteinSeqDf.set_index('refSeqProteinId', drop=False, inplace=True)
    # Remove duplicates with the same accession id
    # There is one case in Mycoplasma genitalium, WP_011113499.1
    allProteinSeqDf = allProteinSeqDf[~allProteinSeqDf.index.duplicated()]
    return allProteinSeqDf



# ## Cluster analysis of protein sequences in MPN

# We will remove protein sequences that have both a high overall identity and a high
# identity at the C-terminal. The resulting list of sequences will form the non-redundant
# database for our analysis. Perform a cluster analysis of all protein sequences in MPN
# using CD-HIT

def add_clusters_to_dataframe(allProteinSeqDf, cluster_dic, verbose=False):
    """Add the clusters to the dataframe (use multi index to group the rows by cluster)"""

    for key, cluster in cluster_dic.items():
        if verbose: print('Nb of seq: ',len(cluster),key,cluster)
        seqlist = extract_seqIdList_from_cluster(key, cluster_dic, extract_refseq_accession_id,
                                                 allProteinSeqDf, 'refSeqProteinId', match_method="contains")
        allProteinSeqDf.loc[seqlist, 'cluster'] = key

    allProteinSeqDf.set_index(['cluster', 'refSeqProteinId'], inplace=True)
    # The multi index dataframe needs the labels to be sorted for some of the slicing/indexing routines to work.
    allProteinSeqDf = allProteinSeqDf.sort_index(level=0)
    # Add a column with the cterm sequence (for convenience)
    allProteinSeqDf['cterm_seq'] = allProteinSeqDf.apply(lambda row: row.proteinSeq[-cTerminalSize:], axis=1)
    # Add a column with the nterm sequence (for convenience)
    allProteinSeqDf['nterm_seq'] = allProteinSeqDf.apply(lambda row: row.proteinSeq[:nTerminalSize], axis=1)
    return allProteinSeqDf
    

# In each cluster, perform a second clustering analysis on the C-terminal part of sequences.

# Perform a multiple sequence alignment (MSA) of all the sequences in the cluster
def multiple_sequence_alignment_cluster(clusterFile, clusteringFolder, verbose=True):
    # Run T-coffee multiple alignment
    currentFolder = os.getcwd()
    os.chdir(clusteringFolder)
    cmd = 't_coffee -seq "' + os.path.basename(clusterFile) + '"'
    cmd = shlex.split(cmd)
    stderr = subprocess.STDOUT if verbose else subprocess.DEVNULL
    cmd_output = subprocess.check_output(cmd, stderr=stderr)
    cmd_output = re.sub(r'\\n','\n',str(cmd_output))
    fileMSA = open(re.sub(r'(.+).faa', r'\1.aln', clusterFile), 'r')
    MSA_result = fileMSA.read()
    os.chdir(currentFolder)
    return MSA_result
            
#@profile
def cterm_clustering(allProteinSeqDf, clusteringFolder, cterm_identity_threshold=0.9, cTerminalSize=20, verbose=True):

    # cwd = os.getcwd()
    # os.chdir(refSeqFolder)
    for cluster_name, cluster_seqs in allProteinSeqDf.groupby(level='cluster'):

        if len(cluster_seqs) > 1:
            # For each cluster, run a second round of clustering on the c-terminal subsequences
            if verbose:
                print('\n### ', cluster_name, ' size: ',len(cluster_seqs))
                print('clustering C-terminal subsequences (CD-HIT)')
                print('identity threshold:', cterm_identity_threshold)
                print('c-terminal size:', cTerminalSize)

            # Write cluster sequences in fasta file
            clusterFilename = os.path.join(clusteringFolder, cluster_name + '.faa')
            Bio.SeqIO.write(cluster_seqs['seqBio'], clusterFilename, "fasta")

            # Write cluster c-terminal subsequences in fasta file
            cterm_seq_list = []
            for seqBio in cluster_seqs['seqBio']:
                if verbose: print(seqBio.id[:19],': ...',seqBio.seq[-cTerminalSize:])
                seqBioCopy = copy.copy(seqBio)
                seqBioCopy.seq = seqBioCopy.seq[-cTerminalSize:]
                cterm_seq_list.append( seqBioCopy )
            ctermFilename = os.path.join(clusteringFolder, cluster_name + '.cterm.faa')
            Bio.SeqIO.write(cterm_seq_list, ctermFilename, "fasta")

            # Run cluster analysis on c-terminal subsequences
            clusteringOutputFile = ctermFilename + '.cluster'
            cmd = 'cd-hit -i ' + ctermFilename + ' -o ' + clusteringOutputFile + ' -c ' + str(cterm_identity_threshold) + ' -n 3'
            cmd = shlex.split(cmd)
            cmd_output = subprocess.check_output(cmd)
            cmd_output = re.sub(r'\\n','\n',str(cmd_output))
            #if verbose: print(cmd_output)

            # Parse output of the clustering
            cterm_cluster_dic = parse_cluster_file(clusteringOutputFile + '.clstr')
            if verbose: print('C-terminal clustering results:')

            # Add the cterminal cluster to the sequences in the dataframe
            for cterm_key, cterm_cluster in cterm_cluster_dic.items():
                if verbose: print('    ', cterm_key, 'nb of seq: ',len(cterm_cluster), cterm_cluster)
                representative_seq_list = [seqtuple[1] for seqtuple in cterm_cluster]
                seqlist = extract_seqIdList_from_cluster(cterm_key, cterm_cluster_dic, extract_refseq_accession_id,
                                                         allProteinSeqDf, 'refSeqProteinId', match_method="contains")
                allProteinSeqDf.loc[pd.IndexSlice[:,seqlist],'cluster_cterm'] = cterm_key
                # Add a boolean True for representative sequence of the cluster, false otherwise
                # We will be able to filter easily the dataframe with this column to get the
                # non-redundant sequence list.
                allProteinSeqDf.loc[pd.IndexSlice[:,seqlist],'non-redundant'] = representative_seq_list

            # Compute MSA to examine relevance of the clustering results
            clusterMSA = multiple_sequence_alignment_cluster(os.path.basename(clusterFilename), clusteringFolder, verbose)
            if verbose: print('\nT-coffee Multiple Sequence Alignment: \n', clusterMSA)
            allProteinSeqDf.loc[cluster_name,'MSA'] = clusterMSA

        else:
            # Define all sequences that were originally in a cluster of size 1 as non-redundant
            allProteinSeqDf.loc[cluster_name,'non-redundant'] = True

    # os.chdir(root_folder)
    return allProteinSeqDf



# ## Enrichment analysis of a.a.

#@profile
def count_bulk_aa_frequency(allProteinSeqDf, cTerminalSize, nTerminalSize):
    """Count the frequency of amino acids in all protein sequences excluding the C-termini."""

    if allProteinSeqDf is None:
        return (collections.Counter(), collections.Counter())
    else:
        bulkSeqString = ""
        for seq in allProteinSeqDf.proteinSeq:
            bulkSeqString += seq[nTerminalSize:-cTerminalSize]

        bulkFreqAA = collections.Counter(bulkSeqString)
        print("Total nb of aa: ",sum(bulkFreqAA.values()))

        bulkRelFreqAA = {aa: count/sum(bulkFreqAA.values()) for aa, count in dict(bulkFreqAA).items()}
        bulkRelFreqAA = sorted(bulkRelFreqAA.items(), key=lambda x: x[1])
        
        return (bulkFreqAA, bulkRelFreqAA)

def convert_bulk_aa_freq_to_Df(bulkFreqAA):
    # Wrapping up in a DataFrame (for output to csv file)
    bulkRelFreqAA = {aa: count/sum(bulkFreqAA.values()) for aa, count in dict(bulkFreqAA).items()}
    bulkFreqAADf = pd.DataFrame([dict(bulkFreqAA),dict(bulkRelFreqAA)]).transpose()\
                        .rename(columns={0:'bulk_aa_freq',1:'bulk_aa_relative_freq'})
    return bulkFreqAADf


#@profile
def count_termina_aa_frequency(allProteinSeqDf, cTerminalSize, nTerminalSize):
    """Computing the frequency of amino acids at the C-terminal and comparing to the reference."""

    if allProteinSeqDf is None:
        ctermFreq = [collections.Counter() for AAListPosJ in range(-cTerminalSize,0)]
        ntermFreq = [collections.Counter() for AAListPosJ in range(1,nTerminalSize+1)]
        return (ctermFreq, ntermFreq)
    else:
        # C-terminus
        # The list of aa starts at position -cTerminalSize, ..., -1 (last a.a. before stop codon)
        ctermFreq = [[seq[j] for j in range(-cTerminalSize,0)
                     if len(seq) > cTerminalSize+nTerminalSize]
                     for seq in allProteinSeqDf.proteinSeq]

        # N-terminus
        # The list of aa starts at position +1 (a.a. after N-terminal residue), +2, ...
        ntermFreq = [[seq[j] for j in range(1,nTerminalSize+1)
                     if len(seq) > cTerminalSize+nTerminalSize]
                     for seq in allProteinSeqDf.proteinSeq]

        ctermFreq = list(filter(None,ctermFreq))
        ntermFreq = list(filter(None,ntermFreq))
        ctermFreq = np.array(ctermFreq).T
        ntermFreq = np.array(ntermFreq).T
        print("ctermFreq.shape: ", ctermFreq.shape)
        print("ntermFreq.shape: ", ntermFreq.shape)

        # Building a collection of aa counts at each Cterminal position
        ctermFreq = [collections.Counter(AAListPosJ) for AAListPosJ in ctermFreq]
        ntermFreq = [collections.Counter(AAListPosJ) for AAListPosJ in ntermFreq]
        print("Total nb of aa at pos 0: ",sum(ctermFreq[0].values()))
        print("Total nb of aa at pos 0: ",sum(ntermFreq[0].values()))
        return (ctermFreq, ntermFreq)
    

#@profile
def compute_odds_ratio(bulkFreqAA, ctermFreqAA, cTerminalSize, ntermFreqAA, nTerminalSize,
                       computeMultipleTestsCorrection=True):

    # C-terminus positions: -ctermsize to -1
    # N-terminus positions: +1 to +ntermsize
    # bulk position: 0
    posFromTermina = [list(range(-len(ctermFreqAA), 0)), list(range(1, len(ntermFreqAA) + 1))]
    multiIndex = pd.MultiIndex.from_product([['N', 'bulk', 'C'],
                                            [0] + posFromTermina[0] + posFromTermina[1],
                                            ['count', 'log2OddsRatio', 'oddsRatio', 'pvalue']],
                                            names=['terminus', 'position from terminus', 'observable'])
    oddsRatioDf = pd.DataFrame(index=multiIndex, columns=aaTable)
    oddsRatioDf.sort_index(level=0, inplace=True)
    
    for aa in aaTable:
        oddsRatioDf.loc[('bulk', 0, 'count'), aa] = bulkFreqAA[aa]

    terminalSize = 0
    termSeqAAFreq = 0
    posFromTerminus = []
    for terminus in ['N','C']:
        if terminus == 'C':
            terminalSize = cTerminalSize
            termSeqAAFreq = ctermFreqAA
            posFromTerminus = posFromTermina[0]
        elif terminus == 'N':
            terminalSize = nTerminalSize
            termSeqAAFreq = ntermFreqAA
            posFromTerminus = posFromTermina[1]
            
        for aa in aaTable:
            NrefA    = bulkFreqAA[aa]
            NrefNotA = sum(bulkFreqAA.values()) - bulkFreqAA[aa]
            
            for j in range(0, len(posFromTerminus)):

                NjA      = termSeqAAFreq[j][aa]
                NjNotA   = sum(termSeqAAFreq[j].values()) - termSeqAAFreq[j][aa]

                pos = posFromTerminus[j]
                # Note: the following manual calculus is correct but does not work when some N is zero!
                #oddsRatioDf.loc[pos,'oddsRatio'][aa] = (NjA/NjNotA)/(NrefA/NrefNotA)
                #oddsRatioDf.loc[pos,'log2OddsRatio'][aa] = log2((NjA/NjNotA)/(NrefA/NrefNotA))
                
                oddsRatioDf.loc[(terminus, pos, 'count'), aa] = NjA
                contingencyTable = [[NjA, NrefA], [NjNotA, NrefNotA]]
                oddsRatioScipy, pvalue = scipy.stats.fisher_exact(contingencyTable, alternative='two-sided')
                if NjA == 0 or NjNotA == 0:
                    oddsRatioDf.loc[(terminus, pos, 'oddsRatio'), aa] = None
                    oddsRatioDf.loc[(terminus, pos,'log2OddsRatio'), aa] = None
                    oddsRatioDf.loc[(terminus, pos,'pvalue'), aa] = None
                else:
                    oddsRatioDf.loc[(terminus, pos, 'oddsRatio'), aa] = oddsRatioScipy
                    oddsRatioDf.loc[(terminus, pos, 'log2OddsRatio'), aa] = log2(oddsRatioScipy)
                    oddsRatioDf.loc[(terminus, pos, 'pvalue'), aa] = pvalue

                #print("j=",j," aa=",aa," NjA=",NjA," NjNotA=",NjNotA," NrefA=",NrefA," NrefNotA=",NrefNotA," logOddsRatio[aa][j]=",logOddsRatio[aa][j])
               
    if computeMultipleTestsCorrection:
        dfList = []
        for terminus in ['N', 'C']:
            # Multiple test correction within the biases of C-terminal at all positions for all a.a.
            df = oddsRatioDf.xs(terminus, level='terminus', drop_level=False).copy()
            df = df.xs('pvalue', level='observable', drop_level=False)
            # We serialize all the values, and drop the NaN
            df2 = df.stack().dropna().copy()
            reject, pvals_corrected, alphacSidak, alphacBonf = \
                multipletests(df2.values, alpha=family_wise_FDR, method='fdr_bh',
                              is_sorted=False, returnsorted=False)
            df2 = pd.DataFrame(reject, index=df2.index)
            # Stack again the values before merging
            df2 = df2.unstack()
            df2.columns = df2.columns.droplevel(0)
            df2 = df2.rename(index={'pvalue':'BH_multiple_tests'})
            dfList.append(df2)
        oddsRatioDf = pd.concat([oddsRatioDf] + dfList, axis=0, sort=True)
        oddsRatioDf.sort_index(inplace=True)
     
    return oddsRatioDf

# #### Plot the results on a color mesh


def select_terminal_positions_in_index(plotData, terminus):
    if terminus == 'C':
        plotData = plotData.loc[plotData.index.get_level_values(level='position from terminus') < 0]
    elif terminus == 'N':
        plotData = plotData.loc[plotData.index.get_level_values(level='position from terminus') > 0]

    plotData = plotData.drop([_startCodonVirtualIndex, _stopCodonVirtualIndex],
                             axis=0, level='position from terminus', errors='ignore')
    return plotData


def compute_oddsratio_plot_data(oddsRatioDf, terminus):
    # Select terminus
    plotData = oddsRatioDf
    plotData = plotData.xs(terminus, level='terminus')
    plotData = select_terminal_positions_in_index(plotData, terminus)
    plotData = plotData.xs('log2OddsRatio', level='observable')
    plotData = plotData[plotData.columns].astype(float)
    plotData = plotData.transpose().loc[aaTable]
    return plotData


def compute_oddsratio_mask_data(oddsRatioDf, pvalueThresholds, terminus):
    # Select terminus
    plotData = oddsRatioDf
    plotData = plotData.xs(terminus, level='terminus')
    plotData = select_terminal_positions_in_index(plotData, terminus)
    plotData = plotData.xs('BH_multiple_tests', level='observable')
    # Mask both rejected cases and null values
    plotData = (plotData != True) & (plotData != 'True')
    plotData = plotData.transpose().loc[aaTable]
    return plotData


# #### Plot with pvalue as text annotation in each square

def compute_pvalueAnnotation(oddsRatioDf, pvalueThresholds, terminus):
    # Build a table of text annotations representing pvalue
    # Select terminus
    pvalAnnotTable = oddsRatioDf
    pvalAnnotTable = pvalAnnotTable.xs(terminus, level='terminus')
    pvalAnnotTable = select_terminal_positions_in_index(pvalAnnotTable, terminus)
    pvalAnnotTable = pvalAnnotTable.xs('pvalue', level='observable')
    pvalAnnotTable = pvalAnnotTable[pvalAnnotTable.columns].astype(float)
    pvalAnnotTable = pvalAnnotTable.transpose().loc[aaTable]

    # We create a copy of the data frame with string type (cannot mix types inside data frame columns)
    pvalAnnotTableCopy = pvalAnnotTable.astype(str)
    pvalAnnotTableCopy[:] = ""
    for i in range(0,len(pvalueThresholds)):
        if (i < len(pvalueThresholds)-1):
            condition = (pvalAnnotTable < pvalueThresholds[i][0]) & (pvalueThresholds[i+1][0] < pvalAnnotTable)
            pvalAnnotTableCopy[condition] = pvalueThresholds[i][1]
        else:
            condition = pvalAnnotTable < pvalueThresholds[i][0]
            pvalAnnotTableCopy[condition] = pvalueThresholds[i][1]

    return pvalAnnotTableCopy


def compute_minimum_n_observed_expected(oddsRatio, freq, nRefTot=1e7, verbose=0):
    nRefExp = freq*nRefTot
    if verbose >= 2: print("nRefExp", nRefExp)
    if nRefExp < 100:
        if verbose >= 1: print("size of expected cases in reference is too small")
    if nRefExp > 10000:
        # In order to speed up the computation, we limit the value of nRefTot
        nRefExp = 10000
        nRefTot = int(nRefExp/freq)
        
    nObsMinimum = int(1e12)
    for nObsExp in range(3*int(nRefExp)):
        nObsTot = nObsExp / freq
        contingencyTable = [[oddsRatio*nObsExp, nObsTot - oddsRatio*nObsExp], [freq*nRefTot, (1-freq)*nRefTot]]
        pval = scipy.stats.fisher_exact(contingencyTable, alternative='two-sided')[1]
        if verbose >= 2: print("nObsExp", nObsExp, "pval", pval)
        if pval < pvalueThresholdMask:
            nObsMinimum = nObsExp
            break

    return nObsMinimum
    

def compute_smallN_mask(meanFreq, nObsExp, counts, pvalue, verbose=0):
    # Detect cases for which n expected observation is small and pvalue of bias is larger than 0.01
    # meaning that we cannot be confident that there is no bias.
    # We can choose the limit of nObsExp based on the following criterium: for a large reference observation
    # with frequency of 2%, if we want to detect a odds ratio of 0.5 with pvalue < 0.001, we need
    # an expected nb of observation to be at least 35, i.e. observed nb of observation of 17.
    # Contingency table: [[oddsRatio*nObsExp, nObsTot], [freq*nRefTot, (1-freq)*nRefTot]]
    # We apply this criterium if nObs < nObsExp.
    # Similarly, for an odds ratio of 2, we need nObsExp >= 16.
    # We apply this criterium if nObs > nObsExp.
    nObsExpMinEnrichment = compute_minimum_n_observed_expected(oddsRatio=2, freq=meanFreq)
    nObsExpMinDepletion = compute_minimum_n_observed_expected(oddsRatio=0.5, freq=meanFreq)
    smallNDf = (
                (
                 ((nObsExp < nObsExpMinDepletion) & (counts < nObsExp)) |
                 ((nObsExp < nObsExpMinEnrichment) & (counts >= nObsExp))
                ) &
                ( (pvalue > pvalueThresholdMask) | pd.isnull(pvalue) )
               )
    smallNDf = smallNDf.T

    return smallNDf


def compute_smallN_mask_data(oddsRatioDf, pvalueThresholds, terminus):
    
    bulkFreq = oddsRatioDf.xs('bulk', level='terminus').xs('count', level='observable').loc[0].astype(np.int64)
    # bulkFreq = oddsRatioDf[oddsRatioDf['terminus'] == 'bulk'].drop('terminus', axis=1).xs('count', level='observable')
    NrefTot = bulkFreq.sum()
    bulkFreq = bulkFreq / NrefTot
    # I have no idea why here we have to do twice mean. Dimensions of the dataframe???
    meanFreq = bulkFreq.mean()
    if type(meanFreq) is not float:
        meanFreq = meanFreq.mean()
    
    counts = oddsRatioDf
    counts = counts.xs(terminus, level='terminus')
    counts = select_terminal_positions_in_index(counts, terminus)
    counts = counts.xs('count', level='observable')
    counts = counts.apply(pd.to_numeric)

    countsTot = counts.apply(lambda row: row.sum(), axis=1)
    nObsExp = countsTot.to_frame().dot(bulkFreq.to_frame().T)
    nObsExp = nObsExp.apply(pd.to_numeric)
    
    pvalue = oddsRatioDf
    pvalue = pvalue.xs(terminus, level='terminus')
    pvalue = select_terminal_positions_in_index(pvalue, terminus)
    pvalue = pvalue.xs('pvalue', level='observable')
    pvalue = pvalue.apply(pd.to_numeric)
    
    # print(meanFreq)
    # print(nObsExp)
    # print(counts)
    # print(pvalue)
    smallNDf = compute_smallN_mask(meanFreq, nObsExp, counts, pvalue)
    return smallNDf.loc[aaTable]



def write_latex_legendPvalueAnnotation(pvalueThresholds):
    legendPvalueAnnotation = \
        (
            "two-tails Fisher exact test \n" +
            r"\begin{align*} " +
            "\\\\".join(["\\text{{{:>4}}}: p &< \\num[scientific-notation=true,round-precision=1,round-mode=figures]{{{:.12f}}}"
                        .format(annotation,pval)
                         for [pval, annotation] in pvalueThresholds[1:]]) +
            r" \end{align*}"
        )
    return legendPvalueAnnotation


#@profile
def plot_aa_composition_map(plotData, maskData, pvalAnnotTable, maskDataSmallN, terminus, speciesName, plotPvalueAnnot=False, width=5):
    
    linewidth1 = 0.5
    aspectratio = 0.6
    figsizeCoeff = 0.5
    figsize = (width, (width/aspectratio))

    fig, (ax,cbar_ax) = plt.subplots(1, 2, figsize=figsize)
    # main axes
    ax.set_position([0, 0, 1, 1])
    ax.set_facecolor(colorBackground1)

    # Create axes for the colorbar for more precise positioning
    cbarOrientation = 'vertical'
    cbar_length = 0.25
    if cbarOrientation == 'vertical':
        cbar_aspect = 10
        cbar_ax.set_position([0.05, -0.3 - cbar_length, cbar_length/cbar_aspect, cbar_length])

    cbarLabel = '$\log_2$(odds ratio)'
    ax = seaborn.heatmap(plotData, square=True, mask=maskData, ax=ax, cmap=cmap,
                         cbar_ax=cbar_ax,
                         cbar_kws=dict(label=cbarLabel, orientation=cbarOrientation),
                         vmin=-vmax, vmax=vmax,
                         xticklabels=True, yticklabels=True,
                         linewidth=linewidth1, linecolor=colorHeatmapLine
                         )
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        # spine.set_color('0.8')
        spine.set_linewidth(0.4)
    tickLength = FontProperties(size='small').get_size()/4
    cbar_ax.tick_params(axis='y', length=tickLength, color=colorAxis)
    tickLabelsPad = 0
    ax.tick_params(axis='x', labelbottom=True, labelsize='small', pad=tickLabelsPad)
    ax.tick_params(axis='y', labelleft=True, labelright=True, labelsize='small', pad=tickLabelsPad)
    if len(ax.xaxis.get_ticklabels()[0].get_text()) > 2:
        ax.xaxis.set_tick_params(rotation=90)
    ax.yaxis.set_tick_params(rotation=0)
    family = 'Liberation Mono'
    ticks_font = FontProperties(family=family, size='small')
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(ticks_font)
        tick.set_bbox(dict(pad=0, facecolor='none', edgecolor='none'))

    if plotPvalueAnnot:
        ny = pvalAnnotTable.shape[0]
        #           !!!!!!!!!!!!!!!!!!!!!
        #           we had to correct the indexing of the pvalue annotation, probably because seaborn changed
        #           the axes coordinates
        # OLD VERSION:
        # for (i,j), value in np.ndenumerate(pvalAnnotTable.values):
        #     ax.annotate(value, xy=(j + 0.5, ny - i - 0.5 - 0.35),   # Note: the coordinates have to be transposed (j,ny-i)!!!
        #                 ha='center', va='center', fontsize=figsizeCoeff*8,
        #                 path_effects=[matplotlib.patheffects.withStroke(linewidth=1.5,foreground="w")])
        for (i,j), value in np.ndenumerate(pvalAnnotTable.values):
            ax.annotate(value, xy=(j + 0.5, i + 0.5 + 0.3),   # Note: the coordinates have to be transposed (j,ny-i)!!!
                        ha='center', va='center', fontsize=figsizeCoeff*7.5,
                        path_effects=[matplotlib.patheffects.withStroke(linewidth=1.5,foreground="w")])
        
    ny = maskDataSmallN.shape[0]
    hatch = '////' if figsizeCoeff > 1.1 else '///'
    for (i,j), value in np.ndenumerate(maskDataSmallN.values):
        if value:
            # ax.add_patch(matplotlib.patches.Rectangle((j, ny - 1 - i), 1, 1,
            ax.add_patch(matplotlib.patches.Rectangle((j, i), 1, 1,
                                                      edgecolor='w', facecolor=colorSmallN, hatch=hatch,
                                                      linewidth=linewidth1))

    ax.set_xlabel('position (' + terminus + '-terminal)')
    # legendPvalueAnnotation = write_latex_legendPvalueAnnotation(pvalueThresholds)
    # ax.annotate(legendPvalueAnnotation,
    #             xy=(1.0,0.1), xycoords='figure fraction', xytext=(-15, 0), textcoords='offset points',
    #             bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.3),
    #             ha='right', va='bottom', fontsize='small')

    title = terminus + '-terminal, ' + speciesName
    # ax.set_title(title)
    #fig.tight_layout()
    
    return fig



# ## Enrichment analysis of codons

#@profile
def count_bulk_codon_frequency(allCDSDf_nr, cTerminalSize, nTerminalSize):
    letters = ["T","C","A","G"]
    codonList = set([l1+l2+l3 for l1 in letters for l2 in letters for l3 in letters])
    
    if allCDSDf_nr is None:
        return (collections.Counter(), {aa: 0.0 for aa in codonList})
    else:
        bulkCodonCounter = collections.Counter()
        for DNAseq in allCDSDf_nr['DNASeq']:
            codonSeq = tuple(extract_codons_list(DNAseq))
            # bulkCodonList = itertools.chain(bulkCodonList, codonSeq[nTerminalSize:-cTerminalSize])
            bulkCodonCounter += collections.Counter(codonSeq[nTerminalSize:-cTerminalSize])
        # bulkCodonList = tuple(bulkCodonList)

        bulkFreqCodon = bulkCodonCounter
        print("Total nb of codons: ",sum(bulkFreqCodon.values()))
        bulkRelFreqCodon = {aa: count/sum(bulkFreqCodon.values()) for aa, count in dict(bulkFreqCodon).items()}

        # Note: in E. coli we have 62 codons represented in the bulk, which are the 64 possible codons - 3 stop codons,
        # with the exception of stop codon TGA which is represented 3 times in all coding sequences.

        # We add the missing codons in the list with a frequency of 0.
        print(set(bulkFreqCodon.keys()))
        for missingCodon in codonList - set(bulkFreqCodon.keys()):
            bulkFreqCodon[missingCodon] = 0
            bulkRelFreqCodon[missingCodon] = 0.0

        return (bulkFreqCodon, bulkRelFreqCodon)

def convert_bulk_codon_freq_to_Df(bulkFreqCodon):
    # Wrapping up in a DataFrame (for output to csv file)
    bulkRelFreqCodon = {aa: count/sum(bulkFreqCodon.values()) for aa, count in dict(bulkFreqCodon).items()}
    bulkFreqCodonDf = pd.DataFrame([dict(bulkFreqCodon),dict(bulkRelFreqCodon)]).transpose()                                  .rename(columns={0:'bulk_codon_freq',1:'bulk_codon_relative_freq'})
    return bulkFreqCodonDf
    

#@profile
def count_termina_codon_frequency(allCDSDf_nr, cTerminalSize, nTerminalSize):
    
    if allCDSDf_nr is None:
        ctermFreq = [collections.Counter() for codonListPosJ in range(-cTerminalSize-1,0)]
        ntermFreq = [collections.Counter() for codonListPosJ in range(0,nTerminalSize+1)]
        return (ctermFreq, ntermFreq)
    else:
        # C-terminus
        # The list of codons starts at position -cTerminalSize, ..., -1
        # Important: NOTE THAT THE LAST POSITION OF THE LIST ctermFreq[-1] CORRESPONDS TO THE STOP CODON (not the last a.a.)
        ctermFreq = [[list(extract_codons_list(DNAseq))[j] for j in range(-cTerminalSize - 1, 0)
                     if len(DNAseq) > 3*(cTerminalSize + nTerminalSize)]
                     for DNAseq in allCDSDf_nr['DNASeq']]

        # N-terminus
        # The list of aa starts at position 0 (start codon N-terminal residue), then +1, +2, ...
        # Important: NOTE THAT THE FIRST POSITION OF THE LIST ntermFreq[0] CORRESPONDS TO THE START CODON (methionine(s))
        ntermFreq = [[list(extract_codons_list(DNAseq))[j] for j in range(0, nTerminalSize + 1)
                     if len(DNAseq) > 3*(cTerminalSize + nTerminalSize)]
                     for DNAseq in allCDSDf_nr['DNASeq']]

        ctermFreq = list(filter(None,ctermFreq))
        ntermFreq = list(filter(None,ntermFreq))
        ctermFreq = np.array(ctermFreq).T
        ntermFreq = np.array(ntermFreq).T
        print("ctermFreq.shape: ", ctermFreq.shape)
        print("ntermFreq.shape: ", ntermFreq.shape)

        # Building a collection of codons counts at each Cterminal position
        ctermFreq = [collections.Counter(codonListPosJ) for codonListPosJ in ctermFreq]
        ntermFreq = [collections.Counter(codonListPosJ) for codonListPosJ in ntermFreq]

        # We add the missing codons in the counters with a frequency of 0.
        def add_missing_codons(counter):
            letters = ["T","C","A","G"]
            codonList = set([l1+l2+l3 for l1 in letters for l2 in letters for l3 in letters])
            for missingCodon in codonList - set(counter):
                counter[missingCodon] = 0

        for counterPosJ in ctermFreq:
            add_missing_codons(counterPosJ)
        for counterPosJ in ntermFreq:
            add_missing_codons(counterPosJ)

        print("Total nb of codons at pos 0: ",sum(ctermFreq[0].values()))
        print("Total nb of codons at pos 0: ",sum(ntermFreq[0].values()))
        return (ctermFreq, ntermFreq)
    

_startCodonVirtualIndex = int(-1e6)
_stopCodonVirtualIndex = int(1e6)


#@profile
def compute_odds_ratio_codons(bulkFreqCodon, ctermFreqCodon, cTerminalSize, ntermFreqCodon, nTerminalSize,
                              computeMultipleTestsCorrection=True, verbose=False):

    letters = ["T","C","A","G"]
    codonList = set([l1+l2+l3 for l1 in letters for l2 in letters for l3 in letters])
    
    # Use integer position for start and stop as index with arbitrary numbers (-1e6 and 1e6).
    # Use 0 as position for bulk.
    posFromTerminaDataFrame = [[_startCodonVirtualIndex] + list(range(1, len(ntermFreqCodon))),
                               list(range(-(len(ctermFreqCodon)-1), 0)) + [_stopCodonVirtualIndex],
                               [0]]   
    multiIndex = pd.MultiIndex.from_product([['N', 'bulk', 'C'],
                                             [val for sublist in posFromTerminaDataFrame for val in sublist],
                                             ['terminus','count','log2OddsRatio','oddsRatio','pvalue']
                                             ],
                                            names=['terminus','position from terminus','observable'])
    oddsRatioDf = pd.DataFrame(index=multiIndex, columns=codonList)
    oddsRatioDf.sort_index(level=0, inplace=True)
   
    for codon in codonList:
        oddsRatioDf.loc[('bulk', 0, 'count'), codon] = bulkFreqCodon[codon]

    terminalSize = 0
    termCodonFreq = 0
    posFromTerminusDataFrame = []
    for terminus in ['N','C']:
        if terminus == 'C':
            terminalSize = cTerminalSize
            termCodonFreq = ctermFreqCodon
            posFromTerminusDataFrame = posFromTerminaDataFrame[1]
        elif terminus == 'N':
            terminalSize = nTerminalSize
            termCodonFreq = ntermFreqCodon
            posFromTerminusDataFrame = posFromTerminaDataFrame[0]
            
        for codon in codonList:
            NrefA    = bulkFreqCodon[codon]
            NrefNotA = sum(bulkFreqCodon.values()) - bulkFreqCodon[codon]
            
            for j in range(0, len(termCodonFreq)):

                NjA      = termCodonFreq[j][codon]
                NjNotA   = sum(termCodonFreq[j].values()) - termCodonFreq[j][codon]

                pos = posFromTerminusDataFrame[j]
                # Note: the following manual calculus is correct but does not work when some N is zero!
                #oddsRatioDf.loc[pos,'oddsRatio'][aa] = (NjA/NjNotA)/(NrefA/NrefNotA)
                #oddsRatioDf.loc[pos,'log2OddsRatio'][aa] = log2((NjA/NjNotA)/(NrefA/NrefNotA))
                                
                oddsRatioDf.loc[(terminus, pos, 'count'), codon] = NjA
                contingencyTable = [[NjA, NrefA], [NjNotA, NrefNotA]]
                oddsRatioScipy, pvalue = scipy.stats.fisher_exact(contingencyTable, alternative='two-sided')
                if NjA == 0 or NjNotA == 0:
                    oddsRatioDf.loc[(terminus, pos, 'oddsRatio'), codon] = None
                    oddsRatioDf.loc[(terminus, pos, 'log2OddsRatio'), codon] = None
                    oddsRatioDf.loc[(terminus, pos, 'pvalue'), codon] = None
                else:
                    oddsRatioDf.loc[(terminus, pos, 'oddsRatio'), codon] = oddsRatioScipy
                    oddsRatioDf.loc[(terminus, pos, 'log2OddsRatio'), codon] = log2(oddsRatioScipy)
                    oddsRatioDf.loc[(terminus, pos, 'pvalue'), codon] = pvalue

                if verbose and j==0:
                    print("terminus=",terminus,"j=",j," codon=",codon," NjA=",NjA," NjNotA=",NjNotA," NrefA=",NrefA," NrefNotA=",NrefNotA," logOddsRatio[codon][j]=",log(oddsRatioScipy))

    if computeMultipleTestsCorrection:
        dfList = []
        for terminus in ['N', 'C']:
            # Multiple test correction within the biases of C-terminal at all positions for all a.a.
            df = oddsRatioDf.xs(terminus, level='terminus', drop_level=False).copy()
            df = df.xs('pvalue', level='observable', drop_level=False)
            # We serialize all the values, and drop the NaN
            df2 = df.stack().dropna().copy()
            reject, pvals_corrected, alphacSidak, alphacBonf = \
                multipletests(df2.values, alpha=family_wise_FDR, method='fdr_bh',
                              is_sorted=False, returnsorted=False)
            df2 = pd.DataFrame(reject, index=df2.index)
            # Stack again the values before merging
            df2 = df2.unstack()
            df2.columns = df2.columns.droplevel(0)
            df2 = df2.rename(index={'pvalue':'BH_multiple_tests'})
            dfList.append(df2)
        oddsRatioDf = pd.concat([oddsRatioDf] + dfList, axis=0, sort=True)
        oddsRatioDf.sort_index(inplace=True)

    return oddsRatioDf



# ### Draw codon biases table using HTML

# `  |  U      |  C      |  A      |  G      |
# --+---------+---------+---------+---------+--
# U | UUU F   | UCU S   | UAU Y   | UGU C   | U
# U | UUC F   | UCC S   | UAC Y   | UGC C   | C
# U | UUA L   | UCA S   | UAA Stop| UGA Stop| A
# U | UUG L(s)| UCG S   | UAG Stop| UGG W   | G
# --+---------+---------+---------+---------+--
# C | CUU L   | CCU P   | CAU H   | CGU R   | U
# C | CUC L   | CCC P   | CAC H   | CGC R   | C
# C | CUA L   | CCA P   | CAA Q   | CGA R   | A
# C | CUG L(s)| CCG P   | CAG Q   | CGG R   | G
# --+---------+---------+---------+---------+--
# A | AUU I(s)| ACU T   | AAU N   | AGU S   | U
# A | AUC I(s)| ACC T   | AAC N   | AGC S   | C
# A | AUA I(s)| ACA T   | AAA K   | AGA R   | A
# A | AUG M(s)| ACG T   | AAG K   | AGG R   | G
# --+---------+---------+---------+---------+--
# G | GUU V   | GCU A   | GAU D   | GGU G   | U
# G | GUC V   | GCC A   | GAC D   | GGC G   | C
# G | GUA V   | GCA A   | GAA E   | GGA G   | A
# G | GUG V(s)| GCG A   | GAG E   | GGG G   | G`


def compute_codon_oddsratio_plot_data(plotData, terminus):
    # Select terminus
    plotData = plotData.xs(terminus, level='terminus')
    plotData = select_terminal_positions_in_index(plotData, terminus)
    plotData = plotData.xs('log2OddsRatio', level='observable')
    plotData = plotData[plotData.columns].astype(float)
    plotData = plotData.transpose()
    return plotData


def compute_codon_pvalueAnnotation(oddsRatioDfCodon, pvalueThresholds, terminus):
    # Select terminus
    pvalAnnotTable = oddsRatioDfCodon.xs(terminus, level='terminus')
    pvalAnnotTable = select_terminal_positions_in_index(pvalAnnotTable, terminus)
    
    # Build a table of text annotations representing pvalue
    pvalAnnotTable = pvalAnnotTable.xs('pvalue', level='observable')
    pvalAnnotTable = pvalAnnotTable[pvalAnnotTable.columns].astype(float)
    pvalAnnotTable = pvalAnnotTable.transpose()

    # We create a copy of the data frame with string type (cannot mix types inside data frame columns)
    pvalAnnotTableCopy = pvalAnnotTable.astype(str)
    pvalAnnotTableCopy[:] = ""
    for i in range(0,len(pvalueThresholds)):
        if (i < len(pvalueThresholds)-1):
            condition = (pvalAnnotTable < pvalueThresholds[i][0]) & (pvalueThresholds[i+1][0] < pvalAnnotTable)
            pvalAnnotTableCopy[condition] = pvalueThresholds[i][1]
        else:
            condition = pvalAnnotTable < pvalueThresholds[i][0]
            pvalAnnotTableCopy[condition] = pvalueThresholds[i][1]

    return pvalAnnotTableCopy





def plot_codon_table_dataframe_styling(plotData1Pos, pvalAnnotTable1Pos, speciesName, codonTableBio,
                                       colormapDiverging, vmin, vmax):
    
    # Adapted from the source code for Module Bio.Data.CodonTable
    # See: http://biopython.org/DIST/docs/api/Bio.Data.CodonTable-pysrc.html

    # Use the main four letters (and the conventional ordering) 
    # even for ambiguous tables 
    letters = ["T","C","A","G"]

    # Build the table...
    codonTableDf = pd.DataFrame(index=range(17), columns=range(10))
    codonTableDf.fillna("")
    
    # header
    codonTableDf.iloc[0,:] = ["","T","T","C","C","A","A","G","G",""]
    codonTableDf.iloc[:,0] = [""] + [letter for letter in letters for dummy in range(4)]

    i, j = 0, 0
    for k1, c1 in enumerate(letters):
        for k3, c3 in enumerate(letters):
            i = 4*k1 + k3 + 1
            codonTableDf.iloc[i,0] = c1
            for k2, c2 in enumerate(letters):
                j = 2*k2 + 1
                #print(k1, k2, k3, i, j, codon)
                codon = c1 + c2 + c3
                codonTableDf.iloc[i,j] = codon
                # Add the pvalue annotation                
                codonTableDf.iloc[i,j] += '<div class="pvalue">' + pvalAnnotTable1Pos.get(codon, "") + '</div>'
                
                # Add the amino acid
                # Here we follow the rules defined in the codon table from the Biopython genome object
                if codon in codonTableBio.stop_codons:
                    codonTableDf.iloc[i,j+1] = "Stop"
                else:
                    try:
                        amino = codonTableBio.forward_table[codon]
                    except KeyError:
                        amino = "?"
                    except TranslationError:
                        amino = "?"
                    if codon in codonTableBio.start_codons:
                        amino += "(s)"
                    codonTableDf.iloc[i,j+1] = amino
            codonTableDf.iloc[i,-1] = c3

    # Style the table

    # Applying style to columns and rows
    headerColor = '#ffffcc'

    def style_column(col):
        style_series = ['background-color: ' + headerColor if not (x==0 or x==len(col)) else '' for x in range(len(col))]
        return style_series

    def style_row(row):
        return ['background-color: ' + headerColor for x in range(len(row))]

    plotDataDict = plotData1Pos.to_dict()
    
    # Spplying style to individual cells
    def style_color(cell):
        color = ''
        codon = re.search(r'^[ATUCG]{3}', cell)
        codon = codon.group(0) if codon is not None else None
        value = plotDataDict.get(codon)
        # Note: value is None if the string in the cell is not a codon
        if value is not None:
            valueScaled = ((value-vmin)/(vmax-vmin))
            #print(value, valueScaled)
            
            # We only apply the backgrond color if the pvalue is significant
            if pvalAnnotTable1Pos.get(codon, "") != "":
                color = colormapDiverging(valueScaled)
                # Note: for some reason Pandas style does not work when applying RGB color
                #return "background-color:rgba({:f}, {:f}, {:f}, {:f})".format(*color)
                return "background-color: " + matplotlib.colors.rgb2hex(color)
            else:
                return ""
        else:
            return ""
        
    def style_align(cell):
        return "text-align: center"
    
    
    codonTableColorDf = codonTableDf.style\
                        .apply(style_column, subset=[0,9])\
                        .apply(style_row, axis=1, subset=0)\
                        .applymap(style_align)\
                        .applymap(style_color)
                        #.set_properties(**{'background-color':'white'})
        
    return codonTableColorDf


def plot_codon_table_html_table_formatting(codonTableColorDf):

    codonTableSoup = bs4.BeautifulSoup(codonTableColorDf.render(), "lxml")


    # Define custom css style for the pvalue annotation
    styleTag = codonTableSoup.find('style')
    styleTag.string = "\n\n            .pvalue { font-size: 50%; line-height:50% }\n\n" +                      "\n\n            table, th, td { border: 1px solid black; border-collapse: collapse; }\n\n" +                      styleTag.string
            
    # Define table in BeautifulSoup
    table = codonTableSoup.find('table')
    table_body = table.find('tbody')
    rows = table_body.find_all('tr')

    # Change some global attribute of the table
    ncol = len(rows[0].find_all('td'))
    for row in rows:
        cells = row.find_all('td')
        for cell in cells:
            None
            cell['style'] = 'padding:2px;'
            #print(cell)

    # Define methods to access individual cell in the table by index
    def get_table_element(i, j):
        return rows[i].find_all('td')[j]

    # Define methods to span cells accross rows or columns.
    # **important**: Note that once a cell has spun over the neihbouring cell, the latter
    # is deleted. The deleted cell is no more accessible and remove one cell from the row/column,
    # thus changing the indexing. Example: we span cell (2,3) in the cell next in the row (2,4).
    # Now the row #2 has one cell less. The original cell (2,5) now has the index (2,4).
    def span_cell_col(i, j, span):
        get_table_element(i, j)['colspan'] = span
        get_table_element(i, j)['style'] += "text-align:center;"
        for step in range(span-1):
            get_table_element(i, j+step+1).decompose()

    def span_cell_row(i, j, span):
        get_table_element(i, j)['rowspan'] = span
        get_table_element(i, j)['style'] += "text-align:center;"
        for step in range(span-1):
            get_table_element(i+step+1, j).decompose()

    for j in range(1,5):
        span_cell_col(0, j, 2)

    span_cell_row(1, 8, 2)     #C
    span_cell_row(5, 8, 4)     #R
    span_cell_row(9, 8, 2)     #S
    span_cell_row(11, 8, 2)    #R
    span_cell_row(13, 8, 4)    #G
    span_cell_row(1, 6, 2)     #Y
    span_cell_row(5, 6, 2)     #H
    span_cell_row(7, 6, 2)     #Q
    span_cell_row(9, 6, 2)     #N
    span_cell_row(11, 6, 2)    #K
    span_cell_row(13, 6, 2)    #D
    span_cell_row(15, 6, 2)    #E
    span_cell_row(1, 4, 4)     #S
    span_cell_row(5, 4, 4)     #P
    span_cell_row(9, 4, 4)     #T
    span_cell_row(13, 4, 4)    #A
    span_cell_row(1, 2, 2)     #F
    span_cell_row(3, 2, 6)     #L
    span_cell_row(9, 2, 3)     #I
    span_cell_row(13, 2, 4)    #V

    for i in range(0,4):
        span_cell_row(1 + 4*i, 0, 4)

    # Setting width of columns
    cells = rows[1].find_all('td')
    for cell in cells[2::2]:
        cell['width'] = '40px'
    for cell in cells[1:8:2]:
        cell['width'] = '40px'
    cells[0]['width'] = '20px'
    cells[-1]['width'] = '20px'

    # Deleting the top and left headers
    headers = table.find_all('th')
    for header in headers:
        header.decompose()

    # Iterate through cells in the table
    for row in rows[:1]:
        cells = row.find_all('td')
        for cell in cells:
            None
            #cell['align'] = 'center'
            #print(cell)
            
    #print(table)

    return str(codonTableSoup)


def convert_table_html_to_pdf_to_png(htmlTable, baseFilename, outputDirectory, width='4.7in', height='6.3in'):

    # Convert html table to pdf using pdfkit

    # Tweaking the paper size to fit the table
    options = {
        'page-width': width,
        'page-height': height,
        'margin-top': '0.1in',
        'margin-right': '0.1in',
        'margin-bottom': '0.1in',
        'margin-left': '0.1in',
        'encoding': "UTF-8",
        'no-outline': None
    }

    filenamePDF = os.path.join(outputDirectory, baseFilename + '.pdf')
    pdfkit.from_string(htmlTable, filenamePDF, options=options)

    # Convert pdf table to png using Wand (bindings for ImageMagick)
    
    with wandImage(filename=filenamePDF, resolution=300) as img:
        with wandImage(width=img.width, height=img.height, background=wandColor("white")) as bg:
            bg.composite(img,0,0)
            filenamePNG = os.path.join(outputDirectory, baseFilename + '.png')
            bg.save(filename=filenamePNG)


def plot_codon_table_import_png_final_layout(fullFilenamePNG, terminus, positionDfIndex, speciesName, colormapDiverging, vmin, vmax):

    # Import codon tables for different positions as png image and edit layout
    
    title = terminus + '-terminal pos ' + '{:0=3d}'.format(positionDfIndex) + ', ' + speciesName
    aspectratio = 1.2
    imagesize = 10
    fig = plt.figure(figsize=(imagesize, imagesize/aspectratio))
    grid = matplotlib.gridspec.GridSpec(2, 2, width_ratios=[1,0.025], height_ratios=[1.8,1], wspace=0.0, hspace=0.0)
    
    # Codon table, import as PNG image
    ax1 = fig.add_subplot(grid[:,0])
    filenameCodonTable = fullFilenamePNG
    imageCodonTable = mpimg.imread(filenameCodonTable)
    ax1.imshow(imageCodonTable)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title(title)

    # Colorbar
    ax2 = fig.add_subplot(grid[0,1])
    colormapNorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    matplotlib.colorbar.ColorbarBase(ax=ax2, cmap=colormapDiverging, norm=colormapNorm, ticks=range(vmin,vmax,1))
    ax2.set_ylabel('$\log_2$(odds ratio)')
    
    # Legend
    ax3 = fig.add_subplot(grid[1,1])
    legendPvalueAnnotation = write_latex_legendPvalueAnnotation(pvalueThresholds)
    ax3.annotate(legendPvalueAnnotation,
                 xy=(0.05,0.05), xycoords='axes fraction', xytext=(0, 0), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.3),
                 ha='left', va='bottom', fontsize='small')
    ax3.set_frame_on(False)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    
    plt.tight_layout()
    
    return fig


#@profile
def plot_codon_table(plotData, pvalAnnotTable, terminus, positionDfIndex, speciesName, codonTableBio, tempDirectory):

    """Final function to plot the codon table with color biases"""    

    # Important: here position is given in our defined index: 0 for stop codon.
    
    # Define global colormap for codon table
    colormapDiverging = seaborn.blend_palette(seaborn.color_palette("RdBu_r"), as_cmap=True, input='rgb')
    # Normalize colormap
    vmax = 3
    vmin = -vmax
    
    codonTableColorDf = plot_codon_table_dataframe_styling(plotData[positionDfIndex],
                                                           pvalAnnotTable[positionDfIndex],
                                                           speciesName, codonTableBio, colormapDiverging, vmin, vmax)
    
    codonTableColorHTML = plot_codon_table_html_table_formatting(codonTableColorDf)
    
    tempFilename = 'temp'+str(np.random.randint(1e18))
    
    convert_table_html_to_pdf_to_png(codonTableColorHTML, tempFilename, tempDirectory)
    
    fig = plot_codon_table_import_png_final_layout(os.path.join(tempDirectory, tempFilename + '.png'),                                                   terminus, positionDfIndex, speciesName, colormapDiverging, vmin, vmax)
    
    os.remove(os.path.join(tempDirectory, tempFilename + '.pdf'))
    os.remove(os.path.join(tempDirectory, tempFilename + '.png'))
        
    return fig



# ### Draw codon biases table using heatmap

def compute_codon_oddsratio_mask_data(oddsRatioDfCodon, pvalueThresholds, terminus):
    plotData = oddsRatioDfCodon
    plotData = plotData.xs(terminus, level='terminus')
    plotData = select_terminal_positions_in_index(plotData, terminus)
    plotData = plotData.xs('BH_multiple_tests', level='observable')
    # Mask both rejected cases and null values
    plotData = (plotData != True) & (plotData != 'True')
    plotData = plotData.transpose()
    return plotData



def compute_codon_smallN_mask_data(oddsRatioDf, pvalueThresholds, terminus):
    
    # Drop start/stop codon
    plotData = oddsRatioDf
    plotData = select_terminal_positions_in_index(plotData, terminus)
    
    bulkFreq = oddsRatioDf.xs('bulk', level='terminus').xs('count', level='observable').loc[0]
    NrefTot = bulkFreq.sum()
    bulkFreq = bulkFreq / NrefTot
    meanFreq = bulkFreq.mean().mean()
    
    counts = plotData
    counts = counts.xs(terminus, level='terminus')
    counts = counts.xs('count', level='observable')
    counts = counts.apply(pd.to_numeric)

    countsTot = counts.apply(lambda row: row.sum(), axis=1)
    nObsExp = countsTot.to_frame().dot(bulkFreq.to_frame().T)
    nObsExp = nObsExp.apply(pd.to_numeric)

    pvalue = plotData
    pvalue = pvalue.xs(terminus, level='terminus')
    pvalue = pvalue.xs('pvalue', level='observable')
    pvalue = pvalue.apply(pd.to_numeric)

    # print(meanFreq)
    # print(NrefTot)
    # print(nObsExp['GGT'])
    # print(counts['GGT'])
    # print(pvalue['GGT'])
    smallNDf = compute_smallN_mask(meanFreq, nObsExp, counts, pvalue)

    return smallNDf


#@profile
def plot_codon_composition_map(data, maskData, pvalAnnotTable, maskDataSmallN, terminus, speciesName, refCodonTableDf, width=5):
    
    plotData = data.copy()
    nameIndex = plotData.index.name
    nameCol = plotData.columns.name
    stopCodonList = list(refCodonTableDf[refCodonTableDf['aa'] == '*'].index)

    plotData = plotData.loc[plotData.index.map(lambda x: x not in stopCodonList)]
    plotData = sort_codon_index(plotData, refCodonTableDf,
                                addAminoAcidLetterToIndex=True, addAminoAcidLetterAsColumn=False)
    plotData.index.name = nameIndex
    plotData.columns.name = nameCol

    maskData = maskData.loc[maskData.index.map(lambda x: x not in stopCodonList)]
    maskData = sort_codon_index(maskData, refCodonTableDf,
                                addAminoAcidLetterToIndex=True, addAminoAcidLetterAsColumn=False)
    maskData.index.name = nameIndex
    maskData.columns.name = nameCol

    maskDataSmallN = maskDataSmallN.loc[maskDataSmallN.index.map(lambda x: x not in stopCodonList)]
    maskDataSmallN = sort_codon_index(maskDataSmallN, refCodonTableDf,
                                addAminoAcidLetterToIndex=True, addAminoAcidLetterAsColumn=False)
    maskDataSmallN.index.name = nameIndex
    maskDataSmallN.columns.name = nameCol
       
    figsizeCoeff = 0.6
    aspectratio = 0.6
    linewidth1 = 0.5
    figsize = (width, (width/aspectratio))

    fig, (ax,cbar_ax) = plt.subplots(1, 2, figsize=figsize)
    # main axes
    ax.set_position([0, 0, 1, 1])
    ax.set_facecolor(colorBackground1)

    # Create axes for the colorbar for more precise positioning
    # vertical bar
    cbar_aspect = 10
    cbar_length = 0.3
    cbar_ax.set_position([0.2, -0.3 - cbar_length, cbar_length/cbar_aspect, cbar_length])

    cbarLabel = '$\log_2$(odds ratio)'
    ax = seaborn.heatmap(plotData, square=True, mask=maskData.values, ax=ax, cmap=cmap,
                         cbar_ax=cbar_ax,
                         cbar_kws={"label":cbarLabel},
                         xticklabels=True, yticklabels=True,
                         vmin=-vmax, vmax=vmax,
                         linewidth=linewidth1, linecolor=colorHeatmapLine)
    ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
    if len(ax.xaxis.get_ticklabels()[0].get_text()) > 2:
        ax.xaxis.set_tick_params(rotation=90)
    ax.yaxis.set_tick_params(rotation=0)
    family = 'Liberation Mono'
    ticks_font = FontProperties(family=family, size='small')
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(ticks_font)
        tick.set_bbox(dict(pad=0, facecolor='none', edgecolor='none'))
           
    ny = maskDataSmallN.shape[0]
    hatch = '////' if figsizeCoeff > 1.1 else '///'
    for (i,j), value in np.ndenumerate(maskDataSmallN.values):
        if value:
            ax.add_patch( matplotlib.patches.Rectangle((j, ny - 1 - i), 1, 1,
                                                       edgecolor='w', facecolor=colorSmallN, hatch=hatch,
                                                       linewidth=linewidth1) )

    ax.set_xlabel('position (' + terminus + '-terminal)')

    title = terminus + '-terminal, ' + speciesName
    # ax.set_title(title)
    return fig



# ## Pairs of amino acids at N-terminal and C-terminal

#@profile
def count_bulk_subsequence_frequency(allProteinSeqDf, cTerminalSize, nTerminalSize, subseqSize, seq_type='protein'):
    """col can be either `proteinSeq` or `DNASeq`"""
    
    if seq_type == 'protein':
        col = 'proteinSeq'
        nTerminalGap = nTerminalSize
        cTerminalGap = nTerminalSize
    elif seq_type == 'DNA':
        col = 'DNASeq'
        nTerminalGap = 3*nTerminalSize
        cTerminalGap = 3*nTerminalSize
    else:
        raise ValueError("seq_type should be either protein or DNA.")
    
    if allProteinSeqDf is None:
        return (collections.Counter(), {})
    else:
        bulkSeqString = ""
        for seq in allProteinSeqDf[col]:
            bulkSeqString += seq[nTerminalGap:-cTerminalGap]

        # build database of subsequences from a sliding window
        bulkSeqSlidingWindowLibrary = list((x for x in sliding_window_string(bulkSeqString, subseqSize)))
        bulkSubseqFreq = collections.Counter(bulkSeqSlidingWindowLibrary)

        bulkSubseqRelativeFreq = {subseq: count/sum(bulkSubseqFreq.values()) for subseq, count in dict(bulkSubseqFreq).items()}
        bulkSubseqRelativeFreq = sorted(bulkSubseqRelativeFreq.items(), key=lambda x: x[1])

        return (bulkSubseqFreq, bulkSubseqRelativeFreq)


#@profile
def count_termina_subseq_frequency(allProteinSeqDf, cTerminalSize, nTerminalSize, subseqSize, seq_type='protein'):
    """col can be either `proteinSeq` or `DNASeq`.
    Note: subseqSize can be either in amino acids, or in nucleotides.
    """
    
    if seq_type == 'protein':
        col = 'proteinSeq'
    elif seq_type == 'DNA':
        col = 'DNASeq'
    else:
        raise ValueError("seq_type should be either protein or DNA.")
        
    if allProteinSeqDf is None:
        return (collections.Counter(), collections.Counter())
    else:
        # C-terminus
        if seq_type == 'protein':
            # C-terminus
            # CHECK THAT THE LAST AA IS NOT STOP CODON ASTERIX *
            ctermLibrary = [seq[-subseqSize:] if seq[-1] != '*' else seq[-subseqSize - 2: -1]
                            for seq in allProteinSeqDf[col] if len(seq) > cTerminalSize + nTerminalSize]
            # N-terminus. we drop the first methionine. Start from position 2.
            ntermLibrary = [seq[1: subseqSize + 1]
                            for seq in allProteinSeqDf[col] if len(seq) > cTerminalSize + nTerminalSize]
            
        elif seq_type == 'DNA':
            # In the case of DNA we drop the stop codon. We are only interested in codon,
            # codon pair (hexamer) frequency
            ctermLibrary = [seq[-3 - subseqSize: -3]
                            for seq in allProteinSeqDf[col] if len(seq) > 3*(cTerminalSize + nTerminalSize)]
            ntermLibrary = [seq[3: 3 + subseqSize]
                            for seq in allProteinSeqDf[col] if len(seq) > 3*(cTerminalSize + nTerminalSize)]
            
        ctermLibrary = list(filter(None,ctermLibrary))
        ntermLibrary = list(filter(None,ntermLibrary))

        # Building a collection of aa counts at each Cterminal position
        ctermFreq = collections.Counter(ctermLibrary)
        ntermFreq = collections.Counter(ntermLibrary)
        return (ctermFreq, ntermFreq)
    

# Compute subsequence frequency at bulk/C/N-termini compared to theoretical frequency
# of independent amino acid chain probabilities
def compute_theoretical_prob_subsequence(subseq, theoreticalAARelativeFreq):
    """
    Given theoretical probabilities (frequencies) for each amino acid,
    compute the probabilty of a subsequence, assuming independence of
    every amino acid probability. P(ABCA) = P(A)^2*P(B)*P(C).

    theoreticalAARelativeFreq: dictionary
    subsequ: string
    """
    prob = np.prod([theoreticalAARelativeFreq[aa] for aa in subseq])
    return prob



def binomial_chi_square_test(nObs, nObsTotal, pExp):
    pvalue = -1.0
    nExp = nObsTotal*pExp
    var = nExp*(1-pExp)
    if (nExp < 1000 or nObs < 1000 or var < 1000) and not (nExp < 1000000 or nObs < 1000 or var < 1000):
        # Use binomial exact test
        pvalue = scipy.stats.binom_test(nObs, n=nObsTotal, p=pExp)
    else:
        # Use Chi-square goodness of fit test
        # In this case the discrete binomial distribution can be approximated by the continuous normal distribution
        pvalue = scipy.stats.chisquare([nObs,nObsTotal-nObs], f_exp=[pExp*nObsTotal,(1-pExp)*nObsTotal])[1]
    return pvalue


#@profile
def compute_odds_ratio_subseq(bulkSubseqFreq, bulkFreqAA,
                              ctermSubseqFreq, cTerminalSize,
                              ntermSubseqFreq, nTerminalSize, subseqSize, seq_type='protein',
                              computeTheoreticalFreq=True,
                              computeMultipleTestsCorrection=True, verbose=0
                             ):
    """
    Compute odds ratios and p-values for subsequences (simplest case: amino acid pairs)
    at C- and N-termini and for bulk.
    
    oddsRatioDf.loc[pd.IndexSlice['C/N',:,'contingency',:],:] :
    odds ratio for subsequence frequency at C/N-termini compared
    to bulk subsequence frequency, using contingency table and exact Fisher test.
    
    oddsRatioDf.loc[pd.IndexSlice['bulk',:,'contingency',:],:] :
    should be set to None.
    
    oddsRatioDf.loc[pd.IndexSlice['bulk/C/N',:,'theoretical',:],:] :
    odds ratio for subsequence frequency at bulk/C/N-termini compared to
    theoretical frequency of position independent amino acid chain probabilities.
    """

    if computeTheoreticalFreq:
        totalCount = sum(bulkFreqAA.values())
        bulkRelFreqAA = {aa: count/totalCount for aa, count in dict(bulkFreqAA).items()}
        bulkRelFreqAA = sorted(bulkRelFreqAA.items(), key=lambda x: x[1])
    
    
    # subseqTable = list(itertools.product(aaTable, aaTable))
    subSeqList = bulkSubseqFreq.keys()
    if seq_type == 'protein':
        # Sort a.a. pairs
        aaTable
    else:
        # Sort codon pairs
        pass
    
    # This dataframe is organized with the list of subsequences as index
    terminiList = ['N','bulk','C']
    methodList = ['contingency','theoretical']
    obsList = ['count','theoreticalProb','NexpectedSubseq','log2OddsRatio','oddsRatio','pvalue']
    indexNameList = ['terminus','subseq','statistical_test','observable']
    multiIndex = pd.MultiIndex.from_product([terminiList, subSeqList, methodList, obsList],
                                            names=indexNameList)
    oddsRatioDf = pd.DataFrame(index=multiIndex, columns=['value'])
    oddsRatioDf.sort_index(level=0, inplace=True)
    
    if seq_type == 'protein':
        # For amino acid pairs: this dataframe is organized with the first amino acid
        # as row index and second amino acid as column index.
        multiIndex = pd.MultiIndex.from_product([terminiList, aaTable, methodList, obsList],
                                                names=['terminus','aa0','statistical_test','observable'])
        oddsRatioPairsDf = pd.DataFrame(index=multiIndex, columns=aaTable)
        # It is better not to sort the dataframe because we want to keep the custom order for the aa table
    else:
        oddsRatioPairsDf = None

    i = 0
    for terminus in ['N','C','bulk']:
        terminalSize = None
        observedFreq = None
        if terminus == 'C':
            terminalSize = cTerminalSize
            observedFreq = ctermSubseqFreq
        elif terminus == 'N':
            terminalSize = nTerminalSize
            observedFreq = ntermSubseqFreq
        elif terminus == 'bulk':
            terminalSize = None
            observedFreq = bulkSubseqFreq
            
        for subseq in subSeqList:
            
            i += 1
            NobservedSubseq = observedFreq.get(subseq, 0)
            NobservedTot = sum(observedFreq.values())
            if computeTheoreticalFreq:
                theoreticalProb = compute_theoretical_prob_subsequence(subseq, dict(bulkRelFreqAA))
                oddsRatio = ((NobservedSubseq/NobservedTot) / theoreticalProb) if theoreticalProb > 0 else None
                log2OddsRatio = log2(oddsRatio)
                pvalue = binomial_chi_square_test(NobservedSubseq, NobservedTot, theoreticalProb)
            # if subseq == 'PP':
            #     print("terminus:",terminus,"subseq:",subseq,"NobservedSubseq:",NobservedSubseq,"NobservedTot:",NobservedTot,
            #           "theoreticalProb:",theoreticalProb,"NexpectedSubseq:",theoreticalProb*NobservedTot,
            #           "oddsRatio:",oddsRatio,"log2OddsRatio:",log2OddsRatio,"pvalue binomial:",pvalue)
            
                oddsRatioDf.loc[terminus,subseq,'theoretical','count'] = NobservedSubseq
                oddsRatioDf.loc[terminus,subseq,'theoretical','theoreticalProb'] = theoreticalProb
                oddsRatioDf.loc[terminus,subseq,'theoretical','NexpectedSubseq'] = theoreticalProb*NobservedTot
                oddsRatioDf.loc[terminus,subseq,'theoretical','oddsRatio'] = oddsRatio
                oddsRatioDf.loc[terminus,subseq,'theoretical','log2OddsRatio'] = log2OddsRatio
                oddsRatioDf.loc[terminus,subseq,'theoretical','pvalue'] = pvalue
                
                if seq_type == 'protein':
                    oddsRatioPairsDf.loc[(terminus,subseq[0],'theoretical','count'),subseq[1]] = NobservedSubseq
                    oddsRatioPairsDf.loc[(terminus,subseq[0],'theoretical','theoreticalProb'),subseq[1]] = theoreticalProb
                    oddsRatioPairsDf.loc[(terminus,subseq[0],'theoretical','NexpectedSubseq'),subseq[1]] = theoreticalProb*NobservedTot
                    oddsRatioPairsDf.loc[(terminus,subseq[0],'theoretical','oddsRatio'),subseq[1]] = oddsRatio
                    oddsRatioPairsDf.loc[(terminus,subseq[0],'theoretical','log2OddsRatio'),subseq[1]] = log2OddsRatio
                    oddsRatioPairsDf.loc[(terminus,subseq[0],'theoretical','pvalue'),subseq[1]] = pvalue

            oddsRatioDf.loc[terminus,subseq,'contingency','count'] = NobservedSubseq
            if seq_type == 'protein':
                oddsRatioPairsDf.loc[(terminus,subseq[0],'contingency','count'),subseq[1]] = NobservedSubseq
            
            if terminus != 'bulk':

                # Compute contingency table for subsequence in terminus and in bulk
                NrefA    = bulkSubseqFreq.get(subseq, 0)
                NrefNotA = sum(bulkSubseqFreq.values()) - bulkSubseqFreq.get(subseq, 0)

                NjA      = observedFreq.get(subseq, 0)
                NjNotA   = sum(observedFreq.values()) - observedFreq.get(subseq, 0)
                
                if verbose >= 2: print("#", i, " subseq=",subseq," NjA=",NjA," NjNotA=",NjNotA,
                                       " NrefA=",NrefA," NrefNotA=",NrefNotA)
                contingencyTable = [[NjA, NrefA], [NjNotA, NrefNotA]]
                oddsRatioScipy, pvalue = scipy.stats.fisher_exact(contingencyTable, alternative='two-sided')
            
                if NjA == 0 or NjNotA == 0:
                    oddsRatioDf.loc[terminus,subseq,'contingency','oddsRatio'] = None
                    oddsRatioDf.loc[terminus,subseq,'contingency','log2OddsRatio'] = None
                    oddsRatioDf.loc[terminus,subseq,'contingency','pvalue'] = None

                    if seq_type == 'protein':
                        oddsRatioPairsDf.loc[(terminus,subseq[0],'contingency','oddsRatio'),subseq[1]] = None
                        oddsRatioPairsDf.loc[(terminus,subseq[0],'contingency','log2OddsRatio'),subseq[1]] = None
                        oddsRatioPairsDf.loc[(terminus,subseq[0],'contingency','pvalue'),subseq[1]] = None
                else:
                    oddsRatioDf.loc[terminus,subseq,'contingency','oddsRatio'] = oddsRatioScipy
                    oddsRatioDf.loc[terminus,subseq,'contingency','log2OddsRatio'] = log2(oddsRatioScipy)
                    oddsRatioDf.loc[terminus,subseq,'contingency','pvalue'] = pvalue

                    if seq_type == 'protein':
                        oddsRatioPairsDf.loc[(terminus,subseq[0],'contingency','oddsRatio'),subseq[1]] = oddsRatioScipy
                        oddsRatioPairsDf.loc[(terminus,subseq[0],'contingency','log2OddsRatio'),subseq[1]] = log2(oddsRatioScipy)
                        oddsRatioPairsDf.loc[(terminus,subseq[0],'contingency','pvalue'),subseq[1]] = pvalue
                      
    if computeMultipleTestsCorrection:
        dfList = []
        for terminus in ['N', 'C']:
            # Multiple test correction within the biases of C-terminal at all positions for all a.a.
            df = oddsRatioDf.xs(terminus, level='terminus', drop_level=False)
            df = df.xs('pvalue', level='observable', drop_level=False)
            # We serialize all the values, and drop the NaN
            df2 = df.stack().dropna().copy()
            reject, pvals_corrected, alphacSidak, alphacBonf = \
                multipletests(df2.values, alpha=family_wise_FDR, method='fdr_bh',
                              is_sorted=False, returnsorted=False)
            df2 = pd.DataFrame(reject, index=df2.index)
            # Stack again the values before merging
            df2 = df2.unstack()
            df2.columns = df2.columns.droplevel(0)
            df2 = df2.rename(index={'pvalue':'BH_multiple_tests'})
            dfList.append(df2)
        oddsRatioDf = pd.concat([oddsRatioDf] + dfList, axis=0, sort=True)
        oddsRatioDf.sort_index(inplace=True)

    if computeMultipleTestsCorrection and seq_type == 'protein':
        dfList = []
        for terminus in ['N', 'C']:
            # Multiple test correction within the biases of C-terminal at all positions for all a.a.
            df = oddsRatioPairsDf.xs(terminus, level='terminus', drop_level=False)
            df = df.xs('pvalue', level='observable', drop_level=False)
            # We serialize all the values, and drop the NaN
            df2 = df.stack().dropna().copy()
            reject, pvals_corrected, alphacSidak, alphacBonf = \
                multipletests(df2.values, alpha=family_wise_FDR, method='fdr_bh',
                              is_sorted=False, returnsorted=False)
            df2 = pd.DataFrame(reject, index=df2.index)
            # Stack again the values before merging
            df2 = df2.unstack()
            df2.columns = df2.columns.droplevel(0)
            df2 = df2.rename(index={'pvalue':'BH_multiple_tests'})
            dfList.append(df2)
        oddsRatioPairsDf = pd.concat([oddsRatioPairsDf] + dfList, axis=0, sort=True)
        oddsRatioPairsDf.sort_index(inplace=True)


    return oddsRatioDf, oddsRatioPairsDf


def compute_subseq_oddsratio_plot_data(subseqOddsRatioDf, terminus, statistical_test):
    plotData = subseqOddsRatioDf.xs('log2OddsRatio', level='observable')
    plotData = plotData.xs(terminus, level='terminus')
    plotData = plotData.xs(statistical_test, level='statistical_test')
    plotData = plotData[plotData.columns].astype(float)
    return plotData


def compute_subseq_oddsratio_mask_data(subseqOddsRatioDf, pvalueThresholds, terminus, statistical_test, subseq_type):
    # Do not show odds ratios for which the Fisher's test pvalue is higher than 0.05
    
    # maskData = subseqOddsRatioDf.xs('pvalue', level='observable')
    # maskData = maskData.xs(terminus, level='terminus')
    # maskData = maskData.xs(statistical_test, level='statistical_test')
    # maskData = maskData[maskData.columns].astype(float)
    # maskData = (maskData > pvalueThresholdMask) | (np.isnan(maskData))

    maskData = subseqOddsRatioDf.xs('BH_multiple_tests', level='observable')
    maskData = maskData.xs(terminus, level='terminus')
    maskData = maskData.xs(statistical_test, level='statistical_test')
    if subseq_type == 'aa':
        level = 'aa0'
    elif subseq_type == 'codon':
        level = 'codon_-2'
    missingIndex = subseqOddsRatioDf.index.get_level_values(level).unique().difference(maskData.index)
    if len(missingIndex) > 0:
        maskData = maskData.append(
            [pd.Series(np.nan, index=subseqOddsRatioDf.columns, name=missingIndexName)
             for missingIndexName in missingIndex])
    maskData = (maskData != True) & (maskData != 'True')
    return maskData


def compute_subseq_smallN_mask_data(oddsRatioDf, pvalueThresholds, terminus, statistical_test):
    
    bulkFreq = oddsRatioDf.xs(('bulk',statistical_test,'count'), level=['terminus','statistical_test','observable'])
    NrefTot = sum(bulkFreq.sum())
    bulkFreq = bulkFreq / NrefTot
    meanFreq = bulkFreq.mean().mean()
    
    counts = oddsRatioDf
    counts = counts.xs((terminus,statistical_test,'count'), level=['terminus','statistical_test','observable'])
    counts = counts.apply(pd.to_numeric)
    countsTot = sum(counts.sum())
    nObsExp = countsTot*(bulkFreq)
    nObsExp = nObsExp.apply(pd.to_numeric)

    pvalue = oddsRatioDf
    pvalue = pvalue.xs((terminus,statistical_test,'pvalue'), level=['terminus','statistical_test','observable'])
    pvalue = pvalue.apply(pd.to_numeric)

    # print(meanFreq)
    # print(nObsExp)
    # print(counts)
    # print(pvalue)
    smallNDf = compute_smallN_mask(meanFreq, nObsExp, counts, pvalue)

    return smallNDf


def compute_subseq_pvalueAnnotation(subseqOddsRatioDf, pvalueThresholds, terminus, statistical_test):
    # Build a table of text annotations representing pvalue
    pvalAnnotTable = subseqOddsRatioDf.xs('pvalue', level='observable')
    pvalAnnotTable = pvalAnnotTable.xs(statistical_test, level='statistical_test')
    pvalAnnotTable = pvalAnnotTable[pvalAnnotTable.columns].astype(float)
    pvalAnnotTable = pvalAnnotTable.xs(terminus, level='terminus')
    
    # We create a copy of the data frame with string type (cannot mix types inside data frame columns)
    pvalAnnotTableCopy = pvalAnnotTable.copy()
    pvalAnnotTableCopy = pvalAnnotTableCopy.astype(str)
    pvalAnnotTableCopy[:] = ""
    for i in range(0,len(pvalueThresholds)):
        if (i < len(pvalueThresholds)-1):
            condition = (pvalAnnotTable < pvalueThresholds[i][0]) & (pvalueThresholds[i+1][0] < pvalAnnotTable)
            pvalAnnotTableCopy[condition] = pvalueThresholds[i][1]
        else:
            condition = pvalAnnotTable < pvalueThresholds[i][0]
            pvalAnnotTableCopy[condition] = pvalueThresholds[i][1]

    return pvalAnnotTableCopy


#@profile
def plot_seq_pair_composition_map(data, maskData, pvalAnnotTable, maskDataSmallN, terminus,
                                  statistical_test, plotTitle, subseq_type='aa',
                                  plotSynonymousCodonGroupSeparationLine=False, refCodonTableDf=None,
                                  width=5, vmin1=-vmax, vmax1=vmax):

    plotData = data.copy()
    nameIndex = plotData.index.name
    nameCol = plotData.columns.name
    if subseq_type == 'codon':
        plotData = sort_codon_index(plotData.T, refCodonTableDf,
                                    addAminoAcidLetterToIndex=True, addAminoAcidLetterAsColumn=False)
        plotData = sort_codon_index(plotData.T, refCodonTableDf,
                                    addAminoAcidLetterToIndex=True, addAminoAcidLetterAsColumn=False)
        plotData.index.name = nameIndex
        plotData.columns.name = nameCol
    elif subseq_type == 'aa':
        plotData = plotData.loc[aaTable, aaTable]

    if subseq_type == 'codon':
        figsizeCoeff = 1.3
        # fontSize = 8.5
    else:
        figsizeCoeff = 0.6
        # fontSize = 16

    linewidth1 = 0.5
    print("v1.1")
    # We place the colorbar under the main plot, so that the limiting size is the width.
    aspectratio = 0.6
    cbarOrientation = 'horizontal'
    figsize = (width, (width/aspectratio))

    plotData.columns.name = re.sub(r'_', r' ', plotData.columns.name)
    plotData.index.name = re.sub(r'_', r' ', plotData.index.name)

    fig, (ax,cbar_ax) = plt.subplots(1, 2, figsize=figsize)
    # main axes
    ax.set_position([0, 0, 1, 1])
    ax.set_facecolor(colorBackground1)

    # Create axes for the colorbar for more precise positioning
    # cbar_ax.set_position([0.85, 0.45, 0.5/25, 0.5])
    cbar_aspect = 20
    cbar_length = 0.4
    if cbarOrientation == 'horizontal':
        cbar_ax.set_position([0.05, -0.3, cbar_length, cbar_length/cbar_aspect])

    cbarLabel = '$\log_2$(odds ratio)'
    ax = seaborn.heatmap(plotData, square=True, mask=maskData.values, ax=ax, cmap=cmap,
                         cbar_ax=cbar_ax,
                         cbar_kws=dict(label=cbarLabel, orientation=cbarOrientation),
                         xticklabels=True, yticklabels=True,
                         vmin=vmin1, vmax=vmax1,
                         linewidth=linewidth1, linecolor=colorHeatmapLine)
    cbar_ax = fig.axes[-1]
    tickLength = FontProperties(size='small').get_size()/4
    cbar_ax.xaxis.set_ticks([-vmax1, 0, vmax1])
    cbar_ax.tick_params(axis='x', length=tickLength, color=colorAxis)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        # spine.set_color('0.8')
        spine.set_linewidth(0.4)
    tickLabelsPad = 0
    ax.tick_params(axis='y', which='both', labelleft='on', labelright='on', labelsize='small', pad=tickLabelsPad)
    ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', labelsize='small', pad=tickLabelsPad)
    if len(ax.xaxis.get_ticklabels()[0].get_text()) > 2:
        ax.xaxis.set_tick_params(rotation=90)
    ax.yaxis.set_tick_params(rotation=0)
    family = 'Liberation Mono'
    ticks_font = FontProperties(family=family, size='small')
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(ticks_font)
        tick.set_bbox(dict(pad=0, facecolor='none', edgecolor='none'))
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(ticks_font)
        tick.set_bbox(dict(pad=0, facecolor='none', edgecolor='none'))

#     ny = pvalAnnotTable.shape[0]
#     for (i,j), value in np.ndenumerate(pvalAnnotTable.values):
#         ax.annotate(value, xy=(j + 0.5, ny - i - 0.5 - 0.35), # Note: the coordinates have to be transposed (j,ny-i)!!!
#                     horizontalalignment='center', verticalalignment='center',
#                     fontsize=figsizeCoeff*8,
#                     path_effects=[matplotlib.patheffects.withStroke(linewidth=1.5,foreground="w")])
    
    ny = maskDataSmallN.shape[0]
    hatch = '////' if figsizeCoeff > 1.1 else '///'
    for (i,j), value in np.ndenumerate(maskDataSmallN.values):
        if value:
            ax.add_patch( matplotlib.patches.Rectangle((j, ny - 1 - i), 1, 1,
                                                       edgecolor='w', facecolor=colorSmallN, hatch=hatch,
                                                       linewidth=linewidth1) )

    xpos = ""
    ypos = ""
    if terminus == 'N':
        ypos = "+2 (N-terminal)"   # Remark: this is row index in the dataframe
        xpos = "+3 (N-terminal)"   # Remark: this is column index in the dataframe
    elif terminus == 'C':
        ypos = "-2 (C-terminal)"   # Remark: this is row index in the dataframe
        xpos = "-1 (C-terminal)"   # Remark: this is column index in the dataframe
    ax.set_xlabel('position '+xpos)
    ax.set_ylabel('position '+ypos)
#     statistical_test_name = ''
#     if statistical_test == 'theoretical':
#         statistical_test_name = "exact binomial test against\ntheor. a.a. pair probability"
#     elif statistical_test == 'contingency':
#         statistical_test_name = "two-tails Fisher exact test"
#     legendPvalueAnnotation = write_latex_legendPvalueAnnotation(pvalueThresholds)
#     ax.annotate(legendPvalueAnnotation,
#                 xy=(1.0,0.1), xycoords='figure fraction', xytext=(-15, 0), textcoords='offset points',
#                 bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.3),
#                 ha='right', va='bottom', fontsize='small')

    if plotTitle is not None:
        if terminus == 'bulk':
            title = terminus + ', ' + plotTitle
        else:
            title = terminus + '-terminal, ' + plotTitle
        ax.set_title(title, y=1.07)
    # fig.tight_layout()
    
    if subseq_type == 'codon' and plotSynonymousCodonGroupSeparationLine:
        synCodonLineWidth = 0.6*linewidth1
        synCodonLineColor = '0.4'
        synCodonLineAlpha = 1
        synCodonLineStyle = '--'
        refCodonTableDf.sort_values('aa_groups_sorted_index')
        # Finding last rows of grouped dataframe on a multiindex column
        groupCol = 'aa'
        df = refCodonTableDf
        colDf = pd.DataFrame(df[groupCol].copy()).reset_index(drop=True).reset_index()
        lastS = colDf.groupby(by=groupCol).last().sort_values('index')['index']
        for aa, i in lastS[:-1].iteritems():
            ax.axvline(i + 1, ls=synCodonLineStyle, lw=synCodonLineWidth, c=synCodonLineColor, alpha=synCodonLineAlpha)
            ax.axhline(i + 1, ls=synCodonLineStyle, lw=synCodonLineWidth, c=synCodonLineColor, alpha=synCodonLineAlpha)
        
    return fig


# ### Merge CDS dataframe with the GO database
def merge_CDS_with_GO_database(allCDSDf_nr, amiGODf, membraneGOTermsDf):

    allCDSDf_merged = pd.merge(allCDSDf_nr, amiGODf, on='refSeqProteinId', how='left', left_index=True)

    # Add a column with boolean category True if the GO term is in the membrane category
    allCDSDf_merged['is_GO_membrane'] = allCDSDf_merged['GO_acc'].map(lambda GOterm: GOterm in tuple(membraneGOTermsDf.GOid) if GOterm is not None else None)
    print("allCDSDf_merged[ allCDSDf_merged['is_GO_membrane'] ][:20] = \n", allCDSDf_merged[ allCDSDf_merged['is_GO_membrane'] ][:20])

    print('found nb of membrane proteins:',
          len(allCDSDf_merged[allCDSDf_merged['is_GO_membrane']]['refSeqProteinId'].unique()))
    print('total nb of proteins:', len(allCDSDf_merged['refSeqProteinId'].unique()))

    # We have the same number of proteinGI as CDSID
    print('Nb of unique proteinGI:', len(allCDSDf_merged['proteinGI'].unique()))
    print('Nb of unique refSeqProteinId:', len(allCDSDf_merged['refSeqProteinId'].unique()))

    # Reduce allCDSDf dataframe by dropping all the GO terms, only keep the membrane category True/False, and removing
    # duplicated entries for the same protein.
    allCDSDf_merged2 = allCDSDf_merged.drop(['GO_genus','GO_species','GO_name','GO_acc','GO_term_type','GO_xref_dbname',
                                             'uniprotID'],
                                            axis=1, inplace=False)

    # Group the dataframe by unique protein sequence (we use refSeqProteinId but we could use another unique identifier for protein)
    grouped = allCDSDf_merged2.groupby(['refSeqProteinId'])

    # In each protein group, aggregate the is_GO_membrane boolean such that
    # the final value is True if at least one of the GO terms is in the membrane category and
    # False if none of the GO terms is in the membrane category.
    GOMembraneAggregated = grouped['is_GO_membrane'].aggregate(lambda x: sum(x) > 0)

    # Drop the membrane category column in the original dataframe, remove duplicates such that we have only
    # one row per protein, and add back the aggregated membrane category
    allCDSDf_merged2 = allCDSDf_merged2.drop('is_GO_membrane', axis=1).drop_duplicates('refSeqProteinId')
    allCDSDf_merged2 = pd.merge(allCDSDf_merged2, GOMembraneAggregated.to_frame().reset_index(),
                                on='refSeqProteinId', how='left', left_index=True)
    
    return allCDSDf_merged2



# ## Analysis pipeline, species group with individual species clustering


#@profile
def concatWrap(dfList):
    return pd.concat(dfList)


def simplify_prefix(prefix):
    return re.sub(r'[-\s=:/\\]', r'_', prefix)


#@profile
def full_analysis_multispecies(refSeqFolder,
                               dataSingleSpeciesFolder,
                               dataSpeciesGroupFolder,
                               plotsFolder,
                               speciesSetName, chooseRandomGenomes=False, nRandomGenomes=0,
                               assemblyAccessionList=[],
                               speciesList=[], multispeciesAllCDSDf_nr=None,
                               dfFormatOutput='csv', skipAnalysisExistingDataFiles=False,
                               skipStatisticsAnalysis=False,
                               method='iterative', skipClustering=False,
                               codonAnalysis=True,
                               categoryAnalysisColumn=None,
                               GOanalysis=False, amiGODf=None, membraneGOTermsDf=None,
                               verbose=1):
    """
    Warning: writing/reading of dataframe is not fully implemented because all objects such as
             biopython objects have to be parsed again from string to object. This is very tedious.
             allCDSDf can now be imported because it only contains string objects.
    Warning: currently Pandas does not support write/read to json files for multiindex dataframes.
    """

    print("version 1.2")
    identity_threshold = 0.8
    cterm_identity_threshold = 0.85

    # Set the codon table to standard bacterial code
    codonTableBio = Bio.Data.CodonTable.unambiguous_dna_by_id[11]
    refCodonTableDf = build_refCodonTable(codonTableBio)
    
    os.chdir(refSeqFolder)
    extractedFolder = 'Genomes_extracted'
    compressedFolder = 'Genomes_compressed'
    
    Path(plotsFolder).mkdir(exist_ok=True)
    (Path(plotsFolder) / speciesSetName).mkdir(exist_ok=True)
    (Path(dataSpeciesGroupFolder) / speciesSetName).mkdir(exist_ok=True)

    
    def compute_statistics(allCDSDf_nr):
        # Count the frequency of amino acids
        bulkFreqAA, bulkRelFreqAA = count_bulk_aa_frequency(allCDSDf_nr, cTerminalSize, nTerminalSize)
        print('\n#### Amino acid frequency bulk:\n')
        print(bulkFreqAA,'\n')
        print(sorted(bulkRelFreqAA, key=lambda item: item[1], reverse=True))
        ctermFreqAA, ntermFreqAA = count_termina_aa_frequency(allCDSDf_nr, cTerminalSize, nTerminalSize)
        print('\n#### Amino acid frequency c-terminal:\n',ctermFreqAA[-1])
        print('\n#### Amino acid frequency n-terminal:\n',ntermFreqAA[1])

        if codonAnalysis:
            # Count the frequency of codons
            bulkFreqCodon, bulkRelFreqCodon = count_bulk_codon_frequency(allCDSDf_nr, cTerminalSize, nTerminalSize)
            ctermFreqCodon, ntermFreqCodon = count_termina_codon_frequency(allCDSDf_nr, cTerminalSize, nTerminalSize)
            print('Bulk codon frequency:\n',bulkFreqCodon)
        else:
            bulkFreqCodon, bulkRelFreqCodon = None, None
            ctermFreqCodon, ntermFreqCodon = None, None

        # Count the frequency of amino acid pairs
        subseqSizeAA = 2
        bulkFreqAApair, bulkRelativeFreqAApair = count_bulk_subsequence_frequency(allCDSDf_nr, cTerminalSize, nTerminalSize,
                                                                                  subseqSizeAA, seq_type='protein')
        print('Bulk amino acid pairs frequency:\n',bulkFreqAApair)
        # Note: for the N-terminus we consider positions 2 and 3 (position 1 is always methionine)
        ctermFreqAApair, ntermFreqAApair = count_termina_subseq_frequency(allCDSDf_nr, cTerminalSize, nTerminalSize,
                                                                          subseqSizeAA, seq_type='protein')
        print('Amino acid pairs frequency c-terminal:\n',ctermFreqAApair)

        # Count the frequency of hexamers (codon pairs)
        bulkFreqHexamer, bulkRelFreqHexamer = count_bulk_subsequence_frequency(allCDSDf_nr, cTerminalSize, nTerminalSize,
                                                                               subseqSize=6, seq_type='DNA')
        print('Bulk frequency hexamers:\n',bulkFreqHexamer)
        ctermFreqHexamer, ntermFreqHexamer = count_termina_subseq_frequency(allCDSDf_nr, cTerminalSize, nTerminalSize,
                                                                            subseqSize=6, seq_type='DNA')
        print('Frequency c-terminal hexamers:\n',ctermFreqHexamer)

        statistics = (bulkFreqAA, ctermFreqAA, ntermFreqAA,
                      bulkFreqCodon, ctermFreqCodon, ntermFreqCodon,
                      bulkFreqAApair, ctermFreqAApair, ntermFreqAApair,
                      bulkFreqHexamer, ctermFreqHexamer, ntermFreqHexamer)
        return statistics
    
    
    
    # Define the different statistics groups to be computed (subsets of sequences)
    statisticsNameList = ['all']
    
    if codonAnalysis:
        # Split sequences into three classes depending on the stop codon identity
        stopCodonList = list(refCodonTableDf[refCodonTableDf['aa'] == '*'].index)
        statisticsNameList = statisticsNameList + ['stopCodon_' + stopCodon for stopCodon in stopCodonList]
        if verbose >= 1: print("stopCodonList", stopCodonList)
    
    if GOanalysis:
        for isMembraneProtein in [True,False]:
            name = ('' if isMembraneProtein else 'not ') + 'membrane proteins'
            statisticsNameList.append(name)

    # Note: the general category analysis is not yet implemented for the iterative method
    if categoryAnalysisColumn is not None and multispeciesAllCDSDf_nr is not None and method == 'allSeqDataframe':
        statisticsNameList = statisticsNameList + list(multispeciesAllCDSDf_nr[categoryAnalysisColumn].unique())
            
    multispeciesStatistics = {name: [] for name in statisticsNameList}
    multispeciesStatisticsSummary = {name: (pd.DataFrame(columns=['genome_accession','species_name','nSeq']))
                                     for name in statisticsNameList}
    

    
    # If we input the final multispecies dataframe, skip the first part of the analysis
    if multispeciesAllCDSDf_nr is not None and method != 'allSeqDataframe':
        print('Error: cannot take as input the multispeciesAllCDSDf_nr dataframe if the method is not set to allSeqDataframe.')
        return
        
    if multispeciesAllCDSDf_nr is not None and method == 'allSeqDataframe':
        print("Taking as input the multispeciesAllCDSDf_nr dataframe. Skipping first part of analysis.")
        print("len(multispeciesAllCDSDf_nr):",len(multispeciesAllCDSDf_nr))
        inputMultispeciesAllCDSDf = True
        
    else:
        
        inputMultispeciesAllCDSDf = False

        assemblySummaryRepDf = read_assemblySummary_file(str(Path(refSeqFolder) / 'assembly_summary_refseq.txt'))

        if chooseRandomGenomes is True:
            # Select genomes randomly in the assemblySummaryDf
            assemblyAccessionList = np.random.choice(assemblySummaryRepDf.index, size=nRandomGenomes, replace=False)

        if speciesList != [] and assemblyAccessionList == []:
            # Convert list of species names to list of assembly accession numbers.
            for speciesName in speciesList:
                compressedGenomeFilename, species_name, genome_accession = import_genome_gbff_file(assemblySummaryRepDf, species_name=speciesName)
                print(species_name, genome_accession)
                assemblyAccessionList.append(genome_accession)
        
        print('\n#### Species set assembly accession list\n\n',
              'len(assemblyAccessionList) = ', len(assemblyAccessionList), '\n\n', assemblyAccessionList,'\n\n')
        
        ### Iterate over the species, initialization of final variables

        #multispeciesAllProteinSeqDf_nr = []
        multispeciesAllCDSDf_nr = []



        ### START ITERATION OVER SPECIES ###

        for genomeAccession in assemblyAccessionList:

            gc.collect()

            os.chdir(refSeqFolder)

            print('\n#### Preparing sequence database for species', genomeAccession)

            print('\n#### Import genome from RefSeq bacterial genomes database (Genbank format)')

            compressedGenomeFilename, species_name, genome_accession = import_genome_gbff_file(assemblySummaryRepDf, genome_accession=genomeAccession)
            outputFilePrefix = genome_accession + '_' + species_name
            outputFilePrefix = simplify_prefix(outputFilePrefix)
            print('refSeqFolder:',refSeqFolder)
            print('compressedGenomeFilename:',compressedGenomeFilename)
            print('species_name:', species_name)
            print('genome_accession:', genome_accession)

            if not os.path.exists(os.path.join(refSeqFolder, compressedFolder, compressedGenomeFilename)):
                # Skip genome and print warning message
                print('Warning: genome file "', os.path.join(refSeqFolder,extractedFolder, compressedGenomeFilename),
                     '" cannot be found. Skipping genome in the analysis.')
                
            else:

                genomeFilename = extract_compressed_genome_file(compressedGenomeFilename, compressedFolder, extractedFolder)

                # Skip analysis if data file already exist
                allProteinSeqDfFileExists = False
                allProteinSeqDf = pd.DataFrame()
                if dfFormatOutput =='csv':
                    allProteinSeqDfFileExists = os.path.isfile(os.path.join(dataSingleSpeciesFolder, outputFilePrefix + "_allProteinSeqDf.csv"))
                elif dfFormatOutput =='json':
                    allProteinSeqDfFileExists = os.path.isfile(os.path.join(dataSingleSpeciesFolder, outputFilePrefix + "_allProteinSeqDf.json"))

                if skipAnalysisExistingDataFiles and allProteinSeqDfFileExists:
                    print('\n#### Reading existing data file allProteinSeqDf found for species ', genomeAccession)
                    if dfFormatOutput =='csv':
                        allProteinSeqDf = pd.read_csv(os.path.join(dataSingleSpeciesFolder, outputFilePrefix + "_allProteinSeqDf.csv"))
                    elif dfFormatOutput =='json':
                        allProteinSeqDf = pd.read_json(os.path.join(dataSingleSpeciesFolder, outputFilePrefix + "_allProteinSeqDf.json"))
                    allProteinSeqDf.set_index('refSeqProteinId', inplace=True, drop=True)
                    allProteinSeqDf.drop('refSeqProteinId.1', axis=1, inplace=True, errors='ignore')

                else:

                    # Parse genome file
                    genomeBio = next(Bio.SeqIO.parse(os.path.join(refSeqFolder,extractedFolder, genomeFilename), format="genbank"))

                    print('\n#### Generate fasta file for protein sequences')
                    # write the fasta file
                    proteinSeqFastaFolder = 'Protein_sequences_fasta'
                    proteinSeqFastaFilename = re.sub(r'(GCF.+)_genomic.gbff', r'\1_protein.faa', genomeFilename)
                    allProteinSeqFastaFile = os.path.join(refSeqFolder, proteinSeqFastaFolder, proteinSeqFastaFilename)
                    with open(allProteinSeqFastaFile, 'w') as file:
                        file.write(generate_protein_seq_fasta(genomeBio, verbose=False))


                    print('\n#### Import protein sequences fasta file')
                    allProteinSeqBio = list(Bio.SeqIO.parse(allProteinSeqFastaFile, "fasta"))
                    allProteinSeqDf = prepare_allSeq(allProteinSeqBio, species_name)
                    print("allProteinSeqDf: \n", allProteinSeqDf.head())

                    if skipClustering:

                        # Define non-redundant database of protein sequences
                        allProteinSeqDf['non-redundant'] = True
                        #allProteinSeqDf_nr = allProteinSeqDf.copy()
                        #allProteinSeqDf_nr.drop('refSeqProteinId', axis=1, inplace=True)

                    else:

                        print('\n#### Cluster analysis of protein sequences')

                        # We will remove protein sequences that have both a high overall identity
                        # and a high identity at the C-terminal. The resulting list of sequences will
                        # form the non-redundant database for our analysis.

                        # Perform a cluster analysis of all protein sequences using CD-HIT

                        # We write the output of the clustering in a folder like "Clustering_GCF_000027345.1_ASM2734v1_protein",
                        # relative to the path of the protein sequence fasta file.
                        clusteringOutputFolder = os.path.join(os.path.dirname(allProteinSeqFastaFile),
                                                              "Clustering_" + re.sub(r'.faa$', r'', os.path.basename(allProteinSeqFastaFile)))
                        clusteringFolder, clusteringOutputFile = clustering2(allProteinSeqFastaFile,
                                                                            clusteringOutputFolder,
                                                                            identity_threshold)

                        # Parse cluster output file and create dictionary of clusters
                        cluster_dic = parse_cluster_file(clusteringOutputFile + '.clstr')

                        # Print clusters with more than 1 sequence
                        print("\nCluster list:")
                        for key, cluster in cluster_dic.items():
                            if len(cluster) > 1:
                                print('Nb of seq: ',len(cluster),key,cluster)
                        print()

                        # Add the clusters to the dataframe (use multi index to group the rows by cluster)
                        allProteinSeqDf = add_clusters_to_dataframe(allProteinSeqDf, cluster_dic)

                        print('\n#### Clustering end')



                        # In each cluster, perform a second clustering analysis on the C-terminal part of sequences.
                        print('\n#### Clustering c-terminal start')
                        os.chdir(refSeqFolder)
                        allProteinSeqDf = cterm_clustering(allProteinSeqDf, clusteringFolder, cterm_identity_threshold,                                                            cTerminalSize, verbose=False)
                        print('\n#### Clustering c-terminal end')

                        # Drop the seqBio objects as they cannot be written to and read from csv file.
                        allProteinSeqDf.drop('seqBio', axis=1, inplace=True)

                        # Save dataframes as files
                        print('\n#### Writing _allProteinSeqDf files, start')
                        allProteinSeqDf.reset_index()
                        filename = os.path.join(dataSingleSpeciesFolder, outputFilePrefix + "_allProteinSeqDf." + dfFormatOutput)
                        write_dataframe(allProteinSeqDf, filename, dfFormatOutput)
                        print('\n#### Writing _allProteinSeqDf files, end')


                # Define non-redundant database of protein sequences
                print("\nallProteinSeqDf:\n",allProteinSeqDf.head())



                allProteinSeqDf_nr = allProteinSeqDf.copy()
                # Filter for proteins that are more than 50 a.a. long
                allProteinSeqDf_nr = allProteinSeqDf_nr[
                    allProteinSeqDf_nr.apply(lambda row: len(row['proteinSeq']) >= cTerminalSize + nTerminalSize + 10, axis=1 )
                    ]
                allProteinSeqDf_nr.dropna(subset=['non-redundant'], inplace=True)
                allProteinSeqDf_nr = allProteinSeqDf_nr[ allProteinSeqDf_nr['non-redundant'] ]
                # Flatten multiindex
                allProteinSeqDf_nr.drop('refSeqProteinId', axis=1, inplace=True, errors='ignore')
                allProteinSeqDf_nr.reset_index(inplace=True)
                allProteinSeqDf_nr.set_index('refSeqProteinId', drop=True, inplace=True)
                # Drop cluster information, not useful
                if 'cluster' in allProteinSeqDf_nr.columns:
                    allProteinSeqDf_nr.drop('cluster', axis=1, inplace=True, errors='ignore')
                print("Nb of sequences: ",len(allProteinSeqDf))
                print("Nb of non-redundant sequences: ",len(allProteinSeqDf_nr))
                print("\nallProteinSeqDf_nr:\n",allProteinSeqDf_nr.head())



                # Skip analysis if data file already exist
                allCDSDfFileExists = False
                allCDSDf = pd.DataFrame()
                if dfFormatOutput == 'csv':
                    allCDSDfFileExists = os.path.isfile(os.path.join(dataSingleSpeciesFolder, outputFilePrefix + "_allCDSDf.csv"))
                elif dfFormatOutput == 'json':
                    allCDSDfFileExists = os.path.isfile(os.path.join(dataSingleSpeciesFolder, outputFilePrefix + "_allCDSDf.json"))

                if skipAnalysisExistingDataFiles and allCDSDfFileExists:
                    print('\n#### Reading existing data file allCDSDf found for species ', genomeAccession)
                    if dfFormatOutput == 'csv':
                        allCDSDf = pd.read_csv(os.path.join(dataSingleSpeciesFolder, outputFilePrefix + "_allCDSDf.csv"))
                    elif dfFormatOutput == 'json':
                        allCDSDf = pd.read_json(os.path.join(dataSingleSpeciesFolder, outputFilePrefix + "_allCDSDf.json"))
                    allCDSDf.set_index('refSeqProteinId', inplace=True, drop=True)
                    allCDSDf.drop('refSeqProteinId.1', axis=1, inplace=True)
                    print(allCDSDf.head())
                else:

                    # Parse genome file
                    genomeBio = next(Bio.SeqIO.parse(os.path.join(refSeqFolder,extractedFolder, genomeFilename), format="genbank"))

                    print('\n#### Extracting all RNA coding sequences from the genome, start')

                    # Extract all coding sequences from the genome into a dataframe
                    allCDSDf = prepare_all_CDS(genomeBio, genome_accession, os.path.join(extractedFolder, genomeFilename), verbose=0)
                    print('\n#### Extracting all RNA coding sequences from the genome, end')

                    # Save dataframes as files
                    print('\n#### Writing allCDSDf files, start')
                    filename = os.path.join(dataSingleSpeciesFolder, outputFilePrefix + "_allCDSDf." + dfFormatOutput)
                    write_dataframe(allCDSDf, filename, dfFormatOutput)
                    print('\n#### Writing allCDSDf files, end')



                # Filter the coding sequences for the non-redundant ones using the index of allProteinSeqDf_nr
                allCDSDf_nr = allCDSDf.loc[allProteinSeqDf_nr.index].copy()

                # Append sequences to the multispecies dataframe list.
                if method == 'allSeqDataframe':

                    multispeciesAllCDSDf_nr.append(allCDSDf_nr)
                    #multispeciesAllProteinSeqDf_nr.append(allProteinSeqDf_nr)



                # Compute statistic and append to the multispeciesStatistics list.
                if method == 'iterative':

                    statistics = {}
                    statisticsSummary = {}

                    # Skip compute statistics if file exists
                    statisticsFilename = os.path.join(dataSingleSpeciesFolder, outputFilePrefix + "_countStatistics.json")
                    statisticsSummaryFilename = os.path.join(dataSingleSpeciesFolder, outputFilePrefix + "_countStatisticsSummary.json")
                
                    statisticsFileExists = os.path.isfile(statisticsFilename)
                    statisticsSummaryFileExists = os.path.isfile(statisticsSummaryFilename)
                    
                    if skipAnalysisExistingDataFiles and statisticsFileExists and statisticsSummaryFileExists:
                        print('\n#### Reading existing data file countStatistics found for species ', genomeAccession)
                        
                        # Import the statistics files
                        with open(statisticsFilename, 'r') as file:
                            statistics = json.load(file)
                        # Convert python dict to Counter
                        statisticsConverted = statistics.copy()
                        for keygroup, group in statistics.items():
                            for ivalue, value in enumerate(group):
                                if type(value) is dict:
                                    statisticsConverted[keygroup][ivalue] = Counter(value)
                                elif type(value) is list:
                                    for i, item in enumerate(value):
                                        statisticsConverted[keygroup][ivalue][i] = Counter(item)
                        statistics = statisticsConverted
                        
                        with open(statisticsSummaryFilename, 'r') as file:
                            statisticsSummary = json.load(file)
    
                    else:

                        print('\n#### Compute statistics, start')
    
                        print('\n#### Compute statistics, all')
                        statistics['all'] = compute_statistics(allCDSDf_nr)
                        nSeq = len(allCDSDf_nr) if allCDSDf_nr is not None else 0
                        statisticsSummary['all'] = {'genome_accession':genome_accession,
                                                    'species_name':species_name,
                                                    'nSeq':nSeq}

                        print('\n#### Compute statistics, stop codon classes')
                        for i, stopCodon in enumerate(stopCodonList):
                            allCDSDfGroup = allCDSDf_nr.groupby('stopCodon')
                            groupKeys = allCDSDfGroup.groups.keys()
                            allCDSDfGroup = allCDSDfGroup.get_group(stopCodon) if stopCodon in groupKeys else None
                            statistics['stopCodon_' + stopCodon] = compute_statistics(allCDSDfGroup)
                            nSeq = len(allCDSDfGroup) if allCDSDfGroup is not None else 0
                            statisticsSummary['stopCodon_' + stopCodon] = {'genome_accession':genome_accession,
                                                                           'species_name':species_name,
                                                                           'nSeq':nSeq}
    
                        # Save statistics to file
                        with open(statisticsFilename, 'w') as outfile:
                            json.dump(statistics, outfile, separators=(',', ':'))
                        with open(statisticsSummaryFilename, 'w') as outfile:
                            json.dump(statisticsSummary, outfile, separators=(',', ':'))
                    
                        print('\n#### Compute statistics, end')
                        
                    ### Append statistics to multispecies statistics list
                    for groupKey, statList in multispeciesStatistics.items():
                        statList.append(statistics[groupKey])

                    # multispeciesStatisticsSummary is a dictionary of pandas dataframes
                    for groupKey, statSummaryDf in multispeciesStatisticsSummary.items():
                        multispeciesStatisticsSummary[groupKey] = statSummaryDf.append(statisticsSummary[groupKey], ignore_index=True)



            #### END OF ITERATION OVER SPECIES ###


            



    if method == 'allSeqDataframe':

        # Concatenate all species dataframes and define stat groups (subsets of sequences)
        
        if not inputMultispeciesAllCDSDf:
            print('\nConcatenate all dataframes')
            multispeciesAllCDSDf_nr = pd.concat(multispeciesAllCDSDf_nr)
        
            # Save multispecies dataframe to file
            outputFilePrefix = speciesSetName
            filename = os.path.join(dataSpeciesGroupFolder, speciesSetName, outputFilePrefix + "_multispeciesAllCDSDf_nr." + dfFormatOutput)
            print("Writing multispeciesAllCDSDf_nr to file. filename:", filename)
            write_dataframe(multispeciesAllCDSDf_nr, filename, dfFormatOutput)


        # Define statistics group
        multispeciesDfGroups = {name: pd.DataFrame() for name in statisticsNameList}

        # All sequences
        multispeciesDfGroups['all'] = multispeciesAllCDSDf_nr
        summaryDf = multispeciesDfGroups['all'].groupby(['genome_accession', 'species_name'])
        summaryDf = summaryDf.size().to_frame().reset_index().rename(columns={0:'nSeq'})
        multispeciesStatisticsSummary['all'] = summaryDf

        if codonAnalysis:
            # Stop codons groups
            stopCodonGroups = multispeciesAllCDSDf_nr.groupby('stopCodon')
            for stopCodon, group in stopCodonGroups:
                multispeciesDfGroups.update({'stopCodon_' + stopCodon: group})
                summaryDf = group.groupby(['genome_accession','species_name'])
                summaryDf = summaryDf.size().to_frame().reset_index().rename(columns={0:'nSeq'})
                multispeciesStatisticsSummary['stopCodon_' + stopCodon] = summaryDf
            
        # Membrane proteins groups
        if GOanalysis:
            multispeciesAllCDSDf_nr = merge_CDS_with_GO_database(multispeciesAllCDSDf_nr, amiGODf, membraneGOTermsDf)
            print("GO analysis, multispeciesAllCDSDf_nr[:5] :\n", multispeciesAllCDSDf_nr[:5])
            
            groups = multispeciesAllCDSDf_nr.groupby('is_GO_membrane')
            for isMembraneProtein, group in groups:
                name = ('' if isMembraneProtein else 'not ') + 'membrane proteins'
                multispeciesDfGroups.update({name: group})
                summaryDf = group.groupby(['genome_accession','species_name'])
                summaryDf = summaryDf.size().to_frame().reset_index().rename(columns={0:'nSeq'})
                multispeciesStatisticsSummary[name] = summaryDf

        # General analysis by category
        if categoryAnalysisColumn is not None:
            groups = multispeciesAllCDSDf_nr.groupby(categoryAnalysisColumn)
            for cat, group in groups:
                name = cat
                multispeciesDfGroups.update({name: group})
                summaryDf = group.groupby(['genome_accession','species_name'])
                summaryDf = summaryDf.size().to_frame().reset_index().rename(columns={0:'nSeq'})
                multispeciesStatisticsSummary[name] = summaryDf


    if not skipStatisticsAnalysis:

        print("statisticsNameList:", statisticsNameList)
        for statName in statisticsNameList:

            # Compute and get statistics (counts)

            multispeciesStatistics1 = []
            multispeciesStatisticsSummary1 = multispeciesStatisticsSummary[statName]
            print('\nmultispeciesStatisticsSummary:',multispeciesStatisticsSummary1)
            nSeq = multispeciesStatisticsSummary1.nSeq.sum()
            emptyStatistics = nSeq <= 0

            if emptyStatistics:
                print('\n#### Zero statistics for stat:', statName, ', skipping analysis')

            else:

                if method == 'allSeqDataframe':

                    # Compute statistics on the whole multispecies dataframe
                    print('\n#### Compute statistics for stat ', statName, ', start')
                    multispeciesStatistics1 = compute_statistics(multispeciesDfGroups[statName])
                    print('\n#### Compute statistics for stat ', statName, ', end')


                elif method == 'iterative':

                    print('\n#### Gathering statistic for stat:', statName, 'start')

                    # Gather statistics
                    # We transpose the list of tuples such that we have a list:
                    # [ (counter0, [counterA0,counterB0]), (counter1, [counterA1,counterB1]), ... ] -->
                    #    [ (counter0,counter1,...), ([counterA0,counterB0],[counterA1,counterB1], ...) ]
                    # and perform a reduction by using the add operator with the Counter dictionaries to add up counts

                    def reduce_statistics_function(stat1, stat2):
                        # The statisticsList can be either a list of counters or a list of lists of counters
                        if type(stat1) == collections.Counter:
                            # Just add the counters
                            return stat1 + stat2
                        elif type(stat1) == list:
                            # Add the counters element-wise in the lists, as
                            # ([counterA0,counterB0],[counterA1,counterB1], ...) -->
                            #    [counterA0 + counterA1, counterB0 + counterB1, ...]
                            return list(map(operator.add, stat1, stat2))

                    multispeciesStatistics1 = [reduce(reduce_statistics_function, statisticsList)
                                               for statisticsList in zip(*(multispeciesStatistics[statName]))]
                    
                    print('\n#### Gathering statistic for stat:',statName,'end')

                outputFilePrefix = simplify_prefix(speciesSetName + '_subset_' + str(statName))

                filename = os.path.join(dataSpeciesGroupFolder, speciesSetName, outputFilePrefix + "_multispeciesStatistics1.json")
                with Path(filename).open('w') as f:
                    json.dump(multispeciesStatistics1, f)

                # Statistical analysis

                # Unpack statistics
                bulkFreqAA, ctermFreqAA, ntermFreqAA,\
                    bulkFreqCodon, ctermFreqCodon, ntermFreqCodon,\
                    bulkFreqAApair, ctermFreqAApair, ntermFreqAApair,\
                    bulkFreqHexamer, ctermFreqHexamer, ntermFreqHexamer = multispeciesStatistics1

                print('\n#### Analysis for stat:',statName,', start')
                print('\noutputFilePrefix:', outputFilePrefix)
                
                # Saving statistics summary to file
                filename = os.path.join(dataSpeciesGroupFolder, speciesSetName, outputFilePrefix + "_multispeciesStatisticsSummary." + dfFormatOutput)
                write_dataframe(multispeciesStatisticsSummary1, filename, dfFormatOutput)
                
                print('\n#### Termini amino acid composition analysis')
                
                # Compute odds ratios
                oddsRatioDf = compute_odds_ratio(bulkFreqAA, ctermFreqAA, cTerminalSize, ntermFreqAA, nTerminalSize)
                # Save dataframes as files
                print('\n#### Writing oddsRatioAADf files, start')
                filename = os.path.join(dataSpeciesGroupFolder, speciesSetName, outputFilePrefix + "_oddsRatioAADf." + dfFormatOutput)
                write_dataframe(oddsRatioDf, filename, dfFormatOutput)
                print('\n#### Writing oddsRatioAADf files, end')

                print('\n#### Termini amino acid composition analysis end')

                if codonAnalysis:
                    print('\n#### Termini codons composition analysis, start')
                    # USE THE NON-REDUNDANT DATABASE
                    oddsRatioDfCodon = compute_odds_ratio_codons(bulkFreqCodon,
                                                                 ctermFreqCodon, cTerminalSize,
                                                                 ntermFreqCodon, nTerminalSize,
                                                                 verbose=False)
                    # Save dataframes as files
                    print('\n#### Writing oddsRatioDfCodon files, start')
                    filename = os.path.join(dataSpeciesGroupFolder, speciesSetName, outputFilePrefix + "_oddsRatioDfCodon." + dfFormatOutput)
                    write_dataframe(oddsRatioDfCodon, filename, dfFormatOutput)
                    print('\n#### Writing oddsRatioDfCodon files, end')
                    print('\n#### Termini codons composition analysis, end')

                print('\n#### AA pairs bias analysis start')
                # Compute subsequence odds ratios for N-terminus, C-terminus and bulk,
                # using contingency table (C-terminus and N-terminus) or 
                # theoretical subsequence probability assuming independent amino acids (C-terminus, N-terminus and bulk)
                oddsRatioDfAApair, oddsRatioDfTableAApair = compute_odds_ratio_subseq(bulkFreqAApair, bulkFreqAA,
                                                                                      ctermFreqAApair, cTerminalSize,
                                                                                      ntermFreqAApair, nTerminalSize,
                                                                                      subseqSize=2, seq_type='protein',
                                                                                      computeTheoreticalFreq=True)
                # Drop the rare amino acid U and ambiguous letter X
                oddsRatioDfTableAApair2 = oddsRatioDfTableAApair.copy()
                oddsRatioDfTableAApair2.columns.name = 'aa1'
                oddsRatioDfTableAApair2 = oddsRatioDfTableAApair2[~oddsRatioDfTableAApair2.index.get_level_values('aa0')\
                                                                  .str.contains(r'[XU]')]
                oddsRatioDfTableAApair2 = oddsRatioDfTableAApair2.loc[:, ~oddsRatioDfTableAApair2.columns.str.contains(r'[XUB]')]
                # Sort the aa columns
                oddsRatioDfTableAApair2 = oddsRatioDfTableAApair2[aaTable]

                # Sort the aa rows
                oddsRatioDfTableAApair3 = oddsRatioDfTableAApair2.copy()
                oddsRatioDfTableAApair3['sortby'] = oddsRatioDfTableAApair3.index.get_level_values(1).map(dict(zip(aaTable, range(len(aaTable)))))
                oddsRatioDfTableAApair3 = oddsRatioDfTableAApair3.reset_index(level=1)
                oddsRatioDfTableAApair3 = oddsRatioDfTableAApair3.dropna(subset=['sortby'])
                oddsRatioDfTableAApair3 = oddsRatioDfTableAApair3.set_index('sortby', append=True)\
                                                                 .reorder_levels(['terminus', 'sortby', 'statistical_test', 'observable'])\
                                                                 .sort_index()
                oddsRatioDfTableAApair3 = oddsRatioDfTableAApair3.reset_index(level='sortby').set_index('aa0', append=True)\
                                                                 .reorder_levels(['terminus', 'aa0', 'statistical_test', 'observable'])\
                                                                 .drop('sortby', axis=1)
                oddsRatioDfTableAApair2 = oddsRatioDfTableAApair3.copy()
                # Save dataframes as files
                print('\n#### Writing oddsRatioDf AA pair files, start')
                filename = os.path.join(dataSpeciesGroupFolder, speciesSetName, outputFilePrefix + "_oddsRatioDfTableAApair." + dfFormatOutput)
                write_dataframe(oddsRatioDfTableAApair, filename, dfFormatOutput)
                filename = os.path.join(dataSpeciesGroupFolder, speciesSetName, outputFilePrefix + "_oddsRatioDfAApair." + dfFormatOutput)
                write_dataframe(oddsRatioDfAApair, filename, dfFormatOutput)
                print('\n#### Writing oddsRatioDf AA pair files, end')
                print('\n#### AA pairs bias analysis end')


                print('\n#### hexamer bias analysis start')
                # Note: the codon pair table is not created in the function.
                oddsRatioDfHexamer, _ = compute_odds_ratio_subseq(bulkFreqHexamer, None,
                                                                  ctermFreqHexamer, cTerminalSize,
                                                                  ntermFreqHexamer, nTerminalSize,
                                                                  subseqSize=6, seq_type='DNA',
                                                                  computeTheoreticalFreq=False)
                # Pivot the dataframe to create the codon pair table
                oddsRatioDfTableHexamer = oddsRatioDfHexamer.copy()
                # Filter out hexamers that contains ambiguous DNA letters
                oddsRatioDfTableHexamer = oddsRatioDfTableHexamer[~oddsRatioDfTableHexamer.index.\
                                                                  get_level_values('subseq').str.contains(r'[^ATGC]')]
                oddsRatioDfTableHexamer['codon_-2'] = oddsRatioDfTableHexamer.index.get_level_values(level=1).map(lambda x: x[:3])
                oddsRatioDfTableHexamer['codon_-1'] = oddsRatioDfTableHexamer.index.get_level_values(level=1).map(lambda x: x[3:])
                oddsRatioDfTableHexamer = oddsRatioDfTableHexamer.set_index(['codon_-1', 'codon_-2'], append=True)
                oddsRatioDfTableHexamer.index = oddsRatioDfTableHexamer.index.droplevel(1)
                oddsRatioDfTableHexamer = oddsRatioDfTableHexamer.reorder_levels([3, 4, 0, 1, 2])
                oddsRatioDfTableHexamer = oddsRatioDfTableHexamer.unstack('codon_-1')
                # Remove the "value" empty level
                oddsRatioDfTableHexamer.columns = oddsRatioDfTableHexamer.columns.droplevel(0)

                # Save dataframes as files
                print('\n#### Writing oddsRatioDf hexamer files, start')
                filename = os.path.join(dataSpeciesGroupFolder, speciesSetName, outputFilePrefix + "_oddsRatioDfTableHexamer." + dfFormatOutput)
                write_dataframe(oddsRatioDfTableHexamer, filename, dfFormatOutput)
                filename = os.path.join(dataSpeciesGroupFolder, speciesSetName, outputFilePrefix + "_oddsRatioDfHexamer." + dfFormatOutput)
                write_dataframe(oddsRatioDfHexamer, filename, dfFormatOutput)
                print('\n#### Writing oddsRatioDf hexamer files, end')
                print('\n#### hexamer bias analysis end')

                print('\n#### Analysis for stat:',statName,', end')



                print('\n#### Drawing plots, start')

                plotTitle = re.sub(r'_', r' ', outputFilePrefix)

                for terminus in ['N','C']:
                    plotData = compute_oddsratio_plot_data(oddsRatioDf, terminus)
                    maskData = compute_oddsratio_mask_data(oddsRatioDf, pvalueThresholds, terminus)
                    pvalAnnotTable = compute_pvalueAnnotation(oddsRatioDf, pvalueThresholds, terminus)
                    maskDataSmallN = compute_smallN_mask_data(oddsRatioDf, pvalueThresholds, terminus)
                    filenamePlot = os.path.join(plotsFolder, speciesSetName, outputFilePrefix + "_composition_bias_aa_" + terminus + "terminal" + ".png")
                    fig = plot_aa_composition_map(plotData, maskData, pvalAnnotTable, maskDataSmallN, terminus, plotTitle)
                    fig.savefig(filenamePlot, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                plt.close('all')

                if codonAnalysis:
                    for terminus in ['N','C']:
                        plotData = compute_codon_oddsratio_plot_data(oddsRatioDfCodon, terminus)
                        maskData = compute_codon_oddsratio_mask_data(oddsRatioDfCodon, pvalueThresholds, terminus)
                        pvalAnnotTable = compute_codon_pvalueAnnotation(oddsRatioDfCodon, pvalueThresholds, terminus)
                        maskDataSmallN = compute_codon_smallN_mask_data(oddsRatioDfCodon, pvalueThresholds, terminus)

                        # # Draw codon biases for one position using HTML genetic code-like codon table
                        # posList = [1, 2] if terminus == 'N' else [-1,-2]
                        # for position in posList:
                        #     fig = plot_codon_table(plotData, pvalAnnotTable, terminus, position, plotTitle,
                        #                            codonTableBio, plotsFolder)

                        #     filenamePlot = outputFilePrefix + '_composition_bias_codon_' + terminus + 'terminal_table_pos' + '{:03d}'.format(position)
                        #     fig.savefig(os.path.join(plotsFolder, speciesSetName, filenamePlot + '.png'), dpi=400, bbox_inches="tight")
                        #     plt.close(fig)

                        # Draw codon biases for all positions using heatmap
                        fig = plot_codon_composition_map(plotData, maskData, pvalAnnotTable, maskDataSmallN, terminus,
                                                         plotTitle, refCodonTableDf)
                        filenamePlot = outputFilePrefix + '_composition_bias_codon_' + terminus + 'terminal_heatmap'
                        fig.savefig(os.path.join(plotsFolder, speciesSetName, filenamePlot + '.png'), dpi=300, bbox_inches="tight")
                        plt.close(fig)
                    plt.close('all')

                # for terminus in ['N','C','bulk']:
                for terminus in ['N','C']:
                    for statistical_test in ['theoretical', 'contingency']:
                        if terminus != 'bulk' or statistical_test != 'contingency':
                            print("terminus", terminus)
                            print("oddsRatioDfTableAApair2.head()\n:", oddsRatioDfTableAApair2.head())
                            plotData = compute_subseq_oddsratio_plot_data(oddsRatioDfTableAApair2, terminus, statistical_test)
                            maskData = compute_subseq_oddsratio_mask_data(oddsRatioDfTableAApair2, pvalueThresholds,
                                                                          terminus, statistical_test, subseq_type='aa')
                            pvalAnnotTable = compute_subseq_pvalueAnnotation(oddsRatioDfTableAApair2, pvalueThresholds,
                                                                             terminus, statistical_test)
                            maskDataSmallN = compute_subseq_smallN_mask_data(oddsRatioDfTableAApair2, pvalueThresholds,
                                                                             terminus, statistical_test)
                            filenamePlot = os.path.join(plotsFolder, speciesSetName, outputFilePrefix +
                                                        "_composition_bias_aa_pairs_" + terminus + "terminal" + '_statistical_test_' +
                                                        statistical_test + ".png")
                            plotData.columns.name = 'aa1'
                            fig = plot_seq_pair_composition_map(plotData, maskData, pvalAnnotTable, maskDataSmallN, terminus,
                                                                statistical_test, plotTitle, subseq_type='aa', refCodonTableDf=refCodonTableDf)
                            fig.savefig(filenamePlot, dpi=300, bbox_inches="tight")
                            plt.close(fig)

                for terminus in ['N','C']:
                    plotData = compute_subseq_oddsratio_plot_data(oddsRatioDfTableHexamer, terminus, statistical_test)
                    maskData = compute_subseq_oddsratio_mask_data(oddsRatioDfTableHexamer, pvalueThresholds,
                                                                  terminus, statistical_test, subseq_type='codon')
                    pvalAnnotTable = compute_subseq_pvalueAnnotation(oddsRatioDfTableHexamer, pvalueThresholds,
                                                                     terminus, statistical_test)
                    maskDataSmallN = compute_subseq_smallN_mask_data(oddsRatioDfTableHexamer, pvalueThresholds,
                                                                     terminus, statistical_test)

                    for plotSynonymousCodonGroupSeparationLine in [True, False]:
                        fig = plot_seq_pair_composition_map(plotData, maskData, pvalAnnotTable, maskDataSmallN, terminus,
                                                            statistical_test, plotTitle, subseq_type='codon',
                                                            refCodonTableDf=refCodonTableDf,
                                                            plotSynonymousCodonGroupSeparationLine=plotSynonymousCodonGroupSeparationLine);
                        if plotSynonymousCodonGroupSeparationLine:
                            suffix2 = '_withSynGroupLines'
                        else:
                            suffix2 = ''
                        filenamePlot = os.path.join(plotsFolder, speciesSetName, outputFilePrefix +
                                                    "_composition_bias_codon_pairs_" + terminus + "terminal" + '_statistical_test_' +
                                                    statistical_test + suffix2 + ".png")
                        fig.savefig(filenamePlot, dpi=300, bbox_inches="tight")

                    plt.close(fig)

                plt.close('all')

                print('\n#### Drawing plots, end')

                print('\n#### Statistics', statName, 'total nb sequences: ', nSeq)
                

            
    print("\n\n######## END ANALYSIS ########")
    
    return multispeciesAllCDSDf_nr
