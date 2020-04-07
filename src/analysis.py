#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import re

from cterminal import full_analysis_multispecies
from cterminal import simplify_prefix
from mwTools.paths import p



p = p()
print(p.rootPath)
mpnAnnotationPath = p.mpnAnnotationPath
rootPath = p.rootPath
cterminalPath = p.cterminalPath
refSeqPath = p.refSeqPath
taxonomyPath = p.taxonomyPath
phylaPath = p.phylaPath
entrezGenePath = p.entrezGenePath
OMAPath = p.OMAPath
eggNOGPath = p.eggNOGPath
NCBI_COG_path = p.NCBI_COG_path
analysisEggnogPath = p.analysisEggnogPath
analysisCtermDataPath = p.analysisCtermDataPath
analysisCtermPlotsPath = p.analysisCtermPlotsPath


# Importing list of genome assemblies of the RefSeq database with their taxonomy
assemblySummaryRepTaxonomyDf = pd.read_csv(str(taxonomyPath / 'assemblySummaryRepTaxonomyDf.csv'))
assemblySummaryRepTaxonomyDf.set_index('assembly_accession', inplace=True)

# Run parallel jobs
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--iJob', type=int, default=1)
arg_parser.add_argument('--taxaId', type=int, default=None)
arg_parser.add_argument('--nJob', type=int, default=1)
arg_parser.add_argument('--option', type=str, default=None)
arg_parser.add_argument('--run', type=str, default=None)
arg_parser.add_argument('--subrun', type=str, default=None)
arg_parser.add_argument('--method', type=str, default='iterative')
args = arg_parser.parse_args()
iJob = args.iJob - 1
nJob = args.nJob

# run: taxon, compute_allCDSDf
# subrun: multispeciesDf, bias
run = args.run
subrun = args.subrun
skipJob = False

if run == 'taxon':

    # Import list of taxon levels, names and taxids
    taxaDf = pd.read_csv(str(taxonomyPath / 'taxaDf.csv'), index_col=0)
    print("len(taxaDf)", len(taxaDf))

    if args.taxaId is None:
        taxaSet = 'full'
        taxaSet = 'incomplete_set'
        taxaSet = 'full'
        if taxaSet == 'full':

            print("args.option", args.option)
            # Choose different subset of taxonomic groups to adjust memory usage
            # HighMem, LowMem, VeryLowMem
            taxaDf = pd.read_csv(str(taxonomyPath / 'taxaDf{}.csv'.format(args.option)), index_col=0)
            print('taxaDf{}.csv length:'.format(args.option), len(taxaDf))
            if iJob >= len(taxaDf):
                raise ValueError("iJob is larger than the length of taxaDf.")
            if nJob != len(taxaDf):
                raise ValueError("nJob should be the same as the length of taxaDf.")
            taxonGroup = taxaDf.iloc[iJob]

        elif taxaSet == 'incomplete_set':

            # Incomplete output of data files
            pass
    else:
        taxonGroup = taxaDf[taxaDf['taxon_id'] == args.taxaId].iloc[0]

    print("taxonGroupRef:", taxonGroup)
    taxonRank = taxonGroup['rank_name']
    taxonName = taxonGroup['taxon_name']
    taxonId = int(taxonGroup['taxon_id'])

    # Select genomes in the taxon group
    taxonAssemblyDf = assemblySummaryRepTaxonomyDf[assemblySummaryRepTaxonomyDf[taxonRank + '_taxid'] == taxonId]
    taxonAssemblyAccessionList = list(taxonAssemblyDf.index)

    taxonSuffix = simplify_prefix('taxaId_{:d}'.format(taxonId).strip("'"))
    print("output path:", str(analysisCtermDataPath / taxonSuffix))
    suffix = taxonSuffix

    if subrun == 'bias':

        method = args.method

        if method == 'allSeqDataframe':
            # Import the multispeciesAllCDSDf_nr dataframe
            filePath = analysisCtermDataPath / 'Taxonomy' / taxonSuffix / (taxonSuffix + '_multispeciesAllCDSDf_nr.csv')
            if filePath.exists():
                multispeciesAllCDSDf_nr = pd.read_csv(str(filePath))
            else:
                multispeciesAllCDSDf_nr = None

        elif method == 'iterative':
            multispeciesAllCDSDf_nr = None

        assemblyAccessionList = taxonAssemblyAccessionList
        skipAnalysisExistingDataFiles = True
        skipStatisticsAnalysis = False


    elif subrun == 'multispeciesDf':

        # Check if multispeciesDf file already exists.
        multispeciesDfFileExists = (analysisCtermDataPath / 'Taxonomy' / taxonSuffix /
                                    (taxonSuffix + '_multispeciesAllCDSDf_nr.csv')).is_file()
        skipJob = multispeciesDfFileExists
        if multispeciesDfFileExists:
            print("multispeciesDfFileExists, skipping job.")

        method = 'allSeqDataframe'
        multispeciesAllCDSDf_nr = None
        assemblyAccessionList = taxonAssemblyAccessionList
        skipAnalysisExistingDataFiles = True
        skipStatisticsAnalysis = True



if run == 'compute_allCDSDf':

    # Split all genomes into blocks for parallel processing
    assemblyAccessionBlockList = np.array_split(assemblySummaryRepTaxonomyDf.index, nJob)
    assemblyAccessionList = list(assemblyAccessionBlockList[iJob])

    multispeciesAllCDSDf_nr = None
    suffix = 'dummy'
    skipAnalysisExistingDataFiles = False
    skipAnalysisExistingDataFiles = True
    skipStatisticsAnalysis = True
    method = 'iterative'


if not skipJob:

    full_analysis_multispecies(refSeqFolder=str(refSeqPath),
                               dataSingleSpeciesFolder=str(analysisCtermDataPath / 'Species'),
                               dataSpeciesGroupFolder=str(analysisCtermDataPath / 'Taxonomy'),
                               plotsFolder=str(analysisCtermDataPath / 'Taxonomy'),
                               speciesSetName=suffix,
                               chooseRandomGenomes=False,
                               nRandomGenomes=0,
                               assemblyAccessionList=assemblyAccessionList,
                               speciesList=[],
                               multispeciesAllCDSDf_nr=multispeciesAllCDSDf_nr,
                               dfFormatOutput='csv',
                               skipAnalysisExistingDataFiles=skipAnalysisExistingDataFiles,
                               skipStatisticsAnalysis=skipStatisticsAnalysis,
                               method=method,
                               skipClustering=False,
                               GOanalysis=False, amiGODf=None, membraneGOTermsDf=None
                               )

print("\n\n######## END ANALYSIS JOB ########")
