# Author: Marc Weber

import pandas as pd
idx = pd.IndexSlice
import scipy
from statsmodels.sandbox.stats.multicomp import multipletests
import argparse
from pathlib import Path
import re
import json
from multiprocessing import Process, Pool
import os

from cterminal import simplify_prefix
from cterminal import full_analysis_multispecies
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
analysisCtermPath = p.analysisCtermPath
conservationPath = analysisCtermPath / 'Conservation5'

aaTable = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'U', 'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V']



def enrichment_analysis_cterminal_subset(taxonAllCDSDf, taxonId, groupCol, groupName, groupValuesList, aaTable,
                                         alpha_family_wise_error_rate, enrichmentTestcolumnName):
    """
    We group protein sequences by group (COG, orthogroup, etc) and compute the enrichment
    of all the protein subsets with a specific amino acid at C-terminal.
    """

    # We create a dataframe with multiindex in both rows and columns, to organize the information in the following way:
    # rows index:
    #   taxon_id / group_index
    # columns index:
    #   C-terminal amino acid / observable
    groupsMultiIndex = pd.MultiIndex.from_product([[taxonId], groupValuesList],
                                                  names=['taxon_id', groupName + '_index'])

    colName_count_cterm_in_group = 'count_cterm_in_group'
    colName_count_NOT_cterm_in_group = 'count_NOT_cterm_in_group'
    colName_count_cterm_NOT_in_group = 'count_cterm_NOT_in_group'
    colName_count_NOT_cterm_NOT_in_group = 'count_NOT_cterm_NOT_in_group'

    ctermMultiIndex = pd.MultiIndex.from_product([aaTable,
                                                  [colName_count_cterm_in_group,
                                                   colName_count_NOT_cterm_in_group,
                                                   colName_count_cterm_NOT_in_group,
                                                   colName_count_NOT_cterm_NOT_in_group,
                                                   'odds_ratio',
                                                   'pvalue_fisher',
                                                   enrichmentTestcolumnName
                                                   ]],
                                                 names=['cterm_aa', 'observable'])
    taxonGroupEnrichDf = pd.DataFrame(index=groupsMultiIndex, columns=ctermMultiIndex)
    taxonGroupEnrichDf = taxonGroupEnrichDf.sort_index(axis=0).sort_index(axis=1)


    for ctermAA in aaTable:
        print("#######", ctermAA)

        # Select proteins with ctermaAA at C-terminal
        subsetBoolDf = taxonAllCDSDf.apply(lambda row: row['proteinSeq'][-1] == ctermAA, axis=1)
        proteinSubsetDf = taxonAllCDSDf[subsetBoolDf]
        proteinNotSubsetDf = taxonAllCDSDf[~subsetBoolDf]
        nProteinSubset = len(proteinSubsetDf)
        nProteinNotSubset = len(proteinNotSubsetDf)

        for groupIndex in groupValuesList:
            # Select one group and compute enrichment

            # Nb of proteins with C-terminal AA that is included in the group.
            # *IMPORTANT*: We include all proteins that **contains as a string the group index**
            #              within the list of group values.
            # We use == True in order to set NaN values to False. Here we want a bi-partition
            # of all records, such that all records that are not in the group index, including
            # NaN values, belong to the second group.
            sel = (proteinSubsetDf[groupCol].str.contains(groupIndex) == True)
            NjA = len(proteinSubsetDf[sel])
            
            # Nb of proteins with C-terminal AA that is not included in the group.
            NjNotA = len(proteinSubsetDf[~sel])
            # print("groupIndex", groupIndex, "NjA", NjA, "NjNotA", NjNotA)

            # Nb of proteins with other C-terminal AA that is included in the group.
            # We use == True in order to set NaN values to False.
            sel = (proteinNotSubsetDf[groupCol].str.contains(groupIndex) == True)
            NrefA = len(proteinNotSubsetDf[sel])
            
            # Nb of proteins with other C-terminal AA that is not included in the group.
            NrefNotA = len(proteinNotSubsetDf[~sel])

            contingencyTable = [[NjA, NrefA], [NjNotA, NrefNotA]]
            oddsRatio, pvalue = scipy.stats.fisher_exact(contingencyTable, alternative='two-sided')
            #print(groupIndex, oddsRatio, pvalue)

            taxonGroupEnrichDf.loc[(taxonId, groupIndex), (ctermAA, colName_count_cterm_in_group)] = NjA
            taxonGroupEnrichDf.loc[(taxonId, groupIndex), (ctermAA, colName_count_cterm_NOT_in_group)] = NjNotA
            taxonGroupEnrichDf.loc[(taxonId, groupIndex), (ctermAA, colName_count_NOT_cterm_in_group)] = NrefA
            taxonGroupEnrichDf.loc[(taxonId, groupIndex), (ctermAA, colName_count_NOT_cterm_NOT_in_group)] = NrefNotA

            taxonGroupEnrichDf.loc[(taxonId, groupIndex), (ctermAA, 'odds_ratio')] = oddsRatio
            taxonGroupEnrichDf.loc[(taxonId, groupIndex), (ctermAA, 'pvalue_fisher')] = pvalue

        # Multiple test correction

        pvalueList = taxonGroupEnrichDf.loc[(taxonId), (ctermAA, 'pvalue_fisher')].dropna()
        pvalueList.name = pvalueList.name[1]

        reject, pvals_corrected, alphacSidak, alphacBonf = \
            multipletests(pvalueList, alpha=alpha_family_wise_error_rate, method='fdr_bh',
                          is_sorted=False, returnsorted=False)

        # The list of reject_null_hypotheses booleans has the same order as the original pvalue list,
        # so we assign the same index to `reject` dataframe.
        pvalueListCorrected = pd.concat([pd.DataFrame(pvalueList),
                                         pd.DataFrame(reject, index=pvalueList.index, columns=[enrichmentTestcolumnName])],
                                        axis=1)
        pvalueListCorrected.index = pd.MultiIndex.from_product([[taxonId], pvalueListCorrected.index])
        pvalueListCorrected.columns = pd.MultiIndex.from_product([[ctermAA], pvalueListCorrected.columns])
        taxonGroupEnrichDf.loc[(taxonId), (ctermAA, enrichmentTestcolumnName)] = pvalueListCorrected.loc[:, (ctermAA, enrichmentTestcolumnName)]
    
    return taxonGroupEnrichDf


#==============================================================================


def compute_bias_for_COG_group(iGroup, taxonAllCDSDf, taxonSuffix, nJobs=0):
    print("iGroup:", iGroup, "/", nJobs - 1, 'process id:', os.getpid())
    groupCol = 'COG cat'
    groupIndex = COGList[iGroup]
    sel = taxonAllCDSDf[groupCol].str.contains(groupIndex) == True
    groupCDSDf = taxonAllCDSDf[sel]
    groupComplementCDSDf = taxonAllCDSDf[~sel]
    print("groupIndex", groupIndex, "len(taxonAllCDSDf)", len(taxonAllCDSDf), "len(groupCDSDf)", len(groupCDSDf),
          "len(groupComplementCDSDf)", len(groupComplementCDSDf))

    if len(groupCDSDf) > 0:
        outputPath = \
            analysisEggnogPath / 'Analysis_COG_groups_cterminal_bias' / 'COG_{}'.format(groupIndex)
        full_analysis_multispecies(
            refSeqFolder=str(refSeqPath),
            dataSingleSpeciesFolder=str(analysisCtermDataPath / 'Species'),
            dataSpeciesGroupFolder=str(outputPath),
            plotsFolder=str(outputPath),
            speciesSetName=taxonSuffix,
            chooseRandomGenomes=False,
            nRandomGenomes=0,
            assemblyAccessionList=None,
            speciesList=None,
            multispeciesAllCDSDf_nr=groupCDSDf,
            dfFormatOutput='csv',
            skipAnalysisExistingDataFiles=False,
            skipStatisticsAnalysis=False,
            method='allSeqDataframe',
            skipClustering=False,
            GOanalysis=False, amiGODf=None, membraneGOTermsDf=None)
    else:
        print("empty dataframe, skipping COG group analysis.")

    if len(groupComplementCDSDf) > 0:
        outputPath = \
            analysisEggnogPath / 'Analysis_COG_groups_cterminal_bias' / 'COG_{}_complement'.format(groupIndex)
        full_analysis_multispecies(
            refSeqFolder=str(refSeqPath),
            dataSingleSpeciesFolder=str(analysisCtermDataPath / 'Species'),
            dataSpeciesGroupFolder=str(outputPath),
            plotsFolder=str(outputPath),
            speciesSetName=taxonSuffix,
            chooseRandomGenomes=False,
            nRandomGenomes=0,
            assemblyAccessionList=None,
            speciesList=None,
            multispeciesAllCDSDf_nr=groupComplementCDSDf,
            dfFormatOutput='csv',
            skipAnalysisExistingDataFiles=False,
            skipStatisticsAnalysis=False,
            method='allSeqDataframe',
            skipClustering=False,
            GOanalysis=False, amiGODf=None, membraneGOTermsDf=None)
    else:
        print("empty dataframe, skipping COG group analysis.")

    print("##### END ANALYSIS #####", "iGroup:", iGroup, "/", nJobs, 'process id:', os.getpid())


#==============================================================================


if __name__ == '__main__':

    # Run parallel jobs
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--iJob', type=int, default=None)
    arg_parser.add_argument('--iTaxon', type=int, default=None)
    arg_parser.add_argument('--iGroup', type=int, default=None)
    arg_parser.add_argument('--taxaSet', type=str, default='full')
    arg_parser.add_argument('--target', type=str, default=None)
    args = arg_parser.parse_args()

    iJob = args.iJob
    if iJob is not None:
        iJob = iJob - 1

    iTaxon = args.iTaxon
    if iTaxon is not None:
        iTaxon = iTaxon - 1

    iGroup = args.iGroup
    if iGroup is not None:
        iGroup = iGroup - 1

    taxaSet = args.taxaSet
    target = args.target

    COGList = ['J', 'U', 'Y', 'P', 'T', 'A', 'M', 'F', 'W', 'V', 'K', 'G', 'D', 'H', 'E', 'Q', 'B', 'Z', 'N', 'S', 'I', 'O', 'L', 'R', 'C']
    # COGList = ['U', 'Y', 'P', 'T', 'A', 'M', 'F', 'W', 'V', 'K', 'G', 'D', 'H', 'E', 'Q', 'B', 'Z', 'N', 'S', 'I', 'O', 'L', 'R', 'C']
    nCOG = len(COGList)

    # In the case of target == 'COG_enrichment', we run one job per taxon
    if target == 'COG_enrichment':
        # We must set the iTaxon argument
        nJobs = None
        if iTaxon is None:
            raise ValueError("iTaxon should be defined.")

    # In the case of target == 'bias_COG_groups', we run one job per (group, taxon) pair
    if target == 'bias_COG_groups':
        if iJob is not None:
            # iJob starts at 0.
            # Taxon1: 0...24
            # Taxon2: 25...49
            iTaxon = int(iJob / nCOG)
            iGroup = iJob % nCOG
        else:
            if iTaxon is None or iGroup is None:
                raise ValueError("iTaxon and iGroup should be defined.")

    # Import list of taxon levels, names and taxids
    taxaEggnogDf = pd.read_csv(str(analysisEggnogPath / 'taxaEggnogDf.csv'), index_col=0)
    print("len(taxaEggnogDf):", len(taxaEggnogDf))


    # Choose different subset of taxonomic groups to adjust memory usage
    nGenomesThreshold = 120

    if taxaSet == 'LowMem':
        taxaDf = taxaEggnogDf[taxaEggnogDf['n_genomes_in_refseq_eggnog_common_species'] < nGenomesThreshold]
        taxonGroup = taxaDf.iloc[iTaxon]

    elif taxaSet == 'HighMem':
        taxaDf = taxaEggnogDf[taxaEggnogDf['n_genomes_in_refseq_eggnog_common_species'] >= nGenomesThreshold]
        taxonGroup = taxaDf.iloc[iTaxon]

    elif taxaSet == 'full':
        taxaDf = taxaEggnogDf
        taxonGroup = taxaDf.iloc[iTaxon]

    elif taxaSet == 'incomplete':
        pass

    print("taxaDf.head():\n", taxaDf.head())
    print("len(taxaDf):", len(taxaDf))

    if target == 'bias_COG_groups':
        nJobs = nCOG * len(taxaDf)

    print("nJobs = nCOG * len(taxaDf):", nJobs, "len(taxaDf):", len(taxaDf), "nCOG:", nCOG, "iJob:",
          iJob, "iTaxon:", iTaxon, "iGroup:", iGroup)

    
    print("taxonGroup:\n", taxonGroup)
    taxonRank = taxonGroup['rank_name']
    taxonName = taxonGroup['taxon_name']
    taxonId = int(taxonGroup['taxon_id'])
    taxonSuffix = simplify_prefix('taxaId_{:d}'.format(taxonId).strip("'"))
    print("taxonSuffix", taxonSuffix)

    # Importing list of genome assemblies with their taxonomy in the RefSeq - EggNOG common set of species
    assemblySummaryCommonSpeciesDf = pd.read_csv(str(analysisEggnogPath / 'assemblySummaryCommonSpeciesDf.csv'))

    if target == 'COG_enrichment':
        # Import all protein sequences in taxon group from file
        multispeciesAllCDSDf_nr = pd.read_csv(str(analysisCtermDataPath / 'Taxonomy' / taxonSuffix / (taxonSuffix + '_multispeciesAllCDSDf_nr.csv')))
        print("len(multispeciesAllCDSDf_nr)", len(multispeciesAllCDSDf_nr))

        # Choose protein sequences from genomes that are in the common set of species RefSeq - EggNOG
        multispeciesAllCDSDf_nr2 = pd.merge(multispeciesAllCDSDf_nr,
                                            assemblySummaryCommonSpeciesDf.set_index('assembly_accession')[['species_taxid', 'taxid']],
                                            left_on='genome_accession', right_index=True, how='left')
        taxonAllCDSEggNOGSpeciesDf = multispeciesAllCDSDf_nr2.dropna(subset=['taxid'], axis=0)
        nSeqTaxonRefSeq = len(multispeciesAllCDSDf_nr2)
        nSeqTaxonRefSeqEggNOGCommon = len(taxonAllCDSEggNOGSpeciesDf)
        print("nSeqTaxonRefSeqEggNOGCommon:", nSeqTaxonRefSeqEggNOGCommon, "nSeqTaxonRefSeq", nSeqTaxonRefSeq,
              "ratio", nSeqTaxonRefSeqEggNOGCommon/nSeqTaxonRefSeq)
        print("Nb of subspecies in taxon group, RefSeq:", len(set(multispeciesAllCDSDf_nr2['taxid'].dropna())))
        print("Nb of taxid at rank species in taxon group, RefSeq:", len(set(multispeciesAllCDSDf_nr2['species_taxid'].dropna())))
        print("Nb of subspecies in taxon group, common:", len(set(taxonAllCDSEggNOGSpeciesDf['taxid'])))
        print("Nb of taxid at rank species, common:", len(set(taxonAllCDSEggNOGSpeciesDf['species_taxid'])))

        # Import eggnog-mapper protein annotations
        eggnogMapperAnnotationResultDf = pd.read_csv(str(analysisEggnogPath / 'eggnogMapperAnnotationResultDf.csv'))
        eggnogMapperAnnotationResultDf.drop('Unnamed: 0', axis=1, inplace=True)
        eggnogMapperOrthogroupAnnotationDf = pd.read_csv(str(analysisEggnogPath / 'eggnogMapperOrthogroupAnnotationDf.csv'))

        # Merging eggnog-mapper annotations with the taxonAllCDS
        taxonAllCDSDf = pd.merge(taxonAllCDSEggNOGSpeciesDf.reset_index(),
                                 eggnogMapperAnnotationResultDf,
                                 on='refSeqProteinId', how='left', suffixes=['_refseq', '_eggnog'])
        taxonAllCDSDf.drop('index', axis=1, inplace=True)
        taxonAllCDSDf.drop('genome_accession_eggnog', axis=1, inplace=True)
        taxonAllCDSDf.rename(columns={'genome_accession_refseq':'genome_accession'}, inplace=True)
        taxonAllCDSDf.drop('taxid_eggnog', axis=1, inplace=True)
        taxonAllCDSDf.rename(columns={'taxid_refseq':'taxid'}, inplace=True)
        print("taxonAllCDSDf:\n", taxonAllCDSDf.head(), "\n")

        print("Nb of protein sequences with EggNOG orthogroup annotation:",
              taxonAllCDSDf['COG cat'].notnull().sum(), "/", len(taxonAllCDSDf), "=",
              taxonAllCDSDf['COG cat'].notnull().sum()/len(taxonAllCDSDf))

    elif target == 'bias_COG_groups':
        print("Importing full multiCDS_eggNOG_taxa...")
        eggResDf3 = (pd.read_csv(analysisEggnogPath / 'multiCDS_eggNOG_taxa.csv.gz', compression='gzip', nrows=None, index_col=0)
                     .rename(columns={'species_name_x':'species_name'}))
        print("Importing full multiCDS_eggNOG_taxa... finished.")
        print("eggResDf3.head():\n", eggResDf3.head())

        # Select genomes in the taxon group
        eggResDf4 = eggResDf3
        eggResDf4 = eggResDf4[eggResDf4[taxonRank + '_taxid'] == taxonId]
        taxonAssemblyAccessionList = list(eggResDf4['genome_accession'].unique())
        print("len(taxonAssemblyAccessionList)", len(taxonAssemblyAccessionList))
        taxonAllCDSDf = eggResDf4


    if target == 'COG_enrichment':

        # ### COG enrichment analysis

        groupCol = 'COG cat'
        groupName = 'COGGroup'

        # False discovery rate for the multiple tests correction
        alpha_family_wise_error_rate = 0.01
        enrichmentTestcolumnName = 'reject_null_multiple_tests_correction_Benjamini-Hochberg_FDR_' + str(alpha_family_wise_error_rate)

        # Compute enrichment
        taxonGroupEnrichDf = \
            enrichment_analysis_cterminal_subset(taxonAllCDSDf, taxonId, groupCol, groupName, COGList, aaTable,
                                                 alpha_family_wise_error_rate, enrichmentTestcolumnName)
        print("taxonGroupEnrichDf:\n", taxonGroupEnrichDf.head())

        taxonGroupEnrichFilename = '{}_enrichment_COG.csv'.format(taxonSuffix)
        taxonGroupEnrichDf.to_csv(str(analysisEggnogPath / 'Analysis_COG_enrichment' / taxonGroupEnrichFilename))

        # Rearrange enrichment Df (stack and filter)

        # Stack the dataframe for c-terminal AA columns
        taxonGroupEnrichStackedDf = taxonGroupEnrichDf.copy()
        taxonGroupEnrichStackedDf = taxonGroupEnrichStackedDf.stack(level=['cterm_aa']).reset_index()
        taxonGroupEnrichStackedDf.columns.name = ''
        taxonGroupEnrichStackedDf.sort_values(['cterm_aa', groupName + '_index'], inplace=True)

        # Read COG categories descriptions
        with open(str(eggNOGPath / 'COG_functional_categories.txt')) as f:
            text = f.read()
            COGCategoriesDict = dict(re.findall(r'\[([A-Z])\]\s+(.*)\s+[$\n]', text))

        # Add COG category description
        taxonGroupEnrichStackedDf['COG_cat_description'] = \
            taxonGroupEnrichStackedDf.apply(lambda x:
                                            COGCategoriesDict.get(x['COGGroup_index'], None),
                                            axis=1)

        # Select only enriched results
        taxonGroupEnrichStackedDf = taxonGroupEnrichStackedDf[taxonGroupEnrichStackedDf[enrichmentTestcolumnName] == True]

        # Change column order
        taxonGroupEnrichStackedDf = taxonGroupEnrichStackedDf[[
            'taxon_id', 'COGGroup_index', 'COG_cat_description', 'cterm_aa',
            'count_NOT_cterm_NOT_in_group', 'count_NOT_cterm_in_group',
            'count_cterm_NOT_in_group', 'count_cterm_in_group', 'odds_ratio',
            'pvalue_fisher',
            enrichmentTestcolumnName]]

        taxonGroupEnrichStackedFilename = '{}_enrichment_COG_stacked.csv'.format(taxonSuffix)
        taxonGroupEnrichStackedDf.to_csv(str(analysisEggnogPath / 'Analysis_COG_enrichment' / taxonGroupEnrichStackedFilename))
        print("taxonGroupEnrichStackedDf:\n", taxonGroupEnrichStackedDf.head())

        # raise SystemExit

        # ### Functional enrichment analysis, orthologous groups

        # We will consider the "Best matching Orthologous Groups" as given by the HMM search and fine-grained orthology assignment by the eggmapper software.

        groupCol = 'bestOG'
        groupName = 'orthologous_group'

        # Extract list of COG categories
        orthogroupList = list(taxonAllCDSDf[groupCol].dropna().unique())

        taxonEnrichOrthoGroupDf = enrichment_analysis_cterminal_subset(taxonAllCDSDf, taxonId, groupCol, groupName,
                                                                       orthogroupList, aaTable,
                                                                       alpha_family_wise_error_rate, enrichmentTestcolumnName)
        taxonEnrichOrthoGroupFilename = '{}_enrichment_orthogroup.csv'.format(taxonSuffix)
        taxonEnrichOrthoGroupDf.to_csv(str(analysisEggnogPath / 'Analysis_COG_enrichment' / taxonEnrichOrthoGroupFilename))
        print("taxonEnrichOrthoGroupDf:\n", taxonEnrichOrthoGroupDf.head())


        # Stack the dataframe for c-terminal AA columns
        taxonEnrichOrthoGroupStackedDf = taxonEnrichOrthoGroupDf.copy()
        taxonEnrichOrthoGroupStackedDf = taxonEnrichOrthoGroupStackedDf.stack(level=['cterm_aa']).reset_index()
        taxonEnrichOrthoGroupStackedDf.columns.name = ''
        taxonEnrichOrthoGroupStackedDf.sort_values(['cterm_aa', groupName + '_index'], inplace=True)

        # Add orthogroup annotation
        taxonEnrichOrthoGroupStackedDf = \
            pd.merge(taxonEnrichOrthoGroupStackedDf, eggnogMapperOrthogroupAnnotationDf, left_on='orthologous_group_index',
                     right_on='bestOG', how='left')

        # Select only enriched results
        taxonEnrichOrthoGroupStackedDf = taxonEnrichOrthoGroupStackedDf[taxonEnrichOrthoGroupStackedDf[enrichmentTestcolumnName] == True]

        # Change column order
        taxonEnrichOrthoGroupStackedDf = taxonEnrichOrthoGroupStackedDf[
            ['taxon_id', 'cterm_aa', 'orthologous_group_index', 'predicted_gene_name', 'KEGG_pathways', 'COG cat', 'eggNOG annot',
             'count_NOT_cterm_NOT_in_group', 'count_NOT_cterm_in_group',
             'count_cterm_NOT_in_group', 'count_cterm_in_group', 'odds_ratio',
             'pvalue_fisher',
             enrichmentTestcolumnName]]

        taxonEnrichOrthoGroupStackedFilename = '{}_enrichment_orthogroup_stacked.csv'.format(taxonSuffix)
        taxonEnrichOrthoGroupStackedDf.to_csv(str(analysisEggnogPath / 'Analysis_COG_enrichment' / taxonEnrichOrthoGroupStackedFilename))
        print("taxonEnrichOrthoGroupStackedDf:\n", taxonEnrichOrthoGroupStackedDf.head())


    if target == 'bias_COG_groups':

        compute_bias_for_COG_group(iGroup=iGroup, taxonAllCDSDf=taxonAllCDSDf, taxonSuffix=taxonSuffix, nJobs=nJobs)

        # Parallel computation of the bias in COG groups
        # print(COGList)
        # nThreads = 8
        # nJobs = len(COGList)
        # pool = Pool(processes=nThreads)
        # for iGroup in range(nJobs):
        #     pool.apply_async(compute_bias_for_COG_group, args=(iGroup,))
        # pool.close()
        # pool.join()


    print("\n\n######## END ANALYSIS JOB ########")
