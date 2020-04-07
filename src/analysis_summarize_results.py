import pandas as pd
from pathlib import Path

from mwTools.paths import p
from cterminal import simplify_prefix

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


def gather_cterm_bias_taxon_results(taxaGroupsList):

    print("Importing C-terminal bias results for all taxon groups.")

    oddsRatioCtermTaxonList = []

    # Importing list of genome assemblies of the RefSeq database with their taxonomy
    assemblySummaryRepTaxonomyDf = pd.read_csv(str(taxonomyPath / 'assemblySummaryRepTaxonomyDf.csv'))
    assemblySummaryRepTaxonomyDf.set_index('assembly_accession', inplace=True)


    for i, taxaGroup in enumerate(taxaGroupsList):

        print("### ", i+1, "/", len(taxaGroupsList), "taxonGroup:", taxaGroup)
        taxonRank = taxaGroup[2]
        taxonName = taxaGroup[3]
        taxonId = int(taxaGroup[4])

        taxonSuffix = simplify_prefix('taxaId_{:d}_{}_{}'.format(taxonId, taxonRank, taxonName).strip("'"))
        
        taxonAssemblyDf = assemblySummaryRepTaxonomyDf[assemblySummaryRepTaxonomyDf[taxonRank + '_taxid'] == taxonId]
        
        taxonDirPath = analysisCtermDataPath / taxonSuffix
        if not (analysisCtermDataPath / taxonSuffix).is_dir():
            print("ERROR: taxon folder does not exist: ", taxonSuffix)
        
        oddsRatioDfPath = taxonDirPath / (taxonSuffix + '_subset_all_oddsRatioAADf.csv')
        
        if oddsRatioDfPath.is_file():

            # Read total nb of sequences from statistics summary file for taxon
            taxonNSeq = None
            multispeciesStatisticsSummaryDfPath = taxonDirPath / (taxonSuffix + '_subset_all_multispeciesStatisticsSummary.csv')
            if multispeciesStatisticsSummaryDfPath.is_file():
                summaryDf = pd.read_csv(multispeciesStatisticsSummaryDfPath)
                if not summaryDf.empty:
                    taxonNSeq = summaryDf['nSeq'].sum()
                    
            # Read nb of genomes for taxon
            taxonNGenomes = len(taxonAssemblyDf)

            # Import odds ratio dataframe for taxon
            oddsRatioDf = pd.read_csv(oddsRatioDfPath)

            oddsRatioCtermDf = oddsRatioDf[(oddsRatioDf['position from terminus'] == -1) & (oddsRatioDf['terminus'] == 'C')]
            oddsRatioCtermDf2 = oddsRatioCtermDf.copy()
            oddsRatioCtermDf2['taxaid'] = taxonId
            oddsRatioCtermDf2['taxon_name'] = taxonName
            oddsRatioCtermDf2['taxon_rank'] = taxonRank
            oddsRatioCtermDf2['n_seq'] = taxonNSeq
            oddsRatioCtermDf2['n_genomes'] = taxonNGenomes

            oddsRatioCtermTaxonList.append(oddsRatioCtermDf2)
            
        else:
            print("ERROR: oddsRatioDfPath file not found", oddsRatioDfPath.name)


    oddsRatioCtermTaxaDf = pd.concat(oddsRatioCtermTaxonList)
    return oddsRatioCtermTaxaDf


def gather_cterm_bias_group_results(path, groupList, suffix, onlyCtermPosition=True):
    
    oddsRatioDfList = []
    suffix = simplify_prefix(suffix)

    if not (path).is_dir():
        print("ERROR: folder does not exist: ", str(path))
        
    for i, group in enumerate(groupList):
            
        oddsRatioDfPath = path / simplify_prefix(suffix + '_subset_{}_oddsRatioAADf.csv'.format(group).strip("'"))
        if oddsRatioDfPath.is_file():

            groupNSeq = None
            groupNGenomes = None
            multispeciesStatisticsSummaryDfPath = path / simplify_prefix(suffix + '_subset_{}_multispeciesStatisticsSummary.csv'.format(group))
            
            if multispeciesStatisticsSummaryDfPath.is_file():
                summaryDf = pd.read_csv(multispeciesStatisticsSummaryDfPath)
                if not summaryDf.empty:
                    groupNSeq = summaryDf['nSeq'].sum()
                    groupNGenomes = len(summaryDf['genome_accession'].unique())

            # Import odds ratio dataframe for taxon
            oddsRatioDf = pd.read_csv(oddsRatioDfPath)

            if onlyCtermPosition:
                oddsRatioCtermDf = oddsRatioDf[(oddsRatioDf['position from terminus'] == -1) & (oddsRatioDf['terminus'] == 'C')]
            else:
                oddsRatioCtermDf = oddsRatioDf
            oddsRatioCtermDf2 = oddsRatioCtermDf.copy()
            oddsRatioCtermDf2['group'] = group
            oddsRatioCtermDf2['n_seq'] = groupNSeq
            oddsRatioCtermDf2['n_genomes'] = groupNGenomes

            oddsRatioDfList.append(oddsRatioCtermDf2)
            
        else:
            print("ERROR: oddsRatioDfPath file not found", oddsRatioDfPath.name)


    oddsRatioDfGroup = pd.concat(oddsRatioDfList)
    return oddsRatioDfGroup
