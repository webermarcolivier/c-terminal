# coding: utf-8
# Author: Marc Weber

"""
=========================================================================================
Protein sequence clustering with CD-HIT
=========================================================================================
"""

import itertools
import subprocess
import shlex
import os
from pathlib import Path
import re
import pandas as pd



# Keep old arguments with the generic output file name "clustering_output_identity"
# that is used in the cterminal script.
def clustering2(allSeqFastaFile, output_folder, identity_threshold, verbose=True):
    """
    Perform a clustering of protein sequences using CD-HIT.
    """
    if verbose:
        print("### Clustering of protein sequences (CD-HIT):")
        print("    identity threshold: ",identity_threshold)
        print("    output_folder: ",output_folder)

    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder,"identity_thresholds"), 'w') as file:
        file.write("identity_threshold,{:.15e}\n".format(identity_threshold))
        #file.write("cterm_identity_threshold,{:.15e}\n".format(cterm_identity_threshold))
    clusteringOutputFile = os.path.join(output_folder, "clustering_output_identity" + str(identity_threshold))
    cmd = 'cd-hit -i ' + allSeqFastaFile + ' -o ' + clusteringOutputFile + ' -c ' + str(identity_threshold) + ' -n 3'
    cmd = shlex.split(cmd)
    cmd_output = subprocess.check_output(cmd, cwd=output_folder)
    cmd_output = re.sub(r'\\n','\n',str(cmd_output))
    
    if verbose:
        print(cmd)
        print(cmd_output)
        print('clusteringOutputFile: ',clusteringOutputFile)
    return output_folder, clusteringOutputFile


def clustering(fastafile, identity_threshold, output_filename=None, output_folder=None, verbose=1):
    """
    Perform a clustering of protein sequences using CD-HIT.
    """
    if verbose >= 1:
        print("### Clustering of protein sequences (CD-HIT):")
        print("    identity threshold: ",identity_threshold)
    if verbose >= 2:
        print("    output_folder: ", output_folder)
        print("    output_filename: ", output_filename)

    fastaPath = Path(fastafile)
    if output_folder is None:
        outputPath = fastaPath.resolve().parent()
    else:
        outputPath = Path(output_folder)
    outputPath.mkdir(exist_ok=True)
    if verbose >= 2: print(outputPath)
    
    if output_filename is None:
        basename = re.search(r'(.+)(\.faa)|(\.fasta)|(\....)', fastaPath.name).group(1)
    else:
        basename = output_filename
    if verbose >= 2: print(basename)

    outputFilename = '{}.cd-hit_identity{:.2f}'.format(basename, identity_threshold)
    cmd = 'cd-hit -i ' + str(fastaPath) + ' -o ' + str(outputFilename) + ' -c ' + str(identity_threshold) + ' -n 3'
    cmd = shlex.split(cmd)
    cmd_output = subprocess.check_output(cmd, cwd=str(outputPath))
    cmd_output = re.sub(r'\\n','\n',str(cmd_output))
    
    if verbose >= 1:
        print(cmd)
        print(cmd_output)
    if verbose >= 2:
        print('outputFilename: ',outputFilename)
    return str(outputPath), outputFilename



def parse_cluster_file(filename):
    """
    Parse the output of the CD-HIT clustering and return a dictionnary of clusters.
    
    In order to parse the list of cluster and sequences, we have to parse the CD-HIT
    output file. Following solution is adapted from a small wrapper script
    ([source code on Github](https://github.com/Y-Lammers/CD-HIT-Filter/blob/master/CD-HIT-Filter.py),
    author: Youri Lammers).
    """

    # parse through the .clstr file and create a dictionary
    # with the sequences per cluster

    # open the cluster file and set the output dictionary
    cluster_file, cluster_dic = open(filename), {}

    # parse through the cluster file and store the cluster name + sequences in the dictionary
 
    # This is a generator comprehension which groups lines together based of wether the
    # line starts with a ">".
    cluster_groups = (x[1] for x in itertools.groupby(cluster_file, key=lambda line: line[0] == '>'))
    
    # Now we get alternate groups of cluster name and sequence list.
    for cluster in cluster_groups:
        # Note: next(cluster) retrieves the first line of the cluster i (>cluster name)
        name = next(cluster).strip()
        name = re.sub(' ', '_', name[1:])
        # Note: next(cluster_groups) retrieves the next cluster i+1 containing the sequences
        # the cluster is itself an iterator (every line)
        seqs = [seq.split('>')[1].split('...') for seq in next(cluster_groups)]
        # Write a boolean value True if sequence is the reference sequence from the cluster
        seqs = [[seq[0], (True if seq[1] == ' *\n' else False)] for seq in seqs]
        cluster_dic[name] = seqs

    # return the cluster dictionary
    return cluster_dic



def extract_seqIdList_from_cluster(cluster_name, cluster_dic, extract_id, allSeqDf, idName, match_method="contains"):
    """
    For all sequences in a cluster, find corresponding sequences in the dataframe.
    
    The extract_id function is used to extract a id that will be used to
    find the matching entry in the dataframe index. If the match_method option is set to "contains",
    the dataframe index will match the id if it contains the string. If the match_method option
    is set to "exact", the dataframe index will match the id if matches exactly.
    
    Return a list of dataframe indices.
    """
    seq_list = []
    for seq in cluster_dic[cluster_name]:
        # The output of the clustering algorithm CD-HIT only print the first 19 characters
        # (excluding the first > character) of the sequence id.
        # Therefore, we need to perform a search to map the printed information to the unique id
        # in the dataframe.
        
        # Extract the accession id which is **unique** for each sequence
        seqAccessionId = extract_id(seq[0])
        
        # Search for the accession id in the dataframe
        # Note: here we assume a multiindex dataframe, because the same function will
        # be used again later on the multiindex version of the allSeqDf dataframe.
        if match_method == "contains":
            pattern = seqAccessionId
        elif match_method == "exact":
            pattern = r'^' + seqAccessionId + r'$'
        else:
            print("Error extract_sequences_from_cluster: invalid match_method option.")
            
        dfSearch = allSeqDf.index.get_level_values(idName).str.contains(pattern)
        dfId = None
        
        # Test if we get more than one match
        nMatches = pd.Series(dfSearch).sum()
        if nMatches == 1:
            #dfId = allSeqDf[dfSearch].index.tolist()[0]
            # Valid for multiindex dataframe
            dfId = allSeqDf.index.get_level_values(idName)[dfSearch].tolist()[0]
        elif nMatches > 1:
            print("Error extract_sequences_from_cluster: sequenceid", seq[0],
                  " with accession id \"", seqAccessionId, "\" has several matches in dataframe.")
            dfId = None
        else:
            print("Error extract_sequences_from_cluster: sequenceid", seq[0],
                  " with accession id \"", seqAccessionId, "\" not found in dataframe.")
            # print("dfId =",allSeqDf.index.get_level_values(idName)[dfSearch].tolist()[0])
            # print("dfSearch.sum() =",dfSearch.sum())
            dfId = None
        
        # seq_list is just the list of sequence ids
        seq_list.append(dfId)
        
    return seq_list
