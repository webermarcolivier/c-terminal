# coding: utf-8
# Author: Marc Weber

"""
=========================================================================================
Extract accession ids
=========================================================================================
"""

import re



def extract_refseq_accession_id(seqIdString):
    """
    Extract the RefSeq accession number in the description string.
    
    See: https://www.ncbi.nlm.nih.gov/books/NBK21091/table/ch18.T.refseq_accession_numbers_and_mole/?report=objectonly
    See: http://www.ncbi.nlm.nih.gov/books/NBK50679/#RefSeqFAQ.what_are_the_distinguishing_fe
    """
    seqId = ''
    regex1 = re.compile(r'(\s|\||^)([A-Z]{2}_[0-9.]+)(\s|\||$)')
    seqIdSearch = re.search(regex1, seqIdString)
    if seqIdSearch:
        seqId = seqIdSearch.group(2)
    return seqId


def extract_fasta_id(seqIdString):
    """
    Extract the id in the fasta description string, taking only the characters till the '|'.
    """
    seqId = None
    regex1 = re.compile(r'^\s*?([^|]+?)(\s|\||$)')
    seqIdSearch = re.search(regex1, seqIdString)
    if seqIdSearch:
        seqId = seqIdSearch.group(1)
    return seqId


def extract_seq_id(seqIdString):
    """
    Extract the sequence id together with other tags in the description string.
    
    Example: print(string,"-->",extract_seq_id('NP_109863.1|MPN-175|DNA polymerase'))
    """
    seqId = ''
    # Case where we have several ids (no spaces), one | character and then a description
    regex1 = re.compile(r'^([\w.|-]+)\|[[\]\s\w.-]+\s+[[\]\s\w.-]*$')
    seqIdSearch = re.search(regex1,seqIdString)
    if seqIdSearch:
        seqId = seqIdSearch.group(1)
        #print("case1")
    else:
        # Case where we only have the id (no spaces), ending with nothing or |
        regex2 = re.compile(r'^([\w.|-]+[^|]+)(\||$)')
        seqIdSearch = re.search(regex2,seqIdString)
        seqId = seqIdSearch.group(1)
        #print("case2")

    return seqId


def extract_mpn_id(seqIdString):
    """
    For special case of Mycoplasma pneumoniae sequences: grep the correct sequence
    id and return id (mpn tag), sequence and Bio_object.
    
    Example: print(string,"-->",extract_mpn_id('NP_109863.1|MPN-175|DNA's))
    """
    seqId = ''
    regex1 = re.compile(r'(MPN[\w.-]+)(\||$)')
    seqIdSearch = re.search(regex1,seqIdString)
    if seqIdSearch:
        seqId = seqIdSearch.group(1)
    return seqId


def extract_id_print_examples():

    _idList = \
    [
        'NP_109863.1|MPN-175|DNA',
        'mpn|NP_109689.1|MPN001|DNA polymerase III subunit beta [Mycoplasma pneumoniae M129]',
        'NP_109689.1|MPN001|DNA polymerase III subunit beta [Mycoplasma pneumoniae M129]',
        'NP_109863.1|MPN-175|',
        'NP_109863.1|MPN-175',
        'mpn|NP_109863.1',
        'mpn|NP_109863.1|',
        'NP_109863.1'
    ]

    print("Extract the accession id together with the mpn id:\n")
    for string in _idList:
        print(string,"-->",extract_seq_id(string))
    print("\n")

    print("Extract the RefSeq accession id\n")
    for string in _idList:
        print(string,"-->",extract_refseq_accession_id(string))
    print("\n")

    print("Extract the mpn id\n")
    for string in _idList:
        print(string,"-->",extract_mpn_id(string))
    print("\n")
