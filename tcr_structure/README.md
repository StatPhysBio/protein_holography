# TCRStructure

## TCR structures in the PDB

`./pdb_tcr_info.tsv` has information on TCR-peptide:MHC structures in the PDB as of 2019-09-09.
There are probably bugs in the parsing and missed/extra structures but it should give a rough
idea of what's available.

The 'parsed' structures, in which extra chains have mostly been deleted,
are in the `./pdbs/` directory. PB will check in his scripts either here or for the ones that
depend on `tcr-dist` into that repository.
