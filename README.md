# STAMP

A workflow for Sub-Tomogram Averaging Membrane Proteins.

## Overview
Given a directory of segmented MRC volumes (i.e. as produced by MemBrain-seg), `STAMP` runs the following workflow:
- Consensus picking
- Unsupervised classification
- Identification against predicted structures
- Refinement

Half-sets and a decoy control are used by default.