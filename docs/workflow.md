# STAMP - workflow

This document provides an outline of the theoretical basis underpinning STAMP. For details on the practical implementation of this workflow, see the [Architecture documentation](./architecture.md).

<br>

## Purpose
STAMP was designed as a label-free tool for resolving membrane protein structures from cryo-electron tomograms for comparison with proteomics data collected by mass spectrometry. The core principle behind STAMP is subtomogram averaging (STA), an overview of which is available [here](https://cryoem101.org/cryoet-chapter-6/#part4), with STAMP's pipeline designed to reduce the risk of template and confirmation bias.

<br>

## Pipeline Stages
STAMP is designed to process segmented tomograms (particularly the output of [MemBrain-seg](https://github.com/teamtomo/membrain-seg)) using the four-stage pipeline described below:

Stage | Purpose
-- | --
Pick | Find positions of candidate particles
Classify | Sort mixed particles into clusters by similarity
Identify | Screen class averages against proteomics data
Refine | Improve class averages and measure quality

### Pick
Picking takes raw/segmented tomograms and identify candidate particles from these. STAMP was written as a modular system to support multiple pickers, with functions for reconciling the outputs of multiple pickers into a single set of consensus particles. At present, however, STAMP only implements a native picker - for information on how this picker works, see [below](#stamps-native-picker).

### Classify
Classification sorts the mixed outputs of picking into groups of similar particles. STAMP implements classification in five steps:

\# | Step | Description
-- | -- | --
**1** | Extract subvolumes | Extract subvolume around each pick and align the membrane normal to the box's Z axis for comparable orientations
**2** | Produce rotational average | Average density around the Z axis to give a rotation-invariant profile
**3** | Fourier transform the rotation angle | Rotational averaging removes symmetry information, so use Fourier transformations to preserve this symmetry information while keeping the profile rotation-invariant
**4** | PCA and cluster | Reduce profile dimensions using using PCA then apply clustering algorithm to produce groups of related profiles
**5** | Average groups | Produce a class average for each group, allowing judgment of size and shape

The produced class averaged will still be smeared about the rotation (Z) axis as spin wasn't deteremined, but should be suitable for the downstream stages.

### Identify
Identification compares each class average to a panel of experimentally-derived candidate protein structures (not an entire proteome). Each protein is converted to a simulated density at the same resolution as the class averages, and the fit between each is scored. The most useful datapoint provided by this stage is the gap between the best and second-best scores. If both scores are near-identical, the proteins are unlikely to be truly matching the class average; if one score is >0.8 and the second-highest is <0.3, this suggests the best fitting protein may be a match. Note that matches are only evidence of consistency, not identity.

### Refine
Refinement iterates over a the class average to improve each particle's estimated orientation, then re-averages with the better estimates (repeating this process *n* times). STAMP outsources refinement to existing tools (e.g. RELION, M) rather than implementing it natively. The refinement stage is where the remaining spin around the normal gets determined.

<br>

## How STAMP reduces bias
### Half-set split
STAMP implements the gold-standard split used in STA, dividing particles randomly into two halves which are processed independently. Once the pipeline has run, the two independent averages are compared to determine if they converge on the same structure and to what extent they converge. This convergence is measured using Fourier shell correlation (FSC), with the point at which agreement falls below the threshold (0.143) representing the resolution. In STAMP, each particle is assigned to half-set A or B at the end of picking and each subsequent stage processes each half-set independently.

### Decoys
As a negative control, STAMP runs a set of decoy picks (where no particles exist) through the exact same pipeline to check that the clustering and averaging steps aren't returning artefactual structures from the underlying algorithms. STAMP's main decoy method uses the rejected positions from the native picker, which provides a set of 'picks' from the same membranes/tomograms as the real picks. If the decoy set produces class averages as convincing as the real data, it's likely that STAMP is responding to membrane shape/ice thickness/missing-wedge artefacts rather than proteins themselves. If the decoy set produces weak/structureless averages (and the real set produce convincing averages), this provides evidence that STAMP is responding to the actual protein structures.

### Predicted-structure separation
STAMP only introduces the candidate proteins (from proteomics experiments) after class averages have been produced, and does not use them references for alignment. This is specifically to avoid [Einstein from noise](https://doi.org/10.1073/pnas.1314449110) which could otherwise appear to be confirmation that the candidate proteins are present in the tomograms. STAMP uses the inverse approach (build averages then compare these to candidates) to control for this bias.

<br>

## STAMP's native picker
STAMP includes a self-contained native picking tool based on geometric principles, which depends only on a small number of Python libraries (`numpy`, `scipy`, `scikit-image`, `mrcfile`). It requires no addiitonal installs, and was designed to run on a laptop (MacBook Air M1).

The native picker doesn't search the entire raw tomogram for particles, but instead uses the membrane segmentation as a strong prior to restrict the search space to a small window on either side of the membrane. This has computational benefits by reducing the area searched, but also provides an orientation prior for downstream classification: membrane proteins must point outward along the local surface normal, which fixes two of the three rotational degrees of freedom before any alignment begins. Only the spin about the normal (sometimes called the in-plane angle) remains unknown, but this can be resolved throughout classification and refinement.

The specific workflow the native picker uses:

\# | Step | Description
-- | -- | --
**1** | Extract surface from segmentation | Uses marching cubes on the binary segmentation to produce a triangle mesh with per-vertex normals
**2** | Downsample vertexes | Downsample voxel-grid to a target spacing so sampling density doesn't depend on mesh resolution
**3** | Apply bidirectional ray sampling | From each surface point, move along the membrane normal (in both directions) across one or more radial offset windows, sampling the raw tomogram by trilinear interpolation.
**4** | Score offset windows | Score each offset window[] and apply normalisation (median/MAD) using either tomogram-global, local-neighbourhood or per-vesicle background.
**5** | Apply multi-window correction and threshold | Raise the threshold for keeping the best-scoring window by an order-statistic correction to avoid upward bias, and only keep points above this
**6** | Combine local neighbours | Use non-maximum supression (greedy, highest score first) to suppress neighbours via a k-d tree to avoid detecting the same particle multiple times
**7** | Produce oriented unit quaternion | Apply a unit quaternion rotating z to the membrane normal to each pick, to give downstream classification an orientation prior

The native picker has two methods for scoring offset windows:
- **Mean:** calculated mean density within the window
- **Profile:** Pearson correlation fo the radial profile against a Gaussian bump centred on the windows midpoint (may help to distinguish proteins from connected vesicles etc)
