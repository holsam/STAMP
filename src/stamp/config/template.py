'''
STAMP: template stamp_run.toml file
'''

# TEMPLATE: annotated stamp_run.toml
TEMPLATE = '''\
# STAMP: run configuration
# Paths are resolved relative to this file.
#   stamp run --config stamp_run.toml

[run]
segmentation_dir = "data/seg"               # --seg-dir  (required for orientation_prior = "normal")
raw_tomogram_dir = "data/tomo"              # --raw-dir
output_dir = "."                            # output directory for each stage
voxel_size_angstrom = ""                    # --voxel-size-a  (blank = auto-detect from tomogram headers)
backend = "local"                           # --backend  (local|mock|cluster; per-stage backends below override)
stop_after = ""                             # optional; blank|pick|classify|identify|refine
orientation_prior = "normal"                # normal (membrane proteins)|axis (filaments)|none (soluble particles)
tilt_max_degrees = ""                       # --tilt-max  (maximum tilt of the tilt series; blank = no missing wedge modelled)
tilt_axis_degrees = 90.0                    # --tilt-axis  (tilt axis angle from the tomogram x axis, 90 = tilt about y)

[decoy]
enabled = true                              # --no-decoy
method = "shifted"                          # --method  (shifted|synthetic-noise|random-volume; random-volume for the axis and none priors)
n_decoys_per_tomogram = 50                  # synthetic-noise only
min_distance_from_real_angstrom = 100.0
min_distance_from_picks_angstrom = 60.0
min_shift_angstrom = 200.0
max_shift_angstrom = 600.0
min_border_angstrom = 200.0                 # random-volume only
n_synthetic_tomograms = 3
n_workers = 1
synthetic_shape_voxels = ""                 # blank = match first real tomogram, or e.g. [128, 128, 128]

[plots]
enabled = true
format = "tiff"                             # png|jpg|tiff|svg
pick_style = "segmented"                    # segmented|none
pick_zstack_movie = true
pick_plot_3d = false

[stage.pick]
picks_files = ["data/picks.csv"]            # positional picks files (.csv|.star|.json|.jsonl), one picker per file
# picker_names = ["picker-a"]               # --name  (one per file, unique; default = file stems)
consensus_rule = "intersection"             # --consensus-rule  (union|intersection)
distance_threshold = 15.0                   # --distance-threshold
half_set_seed = 0                           # --half-set-seed
max_surface_distance_angstrom = 300.0       # --max-surface-distance-a
n_workers = 1
keep_raw = false
backend = ""                                # inert for pick; blank = defaults to [run].backend

[stage.classify]
method = "leiden"                           # --method  (leiden|hdbscan|kmeans)
min_cluster_size = 20                       # smallest cluster, and the smallest seed polish refines
n_clusters = 5
n_components = 20
box_angstrom = 300.0                        # --box-length-a
n_radial_bins = 12                          # --n-bins
strict_halfset_independence = true
inplane_angular_step_degrees = 10.0
bin_factor = 2                              # --bin-factor  (working resolution)
n_rotations = 3000                          # --n-rots  (none prior)
shift_range_voxels = 3                      # --shift-range  (working voxels)
polish_rounds = 2                           # --polish-rounds
null_sigmas = 3.0                           # --null-sigmas
reject_quantile = 0.95                      # --reject-quantile
merge_fraction = 1.0                        # --merge-fraction
random_state = 0                            # --seed
n_workers = 1

[stage.identify]
candidates = "data/candidates.yaml"         # --candidates  (required unless stop_after is pick|classify)
resolution = 25.0                           # --resolution  (required unless stop_after is pick|classify)
fitter = "native"                           # --fitter
fetch_missing = false                       # --fetch-missing
n_workers = 1
backend = ""                                # blank = defaults to [run].backend

[stage.refine]
tool = "relion"                             # --tool  (relion|m)
class_id = "all"                            # --class-id  (or a single class N)
iterations = 5                              # --iterations
mask = ""                                   # --mask  (blank = soft sphere)
backend = ""                                # blank = defaults to [run].backend
'''
