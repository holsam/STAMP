'''
STAMP: mocked pipeline
'''

# Import external dependencies
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter, rotate
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Import internal STAMP objects
from stamp.schemas.particles import OrientationPrior
from stamp.tools.diagram.utils.carriers import carrier_of, make_carriers, place_on
from stamp.tools.diagram.utils.classify import add_averaging_noise, feature_matrix, iterative_average, standardise_columns
from stamp.tools.diagram.utils.config import DiagramConfig
from stamp.tools.diagram.utils.identify import best_fit, fit_score, project_density, read_atom_coordinates, synthetic_truth
from stamp.tools.diagram.utils.pick import beam_deviation, circular_mean, infer_orientations, reconcile, simulate_picks_file
from stamp.tools.diagram.utils.refine import fourier_ring_correlation, resolution_at_threshold
from stamp.tools.diagram.utils.scene import render_scene

# Walkthrough: mocked pipeline over mixture of N species; classes 0 and 1 are the identify search panel, any further classes are 'unknown' species that cluster but should not be identified
class Walkthrough:
    def __init__(self, args: DiagramConfig):
        self.args = args
        
        self.rng = np.random.default_rng(args.seed)
        self.render_rng = np.random.default_rng(args.seed + 1)
        self.decoy_rng = np.random.default_rng(args.seed + 2)
        self.box = args.box_size
        self.pixel_size = args.pixel_size
        self.field_px = args.field_px

        # --- structures: target, distractor, then extras / synthetic ---
        self.structures = []       # (name, truth_image)
        self.structure_paths = []  # PDB path per structure, None for synthetic
        for path in (args.target, args.distractor):
            self.structure_paths.append(Path(path))
            self.structures.append((Path(path).stem, project_density(read_atom_coordinates(path), self.box, self.pixel_size)))
        for path in (args.also or []):
            self.structure_paths.append(Path(path))
            self.structures.append((Path(path).stem, project_density(read_atom_coordinates(path), self.box, self.pixel_size)))
        for i in range(args.n_synthetic_unknowns):
            self.structure_paths.append(None)
            self.structures.append((f'unknown_{i + 1}', synthetic_truth(self.box, i)))

        self.names = [name for name, _ in self.structures]
        self.truths = [image for _, image in self.structures]
        self.n_classes = len(self.structures)
        self.target_name = self.names[0]
        self.distractor_name = self.names[1]
        self.target_truth = self.truths[0]
        self.distractor_truth = self.truths[1]
        # classes covered by the identify candidate panel
        self.panel_class_ids = {0, 1}

    # -- per-class particle counts -------------------------------------
    def _class_counts(self):
        counts = [self.args.n_target, self.args.n_distractor]
        counts += [self.args.n_extra] * (self.n_classes - 2)
        return counts

    # -- stage 1: scene = carriers (vesicles or filaments) + raw tomogram + segmentation ------
    def build_scene(self):
        self.prior = self.args.orientation_prior
        self.carriers = make_carriers(self.prior, self.field_px, self.args.n_vesicles, self.args.n_filaments, self.rng)
        counts = self._class_counts()

        true_particles = []
        for class_id, count in enumerate(counts):
            for _ in range(count):
                if self.carriers:
                    x, y, angle = place_on(self.carriers[self.rng.integers(len(self.carriers))], self.rng)
                else:   # none prior: particles sit anywhere in the volume, no orientation is known
                    (x, y), angle = self.rng.uniform(0.1 * self.field_px, 0.9 * self.field_px, size=2), np.nan
                true_particles.append({'x': x, 'y': y, 'class': class_id, 'angle': angle, 'image': self.truths[class_id]})
        self.rng.shuffle(true_particles)
        # drop particles that fall too close together on the same membrane
        kept = []
        for p in true_particles:
            if all(np.hypot(p['x'] - q['x'], p['y'] - q['y']) > 0.5 * self.box for q in kept):
                kept.append(p)
        self.true_particles = kept
        self.true_xy = np.array([[p['x'], p['y']] for p in kept])
        self.true_class = np.array([p['class'] for p in kept])

        self.raw_tomogram, self.segmentation = render_scene(self.field_px, self.carriers, self.true_particles, self.render_rng)

    # -- stage 2: picks files -> orientations -> consensus -> filters -> half-sets -----
    def pick(self):
        args, box = self.args, self.box
        capture_px = 0.42 * box
        # contaminants (ice, gold) that both pickers fire on, so some false picks survive the intersection
        n_confounders = max(3, len(self.true_xy) // 8)
        if self.carriers:
            confounders = np.array([place_on(self.carriers[self.rng.integers(len(self.carriers))], self.rng, jitter_px=9.0)[:2] for _ in range(n_confounders)])
        else:
            confounders = self.rng.uniform(0.1 * self.field_px, 0.9 * self.field_px, size=(n_confounders, 2))
        self.picks_a = simulate_picks_file(self.true_xy, confounders, self.field_px, self.rng, recall=0.9, jitter_px=3.0, n_false=12)
        self.picks_b = simulate_picks_file(self.true_xy, confounders, self.field_px, self.rng, recall=0.85, jitter_px=4.0, n_false=12)

        # orientation per pick, inferred per file as `stamp pick` does before reconciling
        self.angle_a, self.oriented_a = infer_orientations(self.prior, self.carriers, self.picks_a, self.rng, args.max_surface_distance_a / args.pixel_size)
        self.angle_b, self.oriented_b = infer_orientations(self.prior, self.carriers, self.picks_b, self.rng, args.max_surface_distance_a / args.pixel_size)
        angle_all = np.concatenate([self.angle_a, self.angle_b])
        oriented_all = np.concatenate([self.oriented_a, self.oriented_b])
        beam_all = beam_deviation(len(angle_all), self.rng) if self.prior is OrientationPrior.NORMAL else np.full(len(angle_all), np.nan)

        # consensus: intersection of the two files, position and orientation averaged over the group
        centres, self.rejected_picks, members = reconcile(self.picks_a, self.picks_b, distance_px=0.5 * box)
        self.reconciled = centres
        self.reconciled_oriented = np.array([oriented_all[m].all() for m in members], dtype=bool)
        self.reconciled_beam = np.array([beam_all[m].mean() for m in members])
        self.reconciled_angle = np.array([circular_mean(angle_all[m]) for m in members])
        # label each consensus pick by its nearest true particle, anything further is junk (-1)
        if len(centres):
            distance = np.linalg.norm(centres[:, None, :] - self.true_xy[None, :, :], axis=2)
            self.reconciled_class = np.where(distance.min(axis=1) < capture_px, self.true_class[distance.argmin(axis=1)], -1)
        else:
            self.reconciled_class = np.array([], dtype=int)
        # filters: unoriented picks cannot be classified, steeply tilted ones sit in the missing wedge
        self.beam_dropped = self.reconciled_oriented & (self.reconciled_beam > args.max_beam_angle_deviation)
        keep = self.reconciled_oriented & ~self.beam_dropped
        self.consensus = centres[keep]
        self.consensus_class = self.reconciled_class[keep]
        self.consensus_angle = self.reconciled_angle[keep]

        # half-sets: whole carrier to one half; without carriers each particle is assigned alone
        carrier_half = {i: int(self.decoy_rng.integers(2)) for i in range(len(self.carriers))}
        if len(self.consensus) and self.carriers:
            self.consensus_half = np.array([carrier_half[o] for o in carrier_of(self.carriers, self.consensus)])
        else:
            self.consensus_half = self.decoy_rng.integers(2, size=len(self.consensus))
        self.decoy_points = np.empty((0, 2))    # decoys arrive in 1.3
        self.consensus_norm = self.consensus / self.field_px if len(self.consensus) else self.consensus

    # -- stage 2: extract noisy subvolumes ---------------------------
    def extract_particles(self):
        self.particle_angles = []
        self.particles = []
        self.particle_class = self.consensus_class.copy()
        for class_id in self.consensus_class:
            angle = self.rng.uniform(0, 360)
            if class_id < 0:    # junk pick: nothing but noise at the noise level of a real particle
                noisy = self.rng.normal(0.0, self.truths[0].std() / self.args.per_particle_snr, size=self.truths[0].shape)
            else:
                clean = rotate(self.truths[class_id], angle, reshape=False, order=1, mode='constant')
                noisy = add_averaging_noise(clean, self.args.per_particle_snr, n_averaged=1, rng=self.rng)
            self.particle_angles.append(angle)
            self.particles.append(noisy)
        self.particles = np.array(self.particles)
        self.particle_angles = np.array(self.particle_angles)

    # -- stage 3: features + clustering ---------------------------
    def classify(self, n_clusters: int | None = None):
        default_k = self.n_classes + 1
        self.features = feature_matrix(self.particles)
        self.standardised = standardise_columns(self.features)
        n_components = min(self.args.cluster_components, self.standardised.shape[0] - 1, self.standardised.shape[1])
        self.pca = PCA(n_components=n_components, random_state=self.args.seed)
        self.pca_embedding = self.pca.fit_transform(self.standardised)
        self.embedding = self.pca_embedding[:, :2]

        k = n_clusters or default_k
        raw = KMeans(n_clusters=k, random_state=self.args.seed, n_init=20).fit_predict(self.pca_embedding)
        self.cluster_labels = self._order_clusters(raw)
        self.cluster_ids = sorted(set(self.cluster_labels))
        self.cluster_dominant = {c: self._dominant(self.cluster_labels == c) for c in self.cluster_ids}
        # the cluster that best represents the target species
        target_counts = {c: int(np.sum((self.cluster_labels == c) & (self.particle_class == 0))) for c in self.cluster_ids}
        self.target_cluster = max(target_counts, key=target_counts.get)

    # _dominant: most common true species among a cluster's members, ignoring junk picks
    def _dominant(self, members):
        real = self.particle_class[members][self.particle_class[members] >= 0]
        return int(np.bincount(real).argmax()) if len(real) else 0

    # _order_clusters: relabel clusters so their ids follow the dominant true class
    def _order_clusters(self, labels):
        labels = np.asarray(labels)
        info = []
        for c in sorted(set(labels)):
            members = labels == c
            dominant = self._dominant(members)
            info.append((c, dominant, -int(members.sum())))
        order = [c for c, _, _ in sorted(info, key=lambda t: (t[1], t[2]))]
        remap = {old: new for new, old in enumerate(order)}
        return np.array([remap[l] for l in labels])

    # -- stage 4: class averages -------------------------------
    def class_averages(self):
        self.averages = {}
        self.aligned = {}
        self.class_counts = {}
        for cluster_id in self.cluster_ids:
            members = self.particles[self.cluster_labels == cluster_id]
            aligned, average = iterative_average(members)
            self.aligned[cluster_id] = aligned
            self.averages[cluster_id] = average
            self.class_counts[cluster_id] = len(members)

    # -- stage 5: identification -----------------------------
    def identify(self):
        scrambled = gaussian_filter(self.rng.permutation(self.target_truth.ravel()).reshape(self.target_truth.shape), sigma=2.0)
        self.candidates = {
            self.target_name: self.target_truth,
            self.distractor_name: self.distractor_truth,
            'decoy_model': scrambled / (scrambled.max() or 1.0),
        }
        self.score_matrix = {}
        self.fit_images = {}
        self.calls = {}
        for cluster_id in self.cluster_ids:
            average = self.averages[cluster_id]
            self.score_matrix[cluster_id] = {}
            self.fit_images[cluster_id] = {}
            for name, sim in self.candidates.items():
                score, aligned = best_fit(average, sim)
                self.score_matrix[cluster_id][name] = score
                self.fit_images[cluster_id][name] = aligned
            ranked = sorted(self.score_matrix[cluster_id].items(), key=lambda kv: kv[1], reverse=True)
            (best_name, best_cc), (_, second_cc) = ranked[0], ranked[1]
            decoy_cc = self.score_matrix[cluster_id]['decoy_model']
            identified = (
                best_name != 'decoy_model'
                and best_cc >= self.args.identify_min_cc
                and best_cc - second_cc >= self.args.identify_min_gap
                and best_cc - decoy_cc >= self.args.decoy_margin
            )
            self.calls[cluster_id] = {
                'candidate': best_name if identified else None,
                'best_name': best_name,
                'cc': best_cc,
                'gap': best_cc - second_cc,
                'decoy_cc': decoy_cc,
                'identified': identified,
                'true_species': self.names[self.cluster_dominant[cluster_id]],
                'in_panel': self.cluster_dominant[cluster_id] in self.panel_class_ids,
            }
        self.identified = self.calls[self.target_cluster]['best_name']

    # -- stage 6: decoy control -----------------------------
    def decoy_control(self):
        target_average = self.averages[self.target_cluster]
        real_scores, decoy_scores = [], []
        for _ in range(self.args.n_decoy_trials):
            jittered = add_averaging_noise(self.target_truth, self.args.per_particle_snr, n_averaged=int(self.rng.integers(2, 6)), rng=self.rng)
            real_scores.append(fit_score(target_average, jittered))
            shuffled = gaussian_filter(self.rng.permutation(self.target_truth.ravel()).reshape(self.target_truth.shape), sigma=2.0)
            decoy_scores.append(fit_score(target_average, shuffled))
        self.real_scores = np.array(real_scores)
        self.decoy_scores = np.array(decoy_scores)
        self.decoy_passed = self.real_scores.mean() > self.decoy_scores.mean() + 0.05

    # -- stage 7: refinement / FRC ------------------------
    def refine(self):
        members = self.aligned[self.target_cluster]
        half = len(members) // 2
        shuffled = self.rng.permutation(members)
        self.half_a = shuffled[:half].mean(axis=0)
        self.half_b = shuffled[half:].mean(axis=0)
        self.recovered = shuffled.mean(axis=0)
        self.frc_freq, self.frc_corr = fourier_ring_correlation(self.half_a, self.half_b)
        self.resolution = resolution_at_threshold(self.frc_freq, self.frc_corr, self.pixel_size)

    # run: execute every stage in order
    def run(self):
        self.build_scene()
        self.pick()
        self.extract_particles()
        self.classify()
        self.class_averages()
        self.identify()
        self.decoy_control()
        self.refine()
