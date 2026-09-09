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
from stamp.tools.diagram.utils.classify import add_averaging_noise, feature_matrix, iterative_average, standardise_columns
from stamp.tools.diagram.utils.config import DiagramConfig
from stamp.tools.diagram.utils.identify import best_fit, fit_score, project_density, read_atom_coordinates, synthetic_truth
from stamp.tools.diagram.utils.geometry import non_max_suppression
from stamp.tools.diagram.utils.pick import reconcile, sample_membrane_surface, score_surface
from stamp.tools.diagram.utils.refine import fourier_ring_correlation, resolution_at_threshold
from stamp.tools.diagram.utils.scene import make_vesicles, render_scene

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

    # -- stage 1: scene = vesicles + raw tomogram + segmentation ------
    def build_scene(self):
        self.vesicles = make_vesicles(self.field_px, self.args.n_vesicles, self.rng)
        counts = self._class_counts()
        capture_px = 0.42 * self.box

        true_particles = []
        for class_id, count in enumerate(counts):
            for _ in range(count):
                v = self.vesicles[self.rng.integers(len(self.vesicles))]
                theta = self.rng.uniform(0, 2 * np.pi)
                radius = v['r'] + self.rng.normal(0.0, 1.5)
                true_particles.append({
                    'x': v['cx'] + radius * np.cos(theta),
                    'y': v['cy'] + radius * np.sin(theta),
                    'class': class_id,
                    'image': self.truths[class_id],
                })
        self.rng.shuffle(true_particles)
        # drop particles that fall too close together on the same membrane
        kept = []
        for p in true_particles:
            if all(np.hypot(p['x'] - q['x'], p['y'] - q['y']) > 0.5 * self.box for q in kept):
                kept.append(p)
        self.true_particles = kept
        self.true_xy = np.array([[p['x'], p['y']] for p in kept])
        self.true_class = np.array([p['class'] for p in kept])

        self.raw_tomogram, self.segmentation = render_scene(self.field_px, self.vesicles, self.true_particles, self.render_rng)

        # --- surface sampling + two pickers ---
        self.surface_points, self.surface_owner = sample_membrane_surface(self.vesicles, self.rng)
        score_a, self.surface_nearest = score_surface(self.surface_points, self.true_xy, capture_px, self.rng, noise=0.28)
        score_b, _ = score_surface(self.surface_points, self.true_xy, capture_px, self.rng, noise=0.31)
        self.surface_score = 0.5 * (score_a + score_b)
        self.pick_threshold = 0.5

        # each picker: threshold, then non-max suppression -> discrete picks
        nms_radius = 0.6 * self.box
        self.picker_a_idx = self.picker_b_idx = np.array([], dtype=int)
        above_a = np.where(score_a > self.pick_threshold)[0]
        if len(above_a):
            self.picker_a_idx = above_a[non_max_suppression(
                self.surface_points[above_a], score_a[above_a], nms_radius)]
        above_b = np.where(score_b > self.pick_threshold)[0]
        if len(above_b):
            self.picker_b_idx = above_b[non_max_suppression(self.surface_points[above_b], score_b[above_b], nms_radius)]
        self.picker_a_mask = np.zeros(len(self.surface_points), dtype=bool)
        self.picker_b_mask = np.zeros(len(self.surface_points), dtype=bool)
        self.picker_a_mask[self.picker_a_idx] = True
        self.picker_b_mask[self.picker_b_idx] = True
        self.picker_a = self.surface_points[self.picker_a_idx]
        self.picker_b = self.surface_points[self.picker_b_idx]

        self.consensus, self.rejected_picks = reconcile(self.picker_a, self.picker_b, distance_px=0.5 * self.box)
        # label each consensus pick by its nearest true particle
        if len(self.consensus):
            nearest = np.linalg.norm(self.consensus[:, None, :] - self.true_xy[None, :, :], axis=2).argmin(axis=1)
            self.consensus_class = self.true_class[nearest]
            self.consensus_nearest = nearest
        else:
            self.consensus_class = np.array([], dtype=int)
            self.consensus_nearest = np.array([], dtype=int)

        # rejected-surface decoys: low-scoring points picked by neither
        idle = ~(self.picker_a_mask | self.picker_b_mask) & \
            (self.surface_nearest > capture_px)
        idle_points = self.surface_points[idle]
        if len(idle_points):
            spread = non_max_suppression(idle_points, self.decoy_rng.random(len(idle_points)), 0.7 * self.box)
            idle_points = idle_points[spread]
            want = max(1, (len(self.consensus) or 2) // 2)
            take = self.decoy_rng.choice(len(idle_points), size=min(want, len(idle_points)), replace=False)
            self.decoy_points = idle_points[take]
        else:
            self.decoy_points = np.empty((0, 2))

        # half-set assignment: whole vesicle to one half
        vesicle_half = {i: int(self.decoy_rng.integers(2)) for i in range(len(self.vesicles))}
        if len(self.consensus):
            owners = np.array([int(np.argmin([np.hypot(cx - v['cx'], cy - v['cy']) for v in self.vesicles])) for cx, cy in self.consensus])
            self.consensus_half = np.array([vesicle_half[o] for o in owners])
        else:
            self.consensus_half = np.array([], dtype=int)

        # keep normalised positions for scatter-style plots
        self.consensus_norm = (self.consensus / self.field_px if len(self.consensus) else self.consensus)

    # -- stage 2: extract noisy subvolumes ---------------------------
    def extract_particles(self):
        self.particle_angles = []
        self.particles = []
        self.particle_class = self.consensus_class.copy()
        for class_id in self.consensus_class:
            angle = self.rng.uniform(0, 360)
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
        self.cluster_dominant = {c: int(np.bincount(self.particle_class[self.cluster_labels == c]).argmax()) for c in self.cluster_ids}
        # the cluster that best represents the target species
        target_counts = {c: int(np.sum((self.cluster_labels == c) & (self.particle_class == 0))) for c in self.cluster_ids}
        self.target_cluster = max(target_counts, key=target_counts.get)

    # _order_clusters: relabel clusters so their ids follow the dominant true class
    def _order_clusters(self, labels):
        labels = np.asarray(labels)
        info = []
        for c in sorted(set(labels)):
            members = labels == c
            dominant = int(np.bincount(self.particle_class[members]).argmax())
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
        self.extract_particles()
        self.classify()
        self.class_averages()
        self.identify()
        self.decoy_control()
        self.refine()
