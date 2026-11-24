'''
STAMP: mocked pipeline
'''

# Import external dependencies
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter, rotate
from types import SimpleNamespace

# Import internal STAMP objects
from stamp.identify.decoy_check import evaluate_decoy_control
from stamp.identify.fit import rank_candidates
from stamp.classify.cluster import ClusteringConfig, reduce_and_cluster
from stamp.schemas.particles import OrientationPrior
from stamp.tools.diagram.utils.carriers import carrier_of, make_carriers, place_on
from stamp.tools.diagram.utils.align import align_stack, apply_pose, bin_stack, weighted_average
from stamp.tools.diagram.utils.classify import add_averaging_noise, feature_matrix, standardise_columns
from stamp.tools.diagram.utils.config import DiagramConfig
from stamp.tools.diagram.utils.decoy import random_volume_decoys, shifted_decoys, synthetic_noise_tile
from stamp.tools.diagram.utils.identify import project_density, read_atom_coordinates, synthetic_truth
from stamp.tools.diagram.utils.polish import PolishConfig, Stacks, match_halves, run_polish
from stamp.tools.diagram.utils.search import apply_wedge, search_space, wedge_keep
from stamp.tools.diagram.utils.pick import beam_deviation, circular_mean, infer_orientations, reconcile, simulate_picks_file
from stamp.tools.diagram.utils.register import register_score
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
            if all(np.hypot(p['x'] - q['x'], p['y'] - q['y']) > 0.3 * self.box for q in kept):
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
        n_confounders = max(3, len(self.true_xy) // 6)
        if self.carriers:
            confounders = np.array([place_on(self.carriers[self.rng.integers(len(self.carriers))], self.rng, jitter_px=14.0)[:2] for _ in range(n_confounders)])
        else:
            confounders = self.rng.uniform(0.1 * self.field_px, 0.9 * self.field_px, size=(n_confounders, 2))
        self.picks_a = simulate_picks_file(self.true_xy, confounders, self.field_px, self.rng, recall=0.95, jitter_px=3.0, n_false=12)
        self.picks_b = simulate_picks_file(self.true_xy, confounders, self.field_px, self.rng, recall=0.9, jitter_px=4.0, n_false=12)

        # orientation per pick, inferred per file as `stamp pick` does before reconciling
        self.angle_a, self.oriented_a = infer_orientations(self.prior, self.carriers, self.picks_a, self.rng, args.max_surface_distance_a / args.pixel_size)
        self.angle_b, self.oriented_b = infer_orientations(self.prior, self.carriers, self.picks_b, self.rng, args.max_surface_distance_a / args.pixel_size)
        angle_all = np.concatenate([self.angle_a, self.angle_b])
        oriented_all = np.concatenate([self.oriented_a, self.oriented_b])
        beam_all = beam_deviation(len(angle_all), self.rng) if self.prior is OrientationPrior.NORMAL else np.full(len(angle_all), np.nan)

        # consensus: intersection of the two files, position and orientation averaged over the group
        centres, self.rejected_picks, members = reconcile(self.picks_a, self.picks_b, distance_px=0.2 * box)
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

        # half-sets: every carrier goes whole to one half, largest first to the smaller half so halves stay balanced; without carriers each particle is its own group
        groups = carrier_of(self.carriers, self.consensus) if self.carriers and len(self.consensus) else np.arange(len(self.consensus))
        sizes = np.bincount(groups) if len(groups) else np.array([], dtype=int)
        load, half_of = [0, 0], {}
        for group in np.argsort(-sizes):
            half = int(np.argmin(load))
            half_of[int(group)], load[half] = half, load[half] + int(sizes[group])
        self.consensus_half = np.array([half_of[int(g)] for g in groups], dtype=int)

        # decoys: the method `auto` resolves per prior (shifted for normal, random-volume for axis and none)
        if self.prior is OrientationPrior.NORMAL:
            self.decoy_method = 'shifted'
            self.decoy_points = shifted_decoys(self.consensus, self.carriers, self.true_xy, self.field_px, box, self.decoy_rng)
        else:
            self.decoy_method = 'random-volume'
            self.decoy_points = random_volume_decoys(len(self.consensus), self.true_xy, self.field_px, box, self.decoy_rng)
        self.noise_tile, self.noise_points = synthetic_noise_tile(self.decoy_rng, max(4, len(self.consensus) // 2))
        self.consensus_norm = self.consensus / self.field_px if len(self.consensus) else self.consensus

    # -- stage 3: extract subvolumes, each unknown pose drawn as prior allows, each with its missing wedge -----
    def extract_particles(self):
        args, box, factor = self.args, self.box, self.args.bin_factor
        self.particle_class = self.consensus_class.copy()
        self.particle_half = self.consensus_half.copy()
        images, masks, masks_binned, truth_angle, phis = [], [], [], [], []
        for class_id, prior_angle in zip(self.consensus_class, self.consensus_angle):
            angle, shift = self._draw_pose()
            if class_id < 0:    # junk pick: nothing but noise at the noise level of a real particle
                noisy = self.rng.normal(0.0, self.truths[0].std() / args.per_particle_snr, size=self.truths[0].shape)
            else:
                clean = np.roll(rotate(self.truths[class_id], angle, reshape=False, order=1, mode='constant'), shift, axis=(0, 1))
                noisy = add_averaging_noise(clean, args.per_particle_snr, n_averaged=1, rng=self.rng)
            phi = 0.0 if self.prior is OrientationPrior.NONE or np.isnan(prior_angle) else float(np.degrees(prior_angle))   # wedge follows prior's frame
            images.append(apply_wedge(noisy, wedge_keep(box, args.tilt_max_degrees, phi)))
            masks.append(wedge_keep(box, args.tilt_max_degrees, phi))
            masks_binned.append(wedge_keep(box // factor, args.tilt_max_degrees, phi))
            truth_angle.append(angle)
            phis.append(phi)
        self.particles = np.array(images)
        self.particle_masks = np.array(masks)
        self.particle_masks_binned = np.array(masks_binned)
        self.particle_truth_angle = np.array(truth_angle)
        self.particle_phi = np.array(phis)

    # _draw_pose: pose prior leaves unknown, normal an azimuth, axis a polarity flip and an axial shift, none a rotation and a small shift
    def _draw_pose(self):
        if self.prior is OrientationPrior.AXIS:
            return float(self.rng.choice([0.0, 180.0])), (0, int(self.rng.integers(-5, 6)))
        if self.prior is OrientationPrior.NONE:
            return float(self.rng.uniform(0, 360)), tuple(int(v) for v in self.rng.integers(-2, 3, size=2))
        return float(self.rng.uniform(0, 360)), (0, 0)

    # -- stage 4: features per prior, whitened PCA and Leiden seeds, then polish, each half on its own -----
    def classify(self):
        args = self.args
        self.features = feature_matrix(self.particles, self.prior)
        self.standardised = standardise_columns(self.features)
        self.space = search_space(self.prior, max_shift=args.shift_range_px)
        self.stacks = Stacks(self.particles, self.particle_masks, bin_stack(self.particles, args.bin_factor), self.particle_masks_binned, args.bin_factor)
        polish_config = PolishConfig(args.n_polish_rounds, args.min_cluster_size, args.merge_fraction, args.reference_quantile, args.reject_quantile, args.seed)
        self.halves = []
        for half in (0, 1):
            index = np.where(self.particle_half == half)[0]
            clustering = ClusteringConfig(method=args.cluster_method, n_components=args.cluster_components, min_cluster_size=args.min_cluster_size, n_clusters=self.n_classes + 1, random_state=args.seed)
            reduced = reduce_and_cluster(standardise_columns(self.features[index]), clustering)
            polish = run_polish(self.stacks.take(index), reduced.embedding, reduced.labels, self.space, polish_config)
            self.halves.append(SimpleNamespace(index=index, embedding=reduced.embedding, explained=reduced.explained_variance_ratio, seed_labels=reduced.labels, polish=polish))
        if not all(h.polish.references for h in self.halves):
            raise ValueError('polish kept no class in one half, raise --n-target or lower --reference-quantile')
        # half B joins half A: one-to-one match by reference agreement, B poses expressed in the frame of A
        self.pairs = match_halves(self.halves[0].polish.references, self.halves[1].polish.references, self.space, args.bin_factor)
        labels = np.full(len(self.particles), -1)
        labels[self.halves[0].index] = self.halves[0].polish.labels
        b_to_a = {b: (a, angle, shift) for b, a, angle, shift, _cc in self.pairs}
        for b_index, label in zip(self.halves[1].index, self.halves[1].polish.labels):
            labels[b_index] = b_to_a[label][0] if label in b_to_a else -1
        self.cluster_labels = self._order_clusters(labels)
        self.cluster_map = {int(old): int(new) for old, new in zip(labels[labels >= 0], self.cluster_labels[labels >= 0])}
        self._summarise_clusters()

    # _summarise_clusters: cluster ids, dominant species and cluster that best represents target
    def _summarise_clusters(self):
        self.cluster_ids = sorted(set(self.cluster_labels) - {-1})
        self.cluster_dominant = {c: self._dominant(self.cluster_labels == c) for c in self.cluster_ids}
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
        for c in sorted(set(labels) - {-1}):
            members = labels == c
            dominant = self._dominant(members)
            info.append((c, dominant, -int(members.sum())))
        order = [c for c, _, _ in sorted(info, key=lambda t: (t[1], t[2]))]
        remap = {old: new for new, old in enumerate(order)}
        return np.array([remap.get(l, -1) for l in labels])

    # -- stage 5: class averages, every particle in frame of half A, wedge-weighted -------------
    def class_averages(self):
        self.averages, self.aligned, self.class_counts, self.half_averages = {}, {}, {}, {}
        self.polish_angle = np.zeros(len(self.particles))
        a, b = self.halves
        a_to_b = {ai: (bi, angle, shift) for bi, ai, angle, shift, _cc in self.pairs}
        for old, new in self.cluster_map.items():
            from_a = self._aligned(a, np.where(a.polish.labels == old)[0])
            if old in a_to_b:
                b_label, relative_angle, relative_shift = a_to_b[old]
                from_b = self._aligned(b, np.where(b.polish.labels == b_label)[0], relative_angle, relative_shift)
            else:
                from_b = self._aligned(b, np.array([], dtype=int))
            self.aligned[new] = np.concatenate([from_a[0], from_b[0]])
            self.averages[new] = weighted_average(self.aligned[new], np.concatenate([from_a[1], from_b[1]]))
            self.half_averages[new] = tuple(weighted_average(*frames) if len(frames[0]) else np.zeros_like(self.averages[new]) for frames in (from_a, from_b))
            self.class_counts[new] = len(self.aligned[new])

    # _aligned: selected members of a half in their reference frame, then moved into frame of half A by relative pose when given; records the pose angle per particle
    def _aligned(self, half, sel, relative_angle: float = 0.0, relative_shift=(0, 0)):
        polish, index = half.polish, half.index[sel]
        images, masks = align_stack(self.particles[index], self.particle_masks[index], polish.angle[sel], polish.shift[sel])
        if len(sel) and (relative_angle or any(relative_shift)):
            images = np.array([apply_pose(image, relative_angle, tuple(relative_shift)) for image in images])
            masks = np.fft.ifftshift(np.array([rotate(np.fft.fftshift(m), -relative_angle, reshape=False, order=0) for m in masks]), axes=(-2, -1))
        self.polish_angle[index] = (polish.angle[sel] + relative_angle) % 360
        return images, masks

    # -- stage 6: identification by registration of each candidate to each class average ------------
    def identify(self):
        self.candidates = {self.target_name: self.target_truth, self.distractor_name: self.distractor_truth}
        self.registrations = {}
        self.score_matrix = {}
        self.fit_images = {}
        self.calls = {}
        for cluster_id in self.cluster_ids:
            self.registrations[cluster_id] = {name: register_score(self.averages[cluster_id], simulated, self.prior) for name, simulated in self.candidates.items()}
            self.score_matrix[cluster_id] = {name: result.score for name, result in self.registrations[cluster_id].items()}
            self.fit_images[cluster_id] = {name: result.fitted for name, result in self.registrations[cluster_id].items()}
            ranked = rank_candidates(f'c{cluster_id:02d}', self.score_matrix[cluster_id], method='stamp-register')
            self.calls[cluster_id] = {
                'candidate': ranked.candidate_protein,
                'cc': ranked.fit_score,
                'gap': ranked.score_gap_to_runner_up,
                'true_species': self.names[self.cluster_dominant[cluster_id]],
                'in_panel': self.cluster_dominant[cluster_id] in self.panel_class_ids,
            }
        self.identified = self.calls[self.target_cluster]['candidate']

    # -- stage 7: decoy control, noise-only decoy particles through the same classify, then the same registration -----
    def decoy_control(self):
        args, box, factor = self.args, self.box, self.args.bin_factor
        n = len(self.consensus)
        self.decoy_particles, self.decoy_masks, self.decoy_masks_binned = [], [], []
        for _ in range(n):
            phi = 0.0 if self.prior is OrientationPrior.NONE else float(self.decoy_rng.uniform(0, 360))
            keep = wedge_keep(box, args.tilt_max_degrees, phi)
            self.decoy_particles.append(apply_wedge(self.decoy_rng.normal(0.0, self.truths[0].std() / args.per_particle_snr, size=(box, box)), keep))
            self.decoy_masks.append(keep)
            self.decoy_masks_binned.append(wedge_keep(box // factor, args.tilt_max_degrees, phi))
        particles = np.array(self.decoy_particles)
        stacks = Stacks(particles, np.array(self.decoy_masks), bin_stack(particles, factor), np.array(self.decoy_masks_binned), factor)
        features = standardise_columns(feature_matrix(particles, self.prior))
        polish_config = PolishConfig(args.n_polish_rounds, args.min_cluster_size, args.merge_fraction, args.reference_quantile, args.reject_quantile, args.seed)
        halves = self.decoy_rng.integers(2, size=n)
        self.decoy_polish, self.decoy_references = [], []
        for half in (0, 1):
            index = np.where(halves == half)[0]
            reduced = reduce_and_cluster(features[index], ClusteringConfig(method=args.cluster_method, n_components=args.cluster_components, min_cluster_size=args.min_cluster_size, n_clusters=self.n_classes + 1, random_state=args.seed))
            try:
                result = run_polish(stacks.take(index), reduced.embedding, reduced.labels, self.space, polish_config)
            except ValueError:  # no seed cluster big enough: nothing to polish, no decoy class
                continue
            self.decoy_polish.append(result)
            self.decoy_references += result.references
        self.real_scores = [max(scores.values()) for scores in self.score_matrix.values()]
        self.decoy_scores = [max(register_score(reference, simulated, self.prior).score for simulated in self.candidates.values()) for reference in self.decoy_references]
        self.decoy_result = evaluate_decoy_control(self.real_scores, self.decoy_scores)
        self.decoy_passed = self.decoy_result.passed

    # -- stage 8: refinement / FRC ------------------------
    def refine(self):
        self.half_a, self.half_b = self.half_averages[self.target_cluster]
        self.recovered = self.averages[self.target_cluster]
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
