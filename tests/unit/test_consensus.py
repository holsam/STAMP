'''
STAMP: unit tests for pick consensus reconciliation
'''

# Import internal STAMP objects
from stamp.consensus.reconcile import build_particle_set, reconcile_picks
from stamp.schemas.picks import RawPick

# _pick: return a RawPick instance for use in tests
def _pick(tomogram_id: str, position, picker: str, confidence=0.8, orientation=None) -> RawPick:
    return RawPick(
        tomogram_id=tomogram_id,
        position=position,
        orientation=orientation,
        confidence=confidence,
        source_picker=picker,
    )

# TestConsensusRules: class containing tests for consensus rules (intersection/union)
class TestConsensusRules:
    def test_intersection_merges_close_picks_from_different_pickers(self) -> None:
        picks_by_picker = {
            'stamp-native': [_pick('tomo000', (100.0, 100.0, 100.0), 'stamp-native')],
            'membrain-pick': [_pick('tomo000', (103.0, 101.0, 99.0), 'membrain-pick')],
        }
        reconciled = reconcile_picks(picks_by_picker, 'intersection', distance_threshold=15.0, tomogram_id='tomo000')
        assert len(reconciled) == 1
        assert 'stamp-native' in reconciled[0].source_picker
        assert 'membrain-pick' in reconciled[0].source_picker

    def test_intersection_drops_picks_found_by_only_one_picker(self) -> None:
        picks_by_picker = {
            'stamp-native': [_pick('tomo000', (100.0, 100.0, 100.0), 'stamp-native')],
            'membrain-pick': [_pick('tomo000', (500.0, 500.0, 500.0), 'membrain-pick')],
        }
        reconciled = reconcile_picks(picks_by_picker, 'intersection', distance_threshold=15.0, tomogram_id='tomo000')
        assert reconciled == []

    def test_union_keeps_picks_found_by_only_one_picker(self) -> None:
        picks_by_picker = {
            'stamp-native': [_pick('tomo000', (100.0, 100.0, 100.0), 'stamp-native')],
            'membrain-pick': [_pick('tomo000', (500.0, 500.0, 500.0), 'membrain-pick')],
        }
        reconciled = reconcile_picks(picks_by_picker, 'union', distance_threshold=15.0, tomogram_id='tomo000')
        assert len(reconciled) == 2

    def test_single_picker_intersection_equals_union(self) -> None:
        picks_by_picker = {
            'stamp-native': [
                _pick('tomo000', (100.0, 100.0, 100.0), 'stamp-native'),
                _pick('tomo000', (500.0, 500.0, 500.0), 'stamp-native'),
            ]
        }
        as_intersection = reconcile_picks(picks_by_picker, 'intersection', distance_threshold=15.0, tomogram_id='tomo000')
        as_union = reconcile_picks(picks_by_picker, 'union', distance_threshold=15.0, tomogram_id='tomo000')
        assert len(as_intersection) == len(as_union) == 2

# TestConsensus: class containing unit tests for consensus command
class TestConsensus:
    def test_reconciled_position_is_centroid_of_component(self) -> None:
        picks_by_picker = {
            'stamp-native': [_pick('tomo000', (0.0, 0.0, 0.0), 'stamp-native')],
            'membrain-pick': [_pick('tomo000', (10.0, 0.0, 0.0), 'membrain-pick')],
        }
        reconciled = reconcile_picks(picks_by_picker, 'union', distance_threshold=15.0, tomogram_id='tomo000')
        assert len(reconciled) == 1
        assert reconciled[0].position == (5.0, 0.0, 0.0)


    def test_orientation_carried_over_when_available(self) -> None:
        picks_by_picker = {
            'stamp-native': [_pick('tomo000', (0.0, 0.0, 0.0), 'stamp-native')],
            'membrain-pick': [
                _pick('tomo000', (5.0, 0.0, 0.0), 'membrain-pick', orientation=(1.0, 0.0, 0.0, 0.0))
            ],
        }
        reconciled = reconcile_picks(picks_by_picker, 'union', distance_threshold=15.0, tomogram_id='tomo000')
        assert reconciled[0].orientation == (1.0, 0.0, 0.0, 0.0)


    def test_reconcile_filters_to_requested_tomogram_only(self) -> None:
        picks_by_picker = {
            'stamp-native': [
                _pick('tomo000', (0.0, 0.0, 0.0), 'stamp-native'),
                _pick('tomo001', (0.0, 0.0, 0.0), 'stamp-native'),
            ]
        }
        reconciled = reconcile_picks(picks_by_picker, 'union', distance_threshold=15.0, tomogram_id='tomo000')
        assert len(reconciled) == 1
        assert reconciled[0].tomogram_id == 'tomo000'

    def test_build_particle_set_assigns_ids_and_half_sets(self) -> None:
        reconciled_picks = [
            _pick('tomo000', (0.0, 0.0, 0.0), 'stamp-native+membrain-pick'),
            _pick('tomo001', (5.0, 5.0, 5.0), 'stamp-native+membrain-pick'),
        ]
        particle_set = build_particle_set(
            reconciled_picks=reconciled_picks,
            consensus_rule='intersection',
            contributing_pickers=['stamp-native', 'membrain-pick'],
            half_set_seed=1,
        )
        assert len(particle_set.particles) == 2
        assert len({particle.particle_id for particle in particle_set.particles}) == 2
        assert all(particle.half_set is not None for particle in particle_set.particles)
