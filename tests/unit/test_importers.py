'''
STAMP: unit tests for the picks importer
'''

# Import external dependencies
import json, pandas as pd, pytest, starfile
from pathlib import Path

# Import internal STAMP objects
from stamp.picking.importers import read_picks
from stamp.utils.errors import StampValidationError

# TestReadPicks: class containing tests for read_picks
class TestReadPicks:
    def test_csv_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / 'picks.csv'
        path.write_text('tomogram_id,x,y,z,confidence\nt00,1.5,2.5,3.5,0.9\nt01,4,5,6,\n')
        picks = read_picks(path, source_picker='mine')
        assert [p.tomogram_id for p in picks] == ['t00', 't01']
        assert picks[0].position == (1.5, 2.5, 3.5)
        assert picks[0].confidence == 0.9 and picks[1].confidence is None
        assert all(p.source_picker == 'mine' and p.orientation is None for p in picks)

    def test_csv_quaternion_is_kept(self, tmp_path: Path) -> None:
        path = tmp_path / 'picks.csv'
        path.write_text('tomogram_id,x,y,z,qw,qx,qy,qz\nt00,1,2,3,1,0,0,0\n')
        assert read_picks(path, source_picker='mine')[0].orientation == (1.0, 0.0, 0.0, 0.0)

    def test_csv_partial_quaternion_columns_are_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / 'picks.csv'
        path.write_text('tomogram_id,x,y,z,qw\nt00,1,2,3,1\n')
        with pytest.raises(StampValidationError, match='all of'):
            read_picks(path, source_picker='mine')

    def test_csv_missing_column_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / 'picks.csv'
        path.write_text('tomogram_id,x,y\nt00,1,2\n')
        with pytest.raises(StampValidationError, match="'z'"):
            read_picks(path, source_picker='mine')

    def test_csv_bad_row_reports_line_number(self, tmp_path: Path) -> None:
        path = tmp_path / 'picks.csv'
        path.write_text('tomogram_id,x,y,z\nt00,1,2,3\nt00,oops,2,3\n')
        with pytest.raises(StampValidationError, match='line 3'):
            read_picks(path, source_picker='mine')

    def test_csv_non_unit_quaternion_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / 'picks.csv'
        path.write_text('tomogram_id,x,y,z,qw,qx,qy,qz\nt00,1,2,3,2,0,0,0\n')
        with pytest.raises(StampValidationError, match='line 2'):
            read_picks(path, source_picker='mine')

    def test_star_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / 'picks.star'
        table = pd.DataFrame({
            'rlnTomoName': ['t00', 't00'],
            'rlnCoordinateX': [10.0, 20.0],
            'rlnCoordinateY': [11.0, 21.0],
            'rlnCoordinateZ': [12.0, 22.0],
            'rlnAutopickFigureOfMerit': [0.5, 0.7],
        })
        starfile.write({'particles': table}, path)
        picks = read_picks(path, source_picker='relion')
        assert [p.position for p in picks] == [(10.0, 11.0, 12.0), (20.0, 21.0, 22.0)]
        assert [p.confidence for p in picks] == [0.5, 0.7]

    def test_star_without_coordinates_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / 'picks.star'
        starfile.write({'particles': pd.DataFrame({'rlnTomoName': ['t00'], 'rlnOriginXAngst': [1.0]})}, path)
        with pytest.raises(StampValidationError, match='rlnCoordinateX'):
            read_picks(path, source_picker='relion')

    def test_json_overwrites_source_picker(self, tmp_path: Path) -> None:
        path = tmp_path / 'picks.json'
        path.write_text(json.dumps([{'tomogram_id': 't00', 'position': [1, 2, 3], 'source_picker': 'other'}]))
        assert read_picks(path, source_picker='mine')[0].source_picker == 'mine'

    def test_jsonl_reads_one_record_per_line(self, tmp_path: Path) -> None:
        path = tmp_path / 'picks.jsonl'
        path.write_text('\n'.join(json.dumps({'tomogram_id': 't00', 'position': [i, 0, 0], 'source_picker': 'x'}) for i in range(3)))
        assert len(read_picks(path, source_picker='mine')) == 3

    def test_unknown_suffix_and_empty_file_are_rejected(self, tmp_path: Path) -> None:
        (tmp_path / 'picks.txt').write_text('')
        with pytest.raises(StampValidationError, match='unsupported'):
            read_picks(tmp_path / 'picks.txt', source_picker='mine')
        (tmp_path / 'empty.csv').write_text('tomogram_id,x,y,z\n')
        with pytest.raises(StampValidationError, match='no picks'):
            read_picks(tmp_path / 'empty.csv', source_picker='mine')
