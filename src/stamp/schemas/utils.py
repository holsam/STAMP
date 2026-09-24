'''
STAMP: field validators shared across schema modules
'''

# assert_unit_quaternion: raise unless (w, x, y, z) is unit-length within tolerance
def assert_unit_quaternion(value: tuple[float, float, float, float]) -> None:
    norm = sum(component**2 for component in value) ** 0.5
    if not (0.99 <= norm <= 1.01):
        from stamp.utils.errors import StampPipelineError
        raise StampPipelineError(f'orientation quaternion must be unit-length, got norm={norm:.4f}')
