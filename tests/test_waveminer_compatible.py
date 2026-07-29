import numpy as np

from gcamp_analysis.waves.waveminer_compatible import (
    WaveMinerCompatibleConfig,
    segment_phasic_components,
)


def test_space_time_flood_fill_keeps_connected_front():
    active = np.zeros((20, 20, 20), dtype=bool)
    for frame in range(4, 14):
        active[frame, 5:10, frame - 2 : frame + 1] = True
    config = WaveMinerCompatibleConfig(
        block_size_pixels=1,
        pixel_size_um=10.0,
        fs=10.0,
        minimum_component_voxels=8,
        minimum_spatial_pixels=4,
        minimum_propagation_extent_um=50.0,
        minimum_arrival_span_frames=3,
        propagation_null_repeats=19,
        speed_frame_separation=5,
    )
    components, labels = segment_phasic_components(active, config)
    assert labels.max() == 1
    assert len(components) == 1
    assert components.iloc[0]["propagating_candidate"]
    assert components.iloc[0]["arrival_span_frames"] >= 8


def test_space_time_flood_fill_separates_distant_events():
    active = np.zeros((12, 20, 20), dtype=bool)
    active[2:5, 2:5, 2:5] = True
    active[7:10, 14:17, 14:17] = True
    config = WaveMinerCompatibleConfig(
        block_size_pixels=1,
        minimum_component_voxels=4,
        minimum_spatial_pixels=4,
        propagation_null_repeats=9,
    )
    components, labels = segment_phasic_components(active, config)
    assert labels.max() == 2
    assert len(components) == 2
