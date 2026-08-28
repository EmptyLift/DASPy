import numpy as np
from nptdms import ChannelObject, GroupObject, RootObject, TdmsWriter

from daspy import DASDateTime, read
from daspy.core.dasdatetime import utc


def test_read_ovlink_tdms_auto_detection_and_metadata(tmp_path):
    path = tmp_path / 'ovlink.tdms'
    properties = {
        'Year': 2026.0,
        'Month': 8.0,
        'Day': 28.0,
        'Hour': 10.0,
        'Minute': 7.0,
        'Second': 54.25,
        'Sampling Frequency (Hz)': 1000.0,
        'Spatial Resolution (m)': 0.5,
        'Start Distance (m)': 2.0,
        'Gauge length': 4.0,
        'Sensor Number': 3.0,
    }
    expected = np.arange(15, dtype=np.float32).reshape(3, 5)
    channels = [
        ChannelObject('Data', f'Channel{index + 2}', values)
        for index, values in enumerate(expected)
    ]
    with TdmsWriter(path) as tdms_file:
        tdms_file.write_segment([
            RootObject(properties=properties), GroupObject('Data'), *channels])

    data, metadata = read(path, output_type='array', chmin=3, chmax=5,
                          spmin=1, spmax=4)

    np.testing.assert_array_equal(data, expected[1:3, 1:4])
    assert metadata['file_format'] == \
        'Wuhan Optical Valley Interlink (Ovlink)'
    assert metadata['dx'] == 0.5
    assert metadata['fs'] == 1000.0
    assert metadata['start_channel'] == 3
    assert metadata['start_distance'] == 2.5
    assert metadata['gauge_length'] == 4.0
    assert metadata['start_time'] == DASDateTime(
        2026, 8, 28, 10, 7, 54, 251000, tzinfo=utc)


def test_read_ovlink_tdms_headonly(tmp_path):
    path = tmp_path / 'ovlink.tdms'
    properties = {
        'Year': 2026.0,
        'Month': 8.0,
        'Day': 28.0,
        'Hour': 10.0,
        'Minute': 7.0,
        'Second': 54.0,
        'Sampling Frequency (Hz)': 1000.0,
        'Spatial Resolution (m)': 0.5,
        'Sensor Number': 2.0,
    }
    with TdmsWriter(path) as tdms_file:
        tdms_file.write_segment([
            RootObject(properties=properties), GroupObject('Data'),
            ChannelObject('Data', 'Channel0', np.ones(4, dtype=np.float32)),
            ChannelObject('Data', 'Channel1', np.ones(4, dtype=np.float32)),
        ])

    data, metadata = read(path, output_type='array', headonly=True,
                          file_format='ovlink')

    assert data.shape == (2, 4)
    assert data.dtype == np.float32
    assert not np.any(data)
    assert metadata['file_format'] == \
        'Wuhan Optical Valley Interlink (Ovlink)'
