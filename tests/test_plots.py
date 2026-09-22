import plotly.graph_objects as go
import pytest

from abtest import plots
from abtest.data import CONTROL, EXPERIMENTAL, OUTLIER_COLUMNS, OUTLIER_PERCENTILE, load_ab_test, remove_outliers


@pytest.fixture(scope="module")
def ab_test():
    return load_ab_test()


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_sessions_by_day_has_one_line_per_group_in_group_color(ab_test, theme):
    fig = plots.sessions_by_day(ab_test, theme=theme)

    assert isinstance(fig, go.Figure)
    assert [trace.name for trace in fig.data] == [CONTROL, EXPERIMENTAL]
    for trace in fig.data:
        assert trace.line.color == plots.GROUP_COLORS[theme][trace.name]


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_histogram_by_group_has_one_panel_per_group_in_group_color(ab_test, theme):
    fig = plots.histogram_by_group(ab_test, "order_value", "Order value", theme=theme)

    assert [trace.name for trace in fig.data] == [CONTROL, EXPERIMENTAL]
    for trace in fig.data:
        assert trace.marker.color == plots.GROUP_COLORS[theme][trace.name]
    assert [note.text for note in fig.layout.annotations] == [CONTROL, EXPERIMENTAL]


def test_log_order_value_histogram_plots_the_log_column(ab_test):
    cleaned = remove_outliers(ab_test, OUTLIER_COLUMNS, OUTLIER_PERCENTILE)

    fig = plots.log_order_value_histogram(cleaned)

    assert len(fig.data) == 2
    # Log of the order values, not the raw values (which go up to ~200).
    assert max(max(trace.x) for trace in fig.data) < 10
