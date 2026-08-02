import matplotlib as mpl
import pytest
from _helpers import make_explanation, make_single_explanation
from matplotlib.font_manager import FontProperties, findfont

import shap_editorial as se
from shap_editorial._theme import C_BG, FONT_STACK, set_theme


def themed_font():
    return findfont(FontProperties(family=FONT_STACK))


@pytest.fixture(params=["beeswarm", "waterfall", "bar"])
def chart(request):
    if request.param == "waterfall":
        return se.waterfall, make_single_explanation()
    return getattr(se, request.param), make_explanation()


def test_set_theme_names_concrete_faces_not_the_generic_family():
    # A Text artist copies `font.family` at creation but only resolves a
    # *generic* name at draw time, which for us happens after the chart's rc
    # context has exited. Naming real faces is what keeps the two in sync.
    with mpl.rc_context():
        set_theme()
        assert mpl.rcParams["font.family"] == FONT_STACK


def test_chart_text_keeps_themed_font_after_returning(chart):
    # Regression: charts render at savefig time, outside their rc context. If
    # the artists carry only "sans-serif" they silently fall back to DejaVu.
    fn, exp = chart
    fig, ax = fn(exp, title="Title")
    for label in (ax.get_yticklabels()[0], fig.texts[0]):
        assert "sans-serif" not in label.get_fontfamily()
        assert findfont(FontProperties(family=label.get_fontfamily())) == themed_font()


def test_chart_restores_caller_rcparams(chart):
    fn, exp = chart
    mpl.rcParams["font.size"] = 17.0
    try:
        fn(exp, title="Title", transparent=True)
        assert mpl.rcParams["font.size"] == 17.0
        assert mpl.rcParams["savefig.transparent"] is False
    finally:
        mpl.rcParams["font.size"] = mpl.rcParamsDefault["font.size"]


def test_transparent_chart_stays_transparent_after_returning(chart):
    fn, exp = chart
    fig, _ = fn(exp, transparent=True)
    assert fig.get_facecolor()[3] == 0.0


def test_opaque_chart_is_white(chart):
    fn, exp = chart
    fig, _ = fn(exp)
    assert fig.get_facecolor() == mpl.colors.to_rgba(C_BG)


def test_set_theme_is_still_global_when_called_directly():
    # set_theme stays a deliberately-global public API; only the chart
    # functions are sandboxed.
    before = mpl.rcParams["axes.labelsize"]
    try:
        set_theme()
        assert mpl.rcParams["axes.labelsize"] == 10
        assert mpl.rcParams["axes.labelsize"] != before or before == 10
    finally:
        mpl.rcParams.update(mpl.rcParamsDefault)
