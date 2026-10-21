import os

import numpy as np
import pytest

from complexity_fit import (
    Measurement,
    ModelFit,
    ShapeFilter,
    classify,
    filter_measurements,
    local_slopes,
    asymptotic_reached,
    corrected_local_slopes,
    pow2_fraction,
    parse_performance_file,
    _select_best_and_runnerup,
    signal_to_overhead_ratios,
    count_useful_points,
    PerfFileFormatError,
    PerfMeasurementsMissingError,
)

# This test file is byte-identical in the project root and in version_final/
# (kept in sync deliberately, see SPEC_version_final.md) -- PROJECT_ROOT has
# to resolve correctly from either location, one level up in version_final/,
# the same directory at the root.
_here = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(os.path.join(_here, "Algorythm_results_windows")):
    PROJECT_ROOT = _here
else:
    PROJECT_ROOT = os.path.dirname(_here)

# Real measurements (legacy format) the regression tests were written against.
TEST_DATA = os.path.join(PROJECT_ROOT, "Test_files", "test_data")


def _data(name):
    return os.path.join(TEST_DATA, name)


SQUARE_SIDES = [128, 256, 512, 1024, 2048, 4096]
N_REPS = 12
NOISE_FRAC = 0.05


def _make_measurements(law, seed):
    rng = np.random.default_rng(seed)
    measurements = []
    for side in SQUARE_SIDES:
        n = side * side
        base = law(n)
        reps = base * (1.0 + rng.normal(0, NOISE_FRAC, size=N_REPS))
        measurements.append(Measurement(
            folder=f"{side}x{side}", width=side, height=side, n_pixels=n,
            time_reps=list(reps),
        ))
    return measurements


# NOTE: the coefficients in SPEC_complexity_fit.md section 1.9 (3e-9, 2e-10,
# 1e-15, 1e-22, offset 0.13) do not clear the asymptotic-regime gate (1.6) on
# this exact 6-point square ladder: the 0.13 s overhead still dominates (or is
# comparable to) the algorithmic term even at 4096x4096, so local slopes never
# stabilize near the true exponent within tol=0.15 -- verified numerically,
# flagged back to complex-code-e7. These coefficients are scaled up so the
# algorithmic term dominates by ~30-40x at 1024x1024 (same qualitative shape:
# overhead visible at small N, clean convergence to the true exponent by
# 4096x4096), which is what the gate needs to certify a classification.
def law_n(n):
    return 3.34e-5 * n + 0.13


def law_nlogn(n):
    return 1.67e-6 * n * np.log2(n) + 0.13


def law_n2(n):
    return 3.18e-11 * n**2 + 0.13


def law_n3(n):
    return 3.26e-17 * n**3 + 0.13


def law_buried(n):
    # asymptotic term is negligible compared to the fixed overhead everywhere
    # in the measured range: overhead dominates, no regime is reached.
    return 1e-20 * n**2 + 0.13


def law_n2_5(n):
    return 2e-12 * n**2.5 + 0.13


@pytest.mark.parametrize("law,expected", [
    (law_n, ("O(N)",)),
    # Hardened: O(N log N) used to be accepted as a tie with O(N) because the
    # model was silently broken (see test_nlogn_model_is_not_degenerate) and
    # could never actually win regardless of the data. Fixed, it must win
    # outright on data generated exactly as N*log2(N).
    (law_nlogn, ("O(N log N)",)),
    (law_n2, ("O(N^2)",)),
    (law_n3, ("O(N^3)",)),
])
def test_known_complexity_is_classified_correctly(law, expected):
    measurements = _make_measurements(law, seed=42)
    verdict = classify(measurements, field_name="time_reps",
                        shape_filter=ShapeFilter.NONE, n_bootstrap=200,
                        rng=np.random.default_rng(1))
    assert verdict.model_name in expected, (
        f"expected one of {expected}, got {verdict.model_name} "
        f"(abstain_reason={verdict.abstain_reason}, ranking={verdict.ranking})"
    )
    assert verdict.abstain_reason is None
    assert "[N_scale" not in verdict.equation
    assert "[" not in verdict.equation


def test_nlogn_model_is_not_degenerate():
    # Regression test for a real bug: fit_model used to normalize x by
    # dividing by x_scale BEFORE evaluating f, landing every point in (0, 1].
    # _f_nlogn's np.maximum(N, 1.0) safety clamp (meant to avoid log(0) for
    # tiny real N) then clamped every normalized point to exactly 1.0, whose
    # log2 is 0 -- so O(N log N) always evaluated to the zero function and
    # degenerated to a constant, identical to O(1) down to the last digit.
    # O(N log N) could never win against anything as a result.
    ms = _make_measurements(law_nlogn, seed=42)
    from complexity_fit import fit_all_models, aggregate_measurements, filter_measurements
    filtered, _ = filter_measurements(ms, ShapeFilter.NONE)
    points = aggregate_measurements(filtered, "time_reps")
    x = np.array([p.n_pixels for p in points])
    y = np.array([p.value for p in points])
    fits = {f.name: f for f in fit_all_models(x, y)}

    nlogn_fit = fits["O(N log N)"]
    o1_fit = fits["O(1)"]
    assert nlogn_fit.discarded_reason is None
    assert not np.isclose(nlogn_fit.bic, o1_fit.bic), (
        f"O(N log N) collapsed to O(1): both gave BIC={nlogn_fit.bic:.6f} -- "
        "the model is degenerate again"
    )
    assert not np.isclose(nlogn_fit.r2, o1_fit.r2)
    # a real fit should also beat a flat O(1) badly on data generated with a
    # genuine growth term
    assert nlogn_fit.bic < o1_fit.bic - 10


def test_signal_to_overhead_ratios_and_useful_points():
    y = np.array([0.13, 0.14, 0.20, 0.50, 2.0, 10.0])
    b = 0.13
    ratios = signal_to_overhead_ratios(y, b)
    assert ratios[0] == pytest.approx(0.0)
    assert ratios[1] == pytest.approx((0.14 - 0.13) / 0.13)
    assert ratios[-1] == pytest.approx((10.0 - 0.13) / 0.13)
    assert count_useful_points(ratios) == sum(1 for r in ratios if r >= 1.0)


def test_quadratic_and_frs_gray_useful_point_counts_match_measured():
    # Cross-checked against complex-code-e7's independent count for TIME:
    # quadratic.txt 4/6, performance_fft2d.txt 2/6, performance_FRS_gray.txt 1/6.
    cases = [
        (_data("quadratic.txt"), 4),
        (_data("performance_fft2d.txt"), 2),
        (_data("performance_FRS_gray.txt"), 1),
    ]
    for filename, expected_useful in cases:
        measurements = parse_performance_file(filename)
        verdict = classify(measurements, field_name="time_reps",
                            shape_filter=ShapeFilter.SQUARE, n_bootstrap=50,
                            rng=np.random.default_rng(0))
        assert verdict.n_useful_points == expected_useful, (
            f"{filename}: expected {expected_useful} useful points, got "
            f"{verdict.n_useful_points} (ratios={verdict.signal_overhead_ratios})"
        )
        assert len(verdict.signal_overhead_ratios) == verdict.n_points_used


def test_fft2d_instructions_warns_low_useful_points_despite_high_confidence():
    # The regression this closes: O(N log N) wins fft2d instructions with
    # 100% bootstrap confidence, but only 2/6 points actually carry signal
    # above the overhead (0.00, 0.02, 0.11, 0.43, 2.03, 8.46) -- bootstrap
    # confidence measures stability against repetition noise, not against a
    # shortage of informative points, so it can't be trusted alone here.
    # Must NOT abstain (the O(N log N) call is physically correct for an
    # FFT-based algorithm) but MUST carry a warning saying so.
    measurements = parse_performance_file(_data("performance_fft2d.txt"))
    verdict = classify(measurements, field_name="instr_reps",
                        shape_filter=ShapeFilter.SQUARE, n_bootstrap=200,
                        rng=np.random.default_rng(6))
    assert verdict.model_name == "O(N log N)"
    assert verdict.confidence == pytest.approx(1.0)
    assert verdict.n_useful_points < verdict.n_points_used / 2
    assert any("points" in w and "informative" in w for w in verdict.warnings)


def test_useful_points_warning_absent_when_majority_are_useful():
    measurements = parse_performance_file(_data("quadratic.txt"))
    verdict = classify(measurements, field_name="instr_reps",
                        shape_filter=ShapeFilter.SQUARE, n_bootstrap=200,
                        rng=np.random.default_rng(6))
    assert verdict.n_useful_points >= verdict.n_points_used / 2
    assert not any("informative" in w for w in verdict.warnings)


def test_abstain_message_names_both_models_when_bic_and_bootstrap_disagree():
    # Regression: on performance_fft2d.txt time with the ASPECT_MATCHED
    # default, the point-estimate (BIC) winner is "O(N^p) free" but it only
    # wins 7% of bootstrap iterations, while "O(N^1.5)" wins 66% of them.
    # The old message called "O(N^p) free" "el modelo ganador" and cited its
    # 7% -- naming the least-frequent model as the winner, reading as a
    # contradiction. The fixed message must name both models and both
    # percentages, and the percentage attributed to each name must actually
    # be that model's own bootstrap frequency.
    measurements = parse_performance_file(_data("performance_fft2d.txt"))
    verdict = classify(measurements, field_name="time_reps",
                        shape_filter=ShapeFilter.ASPECT_MATCHED, n_bootstrap=300,
                        rng=np.random.default_rng(6))

    assert verdict.model_name is None
    assert verdict.bic_best_model != verdict.bootstrap_top_model
    assert verdict.bic_best_model in verdict.abstain_reason
    assert verdict.bootstrap_top_model in verdict.abstain_reason

    freq_by_name = {r["name"]: r["confidence"] for r in verdict.ranking}
    bic_freq_str = f"{freq_by_name[verdict.bic_best_model]:.0%}"
    bootstrap_freq_str = f"{verdict.bootstrap_top_confidence:.0%}"
    assert bic_freq_str in verdict.abstain_reason
    assert bootstrap_freq_str in verdict.abstain_reason
    # the message must not call the low-frequency BIC winner "el ganador"
    # without qualification -- it should frame this as two criteria
    # disagreeing, not report a single unqualified "winner"
    assert "discrepancy" in verdict.abstain_reason


def test_confidence_matches_bootstrap_top_when_verdict_succeeds():
    # When classify() actually returns a verdict (didn't abstain), the BIC
    # winner and the bootstrap plurality winner must agree -- mathematically
    # guaranteed once confidence clears the 60% threshold (two models can't
    # both exceed half the iterations), but worth confirming on real data
    # rather than trusting the proof alone.
    cases = [
        (_data("quadratic.txt"), "time_reps", ShapeFilter.SQUARE),
        (_data("quadratic.txt"), "instr_reps", ShapeFilter.SQUARE),
        (_data("performance_fft2d.txt"), "instr_reps", ShapeFilter.ASPECT_MATCHED),
    ]
    for filename, field_name, shape_filter in cases:
        measurements = parse_performance_file(filename)
        verdict = classify(measurements, field_name=field_name,
                            shape_filter=shape_filter, n_bootstrap=300,
                            rng=np.random.default_rng(6))
        assert verdict.model_name is not None, (filename, field_name)
        assert verdict.model_name == verdict.bic_best_model
        assert verdict.bic_best_model == verdict.bootstrap_top_model
        assert verdict.confidence == pytest.approx(verdict.bootstrap_top_confidence)


def test_pure_noise_around_constant_abstains_on_no_signal_not_asymptotic():
    # Gate ordering regression: pure noise around a flat constant has no
    # scaling signal at all -- more image sizes would never reveal one. The
    # abstain reason must say so (no signal / ties with O(1)), not "regimen
    # asintotico no alcanzado, se necesitan tamanos mayores", which sends the
    # user chasing a signal that doesn't exist.
    rng = np.random.default_rng(11)
    measurements = []
    for side in SQUARE_SIDES:
        n = side * side
        reps = list(0.5 * (1 + rng.normal(0, 0.03, N_REPS)))
        measurements.append(Measurement(folder=f"{side}x{side}", width=side,
                                         height=side, n_pixels=n, time_reps=reps))
    verdict = classify(measurements, field_name="time_reps",
                        shape_filter=ShapeFilter.NONE, n_bootstrap=200,
                        rng=np.random.default_rng(1))
    assert verdict.model_name is None
    assert "asymptotic regime" not in verdict.abstain_reason
    assert ("O(1)" in verdict.abstain_reason or "scaling" in verdict.abstain_reason)


def test_buried_asymptotic_term_abstains():
    measurements = _make_measurements(law_buried, seed=7)
    verdict = classify(measurements, field_name="time_reps",
                        shape_filter=ShapeFilter.NONE, n_bootstrap=200,
                        rng=np.random.default_rng(2))
    assert verdict.model_name is None
    assert verdict.abstain_reason is not None


def test_n_to_2_5_selects_free_exponent_model():
    measurements = _make_measurements(law_n2_5, seed=99)
    verdict = classify(measurements, field_name="time_reps",
                        shape_filter=ShapeFilter.NONE, n_bootstrap=200,
                        rng=np.random.default_rng(3))
    assert verdict.model_name == "O(N^p) free", (
        f"got {verdict.model_name}, ranking={verdict.ranking}"
    )
    p_reported = [r for r in verdict.ranking if r["name"] == "O(N^p) free"]
    assert p_reported


def test_local_slopes_and_asymptotic_reached():
    x = np.array([1, 2, 4, 8, 16], dtype=float)
    y = x  # perfectly linear -> constant slope 1
    slopes = local_slopes(x, y)
    assert all(abs(s - 1.0) < 1e-9 for s in slopes)
    assert asymptotic_reached(slopes)

    # legitimate monotonic convergence toward an asymptote must NOT be
    # rejected just for being monotonic (range is small relative to mean).
    assert asymptotic_reached([0.80, 0.95, 0.99, 1.00])

    # sustained drift: significant trend AND range > 15% of the mean.
    assert not asymptotic_reached([0.20, 0.35, 0.55, 0.78])

    # excess noise: lost monotonicity, range > 50% of the mean, no
    # significant trend -- still must not be accepted as converged.
    slopes_noisy = [0.01, -0.00, 0.10, 0.05, 0.43]
    assert not asymptotic_reached(slopes_noisy)

    # noisy but within tolerance around a stable value: accepted.
    assert asymptotic_reached([1.05, 0.93, 1.08])


def test_shape_filter_square():
    measurements = [
        Measurement(folder="512x512", width=512, height=512, n_pixels=512*512),
        Measurement(folder="640x480", width=640, height=480, n_pixels=640*480),
    ]
    kept, discarded = filter_measurements(measurements, ShapeFilter.SQUARE)
    assert len(kept) == 1
    assert discarded == 1
    assert kept[0].width == kept[0].height


def test_shape_filter_square_no_pow2():
    measurements = [
        Measurement(folder="512x512", width=512, height=512, n_pixels=512*512),
        Measurement(folder="300x300", width=300, height=300, n_pixels=300*300),
    ]
    kept, discarded = filter_measurements(measurements, ShapeFilter.SQUARE_NO_POW2)
    assert len(kept) == 1
    assert kept[0].width == 300


def test_shape_filter_aspect_matched():
    measurements = [
        Measurement(folder="640x480", width=640, height=480, n_pixels=640*480),
        Measurement(folder="1280x960", width=1280, height=960, n_pixels=1280*960),
        Measurement(folder="512x512", width=512, height=512, n_pixels=512*512),
    ]
    kept, discarded = filter_measurements(measurements, ShapeFilter.ASPECT_MATCHED)
    assert len(kept) == 2
    assert discarded == 1


def _fit(name, bic, p=None):
    return ModelFit(name=name, a=1.0, b=0.1, p=p, p_stderr=None, r2=0.99,
                     bic=bic, equation="y = ...")


def test_select_best_reports_tie_within_bic_margin():
    # O(N) and O(N log N) within BIC_MARGIN of each other: genuinely
    # ambiguous, classify() must abstain rather than pick one arbitrarily.
    fits = [_fit("O(N)", -50.0), _fit("O(N log N)", -46.0),
            _fit("O(N^2)", -10.0), _fit("O(N^p) free", -50.5, p=1.0)]
    best, runner_up = _select_best_and_runnerup(fits)
    assert best.name == "O(N)"
    assert runner_up.name == "O(N log N)"
    assert (runner_up.bic - best.bic) < 6.0  # would trigger classify()'s tie abstention


def test_select_best_no_tie_with_clear_margin():
    fits = [_fit("O(N)", -50.0), _fit("O(N log N)", -30.0),
            _fit("O(N^p) free", -50.5, p=1.0)]
    best, runner_up = _select_best_and_runnerup(fits)
    assert best.name == "O(N)"
    assert (runner_up.bic - best.bic) >= 6.0


def test_corrected_local_slopes_drops_nonpositive_points():
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    b = 0.9
    y = np.array([1.0, 1.85, 3.9, 8.0, 16.0])  # y[1]-b = 0.95 > 0, all positive here
    slopes, n_dropped, n_used = corrected_local_slopes(x, y, b)
    assert n_dropped == 0
    assert n_used == 5

    y_bad = np.array([1.0, 0.85, 3.9, 8.0, 16.0])  # y_bad[1]-b = -0.05 <= 0
    slopes2, n_dropped2, n_used2 = corrected_local_slopes(x, y_bad, b)
    assert n_dropped2 == 1
    assert n_used2 == 4
    assert len(slopes2) == 3  # one slope per consecutive pair among the 4 kept


def test_aspect_matched_beats_square_default_on_fft2d():
    # Why ASPECT_MATCHED is the default now: on this real file, SQUARE
    # selects 6 points that are all powers of two (triggers the cache-bias
    # warning), while ASPECT_MATCHED selects the 8 points sharing a 4:3
    # aspect ratio, none a power of two -- more points, no bias warning,
    # same physically-correct verdict (O(N log N)).
    measurements = parse_performance_file(_data("performance_fft2d.txt"))

    v_square = classify(measurements, field_name="instr_reps",
                         shape_filter=ShapeFilter.SQUARE, n_bootstrap=200,
                         rng=np.random.default_rng(6))
    v_aspect = classify(measurements, field_name="instr_reps",
                         shape_filter=ShapeFilter.ASPECT_MATCHED, n_bootstrap=200,
                         rng=np.random.default_rng(6))

    assert v_aspect.n_points_used > v_square.n_points_used
    assert not any("power of two" in w for w in v_aspect.warnings)
    assert any("power of two" in w for w in v_square.warnings)
    assert v_square.model_name == "O(N log N)"
    assert v_aspect.model_name == "O(N log N)"


def test_classify_default_shape_filter_is_aspect_matched():
    import inspect
    default = inspect.signature(classify).parameters["shape_filter"].default
    assert default == ShapeFilter.ASPECT_MATCHED


def test_pow2_fraction():
    measurements = [
        Measurement(folder="512x512", width=512, height=512, n_pixels=512*512),
        Measurement(folder="1024x1024", width=1024, height=1024, n_pixels=1024*1024),
        Measurement(folder="300x300", width=300, height=300, n_pixels=300*300),
    ]
    assert abs(pow2_fraction(measurements) - 2 / 3) < 1e-9


def test_too_few_points_abstains():
    measurements = _make_measurements(law_n, seed=1)[:3]
    verdict = classify(measurements, field_name="time_reps",
                        shape_filter=ShapeFilter.NONE, n_bootstrap=50,
                        rng=np.random.default_rng(4))
    assert verdict.model_name is None
    assert "points" in verdict.abstain_reason


def test_parser_raises_format_error_on_unrecognized_file(tmp_path):
    # No '=== FOLDER:' markers at all -- not a perf-results file.
    bogus = tmp_path / "not_a_perf_file.txt"
    bogus.write_text("this is just some random text\nwith no structure at all\n")
    with pytest.raises(PerfFileFormatError):
        parse_performance_file(str(bogus))


def test_parser_raises_measurements_missing_error_when_perf_failed(tmp_path):
    # Same shape as the real Resultados/*.txt files: folder/repetition
    # structure present, but perf itself failed on every repetition, so
    # there isn't a single measurement anywhere in the file.
    content = (
        "Performance analysis - linear_scan_and_sum\n"
        "==================================================\n\n"
        "=== FOLDER: 1024x1024 ===\n"
        "--- IMAGE: hologram_01.png ---\n"
        "  Repetition 1:\n"
        "WARNING: perf not found for kernel 6.6.87.2-microsoft\n\n"
        "  You may need to install the following packages for this specific kernel:\n"
        "    linux-tools-6.6.87.2-microsoft-standard-WSL2\n\n"
        "  Repetition 2:\n"
        "WARNING: perf not found for kernel 6.6.87.2-microsoft\n\n"
    )
    bogus = tmp_path / "perf_failed.txt"
    bogus.write_text(content)
    with pytest.raises(PerfMeasurementsMissingError) as exc_info:
        parse_performance_file(str(bogus))
    assert "perf not found for kernel" in str(exc_info.value)


def test_parser_measurements_missing_error_without_warning_text(tmp_path):
    # Same structural failure (folders exist, zero measurements) but no
    # WARNING line to quote -- must still raise, with a generic reason.
    content = (
        "=== FOLDER: 512x512 ===\n"
        "--- IMAGE: img1.png ---\n"
        "  Repetition 1:\n"
        "(nothing useful happened here)\n"
    )
    bogus = tmp_path / "no_warning.txt"
    bogus.write_text(content)
    with pytest.raises(PerfMeasurementsMissingError) as exc_info:
        parse_performance_file(str(bogus))
    assert "check that perf" in str(exc_info.value)


def test_parser_handles_algorythm_results_windows_regression():
    # Regression: these real files already worked before the diagnostic
    # exceptions were added -- must keep working (14 folders each), not get
    # caught by the new format/measurements-missing checks.
    for filename in ["Results_FRS.txt", "linear.txt"]:
        path = _data(filename)
        measurements = parse_performance_file(path)
        assert len(measurements) == 14, (filename, len(measurements))


def test_parser_handles_real_file_without_crashing():
    measurements = parse_performance_file(_data("quadratic.txt"))
    assert len(measurements) > 0
    for m in measurements:
        assert m.time_reps
        assert m.n_pixels == m.width * m.height
    # sorted by n_pixels
    pixels = [m.n_pixels for m in measurements]
    assert pixels == sorted(pixels)


def test_quadratic_txt_classifies_as_linear_on_instructions():
    measurements = parse_performance_file(_data("quadratic.txt"))
    verdict = classify(measurements, field_name="instr_reps",
                        shape_filter=ShapeFilter.SQUARE, n_bootstrap=200,
                        rng=np.random.default_rng(5))
    assert verdict.model_name is not None, verdict.abstain_reason
    assert verdict.model_name == "O(N)"


def test_frs_gray_time_abstains_and_instructions_tie_between_n_and_nlogn():
    # Time: total elapsed time grows far too little across the ladder for the
    # interpreter-startup overhead to ever wash out -- must abstain.
    # Instructions: after the O(N log N) normalization fix, O(N log N) and
    # O(N) are within the BIC_MARGIN of each other here (delta ~2.6) -- an
    # honest tie, must abstain rather than arbitrarily pick one.
    measurements = parse_performance_file(_data("performance_FRS_gray.txt"))

    v_time = classify(measurements, field_name="time_reps",
                       shape_filter=ShapeFilter.SQUARE, n_bootstrap=200,
                       rng=np.random.default_rng(6))
    assert v_time.model_name is None
    assert v_time.abstain_reason is not None

    v_instr = classify(measurements, field_name="instr_reps",
                        shape_filter=ShapeFilter.SQUARE, n_bootstrap=200,
                        rng=np.random.default_rng(6))
    assert v_instr.model_name is None
    assert "tie" in v_instr.abstain_reason


def test_fft2d_time_abstains_but_instructions_classify_nlogn():
    # Before the O(N log N) normalization fix this misclassified as O(N):
    # the model was silently collapsing to a constant (see
    # test_nlogn_model_is_not_degenerate) and could never win. Fixed, it
    # correctly wins here -- the expected answer for a 2D FFT-based algorithm.
    measurements = parse_performance_file(_data("performance_fft2d.txt"))

    v_time = classify(measurements, field_name="time_reps",
                       shape_filter=ShapeFilter.SQUARE, n_bootstrap=200,
                       rng=np.random.default_rng(6))
    assert v_time.model_name is None
    assert v_time.abstain_reason is not None

    v_instr = classify(measurements, field_name="instr_reps",
                        shape_filter=ShapeFilter.SQUARE, n_bootstrap=200,
                        rng=np.random.default_rng(6))
    assert v_instr.model_name == "O(N log N)"
    assert v_instr.abstain_reason is None


# ---------------------------------------------------------------------------
# Protocolo v2 (bloques LI / L1 / LK): coste por llamada = (LK - L1) / (K - 1)
# ---------------------------------------------------------------------------

def _perf_block(label, k, rep, instr, seconds, dims="64x64"):
    return (
        f"  [BLOCK {label} K={k}] Repetition {rep}:\n"
        f"IMAGE_DIMENSIONS: {dims}\n\n"
        " Performance counter stats for 'python3 x.py':\n\n"
        f"   {instr * 2:>14}      cycles:u\n"
        f"   {instr:>14}      instructions:u\n\n"
        f"       {seconds:.9f} seconds time elapsed\n\n"
    )


def _v2_file(path, folders):
    """folders: {nombre: (K, instr_base, instr_por_llamada)}: instr_base es el
    arranque + carga; cada repeticion agrega +-1000 de ruido determinista."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("Performance analysis - test\n\n--- BASELINE L ---\n")
        f.write(_perf_block("L", 0, 1, 500_000_000, 0.2))
        for name, (k, base, per_call) in folders.items():
            f.write(f"=== FOLDER: {name} ===\n--- IMAGE: a.png ---\n")
            f.write(f"  [PROTOCOL v2] K={k} reps=3\n")
            for rep in range(1, 4):
                f.write(_perf_block("LI", 0, rep, base + rep * 1000, 0.3, name))
                f.write(_perf_block("L1", 1, rep, base + rep * 1000 + per_call, 0.3 + 0.01, name))
                f.write(_perf_block("LK", k, rep, base + rep * 1000 + k * per_call, 0.3 + 0.01 * k, name))


def test_v2_cold_estimator_is_l1_minus_li(tmp_path):
    p = tmp_path / "v2_cold.txt"
    _v2_file(p, {"64x64": (5, 600_000_000, 2_000_000)})
    m = parse_performance_file(str(p))[0]
    # arranque y carga (LI) se restan; queda la 1a llamada, sin importar K
    # (el ruido +-1000 por repeticion no se cancela porque LI se resta por mediana)
    assert m.instr_reps == pytest.approx([2_000_000] * 3, abs=1500)


def test_v2_parser_recovers_per_call_cost_and_ignores_startup(tmp_path):
    p = tmp_path / "v2.txt"
    _v2_file(p, {"64x64": (5, 600_000_000, 2_000_000)})
    ms = parse_performance_file(str(p), estimator="warm")
    assert len(ms) == 1
    m = ms[0]
    assert (m.width, m.height, m.n_pixels) == (64, 64, 4096)
    # el arranque (6e8) y la carga no aparecen; queda exactamente una llamada
    assert m.instr_reps == pytest.approx([2_000_000] * 3)
    assert m.time_reps == pytest.approx([0.01] * 3)


def test_v2_parser_drops_nonpositive_differences_instead_of_clamping(tmp_path):
    p = tmp_path / "v2_noise.txt"
    with open(p, "w", encoding="utf-8") as f:
        f.write("=== FOLDER: 64x64 ===\n--- IMAGE: a.png ---\n  [PROTOCOL v2] K=4 reps=1\n")
        f.write(_perf_block("L1", 1, 1, 1_000_000_000, 0.50))
        f.write(_perf_block("LK", 4, 1, 1_010_000_000, 0.45))  # tiempo: LK < L1
    m = parse_performance_file(str(p), estimator="warm")[0]
    assert m.instr_reps == pytest.approx([10_000_000 / 3])
    assert m.time_reps == []  # ruido de tiempo: se descarta, no se fuerza


def test_v2_classify_uses_corrected_values_to_find_exponent(tmp_path):
    # costo por llamada ~ N**1.5 (como process_nm2), con base enorme: sin la
    # correccion el arranque de 5e8 dominaria todos los puntos.
    p = tmp_path / "v2_law.txt"
    sides = [128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
    # ruido determinista del +-2%: un ajuste exacto da BIC=-inf y select_best lo
    # descarta por isfinite (no ocurre con mediciones reales)
    folders = {f"{s}x{s}": (3, 500_000_000, int(0.5 * (s * s) ** 1.5 * (1 + 0.02 * np.sin(s))))
               for s in sides}
    _v2_file(p, folders)
    ms = parse_performance_file(str(p))
    v = classify(ms, field_name="instr_reps", shape_filter=ShapeFilter.SQUARE,
                 n_bootstrap=100, rng=np.random.default_rng(1))
    assert v.model_name == "O(N^1.5)"


def test_legacy_format_still_parses_unchanged():
    # el formato sin marcador [PROTOCOL v2] sigue por el camino de siempre
    ms = parse_performance_file(_data("performance_fft2d.txt"))
    assert len(ms) > 0 and all(m.instr_reps for m in ms)
