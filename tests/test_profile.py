from tfmlab.profile import gemm_rows, token_count


def test_token_count_matches_the_measured_default():
    # Context 15,360 and horizon 64 reach the transformer as 482 tokens.
    assert token_count(15360, 64) == 482


def test_token_count_for_other_contexts():
    assert token_count(2048, 64) == 66
    assert token_count(512, 64) == 18
    assert token_count(15360, 1024) > 482


def test_gemm_rows_keep_raw_samples():
    rows = gemm_rows((16,), dim=64, repeats=2)
    assert rows[0]["repeats"] == 2
    assert len(rows[0]["fp32_ms_samples"]) == 2
    assert len(rows[0]["int8_ms_samples"]) == 2
    assert len(rows[0]["int4_ms_samples"]) == 2
