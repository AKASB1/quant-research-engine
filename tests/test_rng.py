"""Stream derivation (shared contract 1.2) against the known answers of Appendix B."""

from quant_research_engine.rng import fnv1a64, splitmix64, stream, stream_seeds


def test_fnv1a64_known_answers():
    assert fnv1a64("") == 0xCBF29CE484222325
    assert fnv1a64("a") == 0xAF63DC4C8601EC8C
    assert fnv1a64("foobar") == 0x85944171F73967E8


def test_splitmix64_known_answers():
    assert splitmix64(0) == 0xE220A8397B1DCDAF
    assert splitmix64(1) == 0x910A2DEC89025CC1


def test_stream_known_answers():
    s1, s2 = stream_seeds(42, "alpha")
    assert s1 == 0xDDA774F898BBCFB5
    assert s2 == 0xE8845643B324C5EA
    g = stream(42, "alpha")
    assert g.random(3).tolist() == [0.16959585488328055, 0.282582992635685, 0.6960747057340525]


def test_streams_are_independent_and_reproducible():
    a = stream(7, "x").random(5)
    b = stream(7, "x").random(5)
    c = stream(7, "y").random(5)
    assert a.tolist() == b.tolist()
    assert a.tolist() != c.tolist()
