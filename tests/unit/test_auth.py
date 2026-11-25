from badebhai.app import auth


def test_passwords_are_salted_and_verified():
    h1, h2 = auth.hash_password("correct horse"), auth.hash_password("correct horse")
    assert h1 != h2 and h1.startswith("scrypt$")
    assert auth.verify_password("correct horse", h1)
    assert not auth.verify_password("wrong", h1)
    assert not auth.verify_password("x", "garbage")


def test_sessions_are_signed_and_expire():
    tok = auth.sign_session("alice", "s3cret", now=1000)
    assert auth.read_session(tok, "s3cret", max_age_s=60, now=1030) == "alice"
    assert auth.read_session(tok, "s3cret", max_age_s=60, now=1100) is None  # expired
    assert auth.read_session(tok, "other-secret", max_age_s=60, now=1030) is None
    import base64

    _, issued, sig = base64.urlsafe_b64decode(tok).decode().rsplit("|", 2)
    forged = base64.urlsafe_b64encode(
        f"root|{issued}|{sig}".encode()
    ).decode()  # alice's signature, root's name
    assert auth.read_session(forged, "s3cret", max_age_s=60, now=1030) is None
    assert auth.read_session("not-base64!!", "s3cret", 60) is None


def test_api_keys_are_random_and_hashed():
    k1, k2 = auth.new_api_key(), auth.new_api_key()
    assert k1 != k2 and k1.startswith("bb_") and auth.hash_api_key(k1) != k1


def test_usage_limiter_sliding_window():
    lim = auth.UsageLimiter()
    assert [lim.check("u", "ask", 2, 60, now=t) for t in (0, 1)] == [None, None]
    assert lim.check("u", "ask", 2, 60, now=2) == 59  # wait until the first event leaves the window
    assert lim.check("other", "ask", 2, 60, now=2) is None
    assert lim.check("u", "ask", 2, 60, now=61) is None
    assert lim.check("u", "ask", 0, 60) is None  # 0 = unlimited
