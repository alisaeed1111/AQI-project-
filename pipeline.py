"""
pipeline.py — One entry point for the whole project.

Usage (run from the project root):

    python pipeline.py test-api    # check whether your live API key works
    python pipeline.py backfill    # build the full historical feature set
    python pipeline.py features    # run the feature pipeline (hourly job)
    python pipeline.py train       # train models and register the best one
    python pipeline.py predict     # print the 3-day AQI forecast
    python pipeline.py all         # backfill + train + predict (recommended first run)
"""
import os
import sys

# Make everything in src/ importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

USAGE = __doc__


def _test_api():
    """Check whether the live OpenWeather API is reachable with your key.

    This makes a single lightweight call and reports clearly what happened,
    so you never have to guess why you're seeing sample vs. live data.
    """
    import requests

    from config import (
        CITY_LAT,
        CITY_LON,
        CITY_NAME,
        DATA_MODE,
        OPENWEATHER_API_KEY,
        OW_AIR_NOW_URL,
    )

    print("=" * 60)
    print("  OpenWeather API check")
    print("=" * 60)
    print(f"  City       : {CITY_NAME}  ({CITY_LAT}, {CITY_LON})")
    print(f"  Data mode  : {DATA_MODE}")

    if not OPENWEATHER_API_KEY:
        print("  API key    : (none found)")
        print("\n  No API key is set, so the project uses realistic SAMPLE data.")
        print("  To switch to live online data:")
        print("    1. Get a free key: https://home.openweathermap.org/users/sign_up")
        print("    2. Copy it from:   https://home.openweathermap.org/api_keys")
        print("    3. Open the .env file and paste it after the = sign:")
        print("         OPENWEATHER_API_KEY=your_key_here")
        print("    4. Re-run this check:  python pipeline.py test-api")
        return

    masked = (
        OPENWEATHER_API_KEY[:4] + "…" + OPENWEATHER_API_KEY[-4:]
        if len(OPENWEATHER_API_KEY) > 8
        else "(set)"
    )
    print(f"  API key    : detected ({masked})")
    print("\n  Calling the live air-pollution endpoint…")

    try:
        resp = requests.get(
            OW_AIR_NOW_URL,
            params={"lat": CITY_LAT, "lon": CITY_LON, "appid": OPENWEATHER_API_KEY},
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  ❌ Could not reach OpenWeather: {exc}")
        print("     Check your internet connection and try again.")
        return

    if resp.status_code == 200:
        from aqi import compute_aqi

        comp = resp.json()["list"][0]["components"]
        aqi = compute_aqi(comp.get("pm2_5"), comp.get("pm10"))
        print("  ✅ Success — live data is working!")
        print(
            f"     Right now in {CITY_NAME}: "
            f"PM2.5={comp.get('pm2_5')} µg/m³, PM10={comp.get('pm10')} µg/m³  "
            f"→  AQI {aqi}"
        )
        print("\n  You're all set. Rebuild everything on live data with:")
        print("     python pipeline.py all")
    elif resp.status_code == 401:
        print("  ❌ 401 Unauthorized — the key isn't accepted (yet).")
        print("     Brand-new keys take 1–2 hours to activate. If you just made it,")
        print("     wait a bit and re-run. Otherwise re-check the key in .env")
        print("     was pasted correctly (no quotes, no extra spaces).")
    elif resp.status_code == 429:
        print("  ❌ 429 Too Many Requests — you hit the free-tier rate limit.")
        print("     Wait a minute, then try again.")
    else:
        print(f"  ❌ HTTP {resp.status_code}: {resp.text[:200]}")


def _print_forecast():
    import predict

    fc, reg, last_date = predict.forecast()
    print("\n" + "=" * 48)
    print(f"  3-DAY AQI FORECAST  (model: {reg['model_name']})")
    print(f"  Last observed day: {last_date.date()}")
    print("=" * 48)
    for _, r in fc.iterrows():
        bar = "█" * int(r["aqi"] / 10)
        print(f"  {r['date'].date()}  AQI {r['aqi']:>5.0f}  {r['category']:<32} {bar}")
    print("=" * 48 + "\n")


def main():
    cmd = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    if cmd in ("test-api", "testapi", "test", "check", "check-api"):
        _test_api()

    elif cmd == "backfill":
        import feature_pipeline
        from config import BACKFILL_DAYS
        feature_pipeline.run(BACKFILL_DAYS)

    elif cmd == "features":
        import feature_pipeline
        feature_pipeline.run()

    elif cmd == "train":
        import train
        train.train()

    elif cmd == "predict":
        _print_forecast()

    elif cmd == "all":
        import feature_pipeline
        import train
        from config import BACKFILL_DAYS, CITY_NAME
        print(f"[pipeline] Building end-to-end AQI predictor for {CITY_NAME}…\n")
        feature_pipeline.run(BACKFILL_DAYS)
        train.train()
        _print_forecast()
        print("[pipeline] All done! Launch the dashboard with:")
        print("    streamlit run app/dashboard.py")

    else:
        print(USAGE)
        sys.exit(1)


if __name__ == "__main__":
    main()
