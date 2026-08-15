import os
import time

import httpx

from simulator.scenarios import ScenarioKind, build_scenario


def run() -> None:
    api_url = os.getenv("API_URL", "http://api:8000/api/v1")
    interval = int(os.getenv("SIM_INTERVAL_MS", "2000")) / 1000
    seed = int(os.getenv("SIM_SEED", "1000"))
    mix = [ScenarioKind(item.strip()) for item in os.getenv("SIM_SCENARIO_MIX", "NORMAL,NORMAL,DEFECT,REVIEW").split(",")]
    with httpx.Client(timeout=5) as client:
        while True:
            scenario = mix[seed % len(mix)]
            response = client.post(f"{api_url}/inspections", json=build_scenario(seed, scenario))
            response.raise_for_status()
            seed += 1
            time.sleep(interval)


if __name__ == "__main__":
    if os.getenv("SIM_MODE", "continuous").lower() == "e2e":
        from simulator.e2e import main

        raise SystemExit(main())
    run()
