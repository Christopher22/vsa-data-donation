# SPDX-License-Identifier: MIT
"""
Should I join? A privacy self-assessment for wearable data
==========================================================
A single-file teaching implementation on synthetic Apple Health exports.

The story in one paragraph
--------------------------
A study wants a summary of many people's wearable data. Each person turns
28 days of their Apple Health export into ONE vector (a "record hypervector").
The study adds all vectors up, adds random noise, and publishes the sum.
Because the summary is just a noisy SUM, two things become simple:

  1. SELF-ASSESSMENT.  Before joining, a person can compute on their own
     device how well anyone could ever tell that they took part. The number
     depends on THEIR data (a watch wearer with a regular life stands out
     more than a phone-only user), so it is a personal bound, not an average.
  2. TEST THE BOUND.   We attack the published summary and check that no
     attacker beats the bound, and whether the bound tells us who is really
     most at risk.

How to run (needs only Python >= 3.9 and numpy)
-----------------------------------------------
    python healthtwin_self_assessment.py            # full demo, about 30 seconds
    python healthtwin_self_assessment.py --test     # self-tests
    python healthtwin_self_assessment.py --export path/to/export.xml
                                                    # your own bound; nothing leaves your machine
"""
from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

# ---------------------------------------------------------------------------
# Public study protocol. Everything in this block is published by the study
# BEFORE anyone joins. A person needs nothing else to compute their bound.
# ---------------------------------------------------------------------------
DIM = 1024                # length of every hypervector
CODEBOOK_SEED = 2026      # public seed: everyone must build the same atoms
WINDOW_DAYS = 28          # each person contributes at most this many days
SIGMA = 900.0             # standard deviation of the noise the study adds

# Which Apple Health types the study uses, and the public bin edges that turn
# a daily value into one of six levels (below first edge = level 0, ...).
PROTOCOL = {
    "HKQuantityTypeIdentifierStepCount":               [2500, 5000, 7500, 10000, 12500],
    "HKQuantityTypeIdentifierWalkingSpeed":            [3.5, 4.0, 4.5, 5.0, 5.5],      # km/hr
    "HKQuantityTypeIdentifierRestingHeartRate":        [55, 60, 65, 70, 80],           # count/min
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": [20, 35, 50, 65, 80],          # ms
    "HKQuantityTypeIdentifierVO2Max":                  [30, 35, 40, 45, 50],           # mL/min·kg
    "HKCategoryTypeIdentifierSleepAnalysis":           [5, 6, 7, 8, 9],                # hours asleep
}
PHONE_TYPES = list(PROTOCOL)[:2]      # an iPhone alone records these
SUMMED_TYPES = {"HKQuantityTypeIdentifierStepCount",
                "HKCategoryTypeIdentifierSleepAnalysis"}   # daily total; others: daily mean
N_LEVELS = 6
DATE_FORMAT = "%Y-%m-%d %H:%M:%S %z"  # the format Apple Health uses in export.xml


# ===========================================================================
# Part 1. Synthetic Apple Health exports
# ===========================================================================
# An Apple Health export ("Export All Health Data") is a file export.xml with
# one <Record .../> element per sample. We model a record as a small tuple
# (type, start, end, value) and can write/read the real XML layout.

def synthetic_person(rng: np.random.Generator, days: int = WINDOW_DAYS) -> list[tuple]:
    """Invent one person and return their export as a list of records.

    Three hidden traits make people differ in ways that matter for privacy:
    fitness (shifts all values), regularity (how similar their days are) and
    whether they wear an Apple Watch (how MANY data types they have).
    """
    fitness = rng.normal()
    wobble = rng.lognormal(mean=-0.2, sigma=0.6)      # small = very regular life
    has_watch = rng.random() < 0.5
    first_day = datetime.strptime("2026-09-01 00:00:00 +0100", DATE_FORMAT)
    records = []
    for d in range(days):
        day = first_day + timedelta(days=d)

        def add(kind, value, hour=12, hours_long=0.0):
            start = day + timedelta(hours=hour)
            records.append((kind, start, start + timedelta(hours=hours_long), float(value)))

        # The phone counts steps in several chunks per day; the daily total matters.
        steps_today = max(300.0, 7500 + 2000 * fitness + 2500 * wobble * rng.normal())
        for hour, share in zip((9, 14, 19), rng.dirichlet([2, 2, 2])):
            add("HKQuantityTypeIdentifierStepCount", round(steps_today * share), hour, 1.0)
        add("HKQuantityTypeIdentifierWalkingSpeed",
            4.5 + 0.4 * fitness + 0.3 * wobble * rng.normal())
        if has_watch and rng.random() < 0.9:          # the watch is not worn every day
            add("HKQuantityTypeIdentifierRestingHeartRate",
                64 - 5 * fitness + 3 * wobble * rng.normal(), hour=6)
            add("HKQuantityTypeIdentifierHeartRateVariabilitySDNN",
                45 + 12 * fitness + 10 * wobble * rng.normal(), hour=3)
            add("HKQuantityTypeIdentifierVO2Max",
                40 + 5 * fitness + 1.0 * wobble * rng.normal(), hour=18)
            asleep = float(np.clip(7.2 + 0.2 * fitness + 0.9 * wobble * rng.normal(), 3, 11))
            add("HKCategoryTypeIdentifierSleepAnalysis", 1.0, hour=-1, hours_long=asleep)
    return records


def write_export_xml(records: list[tuple], path: str) -> None:
    """Write records in the layout of a real Apple Health export.xml."""
    root = ET.Element("HealthData", locale="en_CH")
    for kind, start, end, value in records:
        is_sleep = kind.startswith("HKCategoryType")
        ET.SubElement(root, "Record", type=kind, sourceName="Synthetic",
                      startDate=start.strftime(DATE_FORMAT), endDate=end.strftime(DATE_FORMAT),
                      value="HKCategoryValueSleepAnalysisAsleepCore" if is_sleep else repr(value))
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def read_export_xml(path: str) -> list[tuple]:
    """Read the records the study protocol uses from an export.xml (real or synthetic)."""
    records = []
    for _, node in ET.iterparse(path):          # iterparse copes with very large exports
        if node.tag != "Record" or node.get("type") not in PROTOCOL:
            node.clear()
            continue
        kind, raw = node.get("type"), node.get("value", "")
        start = datetime.strptime(node.get("startDate"), DATE_FORMAT)
        end = datetime.strptime(node.get("endDate"), DATE_FORMAT)
        if kind.startswith("HKCategoryType"):
            if "Asleep" in raw:                 # skip "InBed" and "Awake" samples
                records.append((kind, start, end, 1.0))
        else:
            records.append((kind, start, end, float(raw)))
        node.clear()
    return records


def daily_levels(records: list[tuple]) -> dict[str, np.ndarray]:
    """Turn raw records into, per data type, a count of days at each level.

    Example result: {"...StepCount": [0, 3, 12, 9, 4, 0], ...} means 3 days in
    step level 1, 12 days in level 2, and so on. Only the latest WINDOW_DAYS
    days are used, which caps how much any one person can contribute.
    """
    per_day = defaultdict(list)                 # (type, day) -> values of that day
    for kind, start, end, value in records:
        if kind == "HKCategoryTypeIdentifierSleepAnalysis":
            value = (end - start).total_seconds() / 3600.0      # hours asleep
            day = end.date()                    # a night belongs to the morning it ends
        else:
            day = start.date()
        per_day[(kind, day)].append(value)
    if not per_day:
        return {}
    last_day = max(day for _, day in per_day)
    histograms = {}
    for (kind, day), values in per_day.items():
        if (last_day - day).days >= WINDOW_DAYS:
            continue
        daily = sum(values) if kind in SUMMED_TYPES else sum(values) / len(values)
        level = int(np.searchsorted(PROTOCOL[kind], daily, side="right"))
        histograms.setdefault(kind, np.zeros(N_LEVELS, dtype=np.int64))[level] += 1
    return histograms


# ===========================================================================
# Part 2. Encoding one person as one hypervector
# ===========================================================================

def build_codebook(dim: int = DIM, seed: int = CODEBOOK_SEED) -> dict[tuple[str, int], np.ndarray]:
    """One atom per (data type, level): the binding role * level.

    The codebook is PUBLIC. That is deliberate: a secret codebook would only
    be obscurity. All privacy in this file comes from the added noise.
    """
    rng = np.random.default_rng(seed)
    book = {}
    for kind in PROTOCOL:
        role = rng.choice([-1, 1], size=dim)                    # atom for the data type
        for level in range(N_LEVELS):
            value = rng.choice([-1, 1], size=dim)               # atom for the level
            book[(kind, level)] = role * value                  # binding
    return book


def encode_person(histograms: dict[str, np.ndarray], book: dict) -> np.ndarray:
    """Bundle a person's days: x = sum over (type, level) of days * atom."""
    dim = len(next(iter(book.values())))
    x = np.zeros(dim, dtype=np.int64)
    for kind, counts in histograms.items():
        for level, days in enumerate(counts):
            if days:
                x += int(days) * book[(kind, level)]            # bundling
    return x


# ===========================================================================
# Part 3. The study's release
# ===========================================================================

def integer_noise(rng: np.random.Generator, dim: int, sigma: float = SIGMA) -> np.ndarray:
    """Gaussian noise rounded to integers, so the published sum stays integer."""
    return np.rint(rng.normal(0.0, sigma, size=dim)).astype(np.int64)


def release(vectors: np.ndarray, rng: np.random.Generator, sigma: float = SIGMA):
    """Published summary = sum of all record hypervectors + noise."""
    noise = integer_noise(rng, vectors.shape[1], sigma)
    return vectors.sum(axis=0) + noise, noise


def decode_shares(y: np.ndarray, book: dict, kind: str) -> np.ndarray:
    """What the study gets out of it: the share of person-days at each level.

    Unbinding: the dot product with an atom, divided by DIM, estimates how
    often that atom was bundled into y.
    """
    counts = np.array([y @ book[(kind, lv)] / len(y) for lv in range(N_LEVELS)])
    counts = np.clip(counts, 0, None)
    return counts / max(counts.sum(), 1e-9)


# ===========================================================================
# Part 4. The personal bound ("Should I join?")
# ===========================================================================
# Adding or removing person i shifts the release by exactly x_i. With Gaussian
# noise of scale sigma, the release is "mu-Gaussian differentially private"
# for that person with            mu_i = ||x_i|| / sigma.
# (Dong, Roth, Su 2022 for Gaussian DP; Feldman & Zrnic 2021 for the idea of
# an INDIVIDUAL privacy loss.) mu_i is a personal number: only x_i enters it.

def phi(z: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def personal_mu(x: np.ndarray, sigma: float = SIGMA) -> float:
    return float(np.linalg.norm(x)) / sigma


def best_guess_accuracy(mu: float) -> float:
    """The best any attacker can do when guessing "did this person take part?"

    For a fair yes/no guess (half the time the person is in, half not), no
    attacker can be right more often than Phi(mu / 2), even one who knows the
    person's full data AND everybody else's data. 0.5 is blind guessing.
    """
    return phi(mu / 2.0)


def worst_case_mu(n_types: int = len(PROTOCOL), sigma: float = SIGMA, dim: int = DIM) -> float:
    """The study-wide promise: the bound for the most exposed possible person
    (every data type present, every day at the same level)."""
    return math.sqrt(n_types * WINDOW_DAYS ** 2 * dim) / sigma


def lifetime_mu(mus: list[float]) -> float:
    """Ledger rule: releases that contain the person add up as sqrt(sum mu^2)."""
    return math.sqrt(sum(m * m for m in mus))


def in_plain_words(mu: float) -> str:
    right = round(100 * best_guess_accuracy(mu))
    return (f"Someone trying to tell whether you took part would be right at most "
            f"{right} times in 100. Blind guessing is right 50 times in 100.")


# ===========================================================================
# Part 5. Attacks: is the bound true, and is it useful?
# ===========================================================================

def attack_all_knowing(x: np.ndarray, rng: np.random.Generator, trials: int = 500,
                       sigma: float = SIGMA) -> float:
    """The strongest possible attacker: knows x and the exact sum of all others.

    They subtract the others from the release and look at what is left, which
    is either noise (person absent) or x + noise (person present). The best
    test is: project the leftover onto x and compare with ||x||^2 / 2.
    Returns the fraction of correct guesses. Theory says: Phi(mu / 2).
    """
    threshold = float(x @ x) / 2.0
    noise_out = np.rint(rng.normal(0, sigma, size=(trials, len(x))))
    noise_in = np.rint(rng.normal(0, sigma, size=(trials, len(x))))
    says_out_when_out = (noise_out @ x) < threshold
    says_in_when_in = ((noise_in + x) @ x) >= threshold
    return float((says_out_when_out.mean() + says_in_when_in.mean()) / 2.0)


def attack_population(targets: np.ndarray, pool: np.ndarray, n_people: int,
                      rng: np.random.Generator, trials: int = 1000,
                      sigma: float = SIGMA) -> np.ndarray:
    """A realistic attacker: knows the target's data and what the population
    looks like in general, but NOT who else is in the study.

    This is the classic test of Homer et al. (2008): compare the release with
    what an average cohort would give, in the direction in which the target
    differs from the average person. The threshold is calibrated on simulated
    ("shadow") studies and then scored on fresh ones, as in LiRA (Carlini et
    al. 2022). Returns one accuracy per target.
    """
    mean = pool.mean(axis=0)
    direction = targets - mean                               # how each target differs

    def simulate(target_is_in: bool) -> np.ndarray:
        others = n_people - 1 if target_is_in else n_people
        picks = rng.multinomial(others, np.full(len(pool), 1 / len(pool)), size=trials)
        cohort_sum = picks @ pool                            # trials x DIM
        noisy = cohort_sum + np.rint(rng.normal(0, sigma, size=cohort_sum.shape))
        scores = (noisy - n_people * mean) @ direction.T     # trials x targets
        if target_is_in:                                     # add each target to "their" column
            scores = scores + np.einsum("td,td->t", targets, direction)
        return scores

    shadow_in, shadow_out = simulate(True), simulate(False)
    threshold = (shadow_in.mean(axis=0) + shadow_out.mean(axis=0)) / 2.0
    test_in, test_out = simulate(True), simulate(False)
    return ((test_in >= threshold).mean(axis=0) + (test_out < threshold).mean(axis=0)) / 2.0


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation: do two lists put people in the same order?"""
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


# ===========================================================================
# Part 6. The demo
# ===========================================================================

def make_population(n: int, seed: int, book: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return n record hypervectors and, for each person, whether they wear a watch."""
    rng = np.random.default_rng(seed)
    histograms = [daily_levels(synthetic_person(rng)) for _ in range(n)]
    vectors = np.stack([encode_person(h, book) for h in histograms])
    has_watch = np.array([len(h) > len(PHONE_TYPES) for h in histograms])
    return vectors, has_watch


def demo() -> None:
    rng = np.random.default_rng(7)
    book = build_codebook()
    line = "-" * 78

    print(line + "\nSTEP 0  One synthetic export, written and read like a real export.xml")
    alex_records = synthetic_person(np.random.default_rng(1))
    write_export_xml(alex_records, "synthetic_export.xml")
    alex_hist = daily_levels(read_export_xml("synthetic_export.xml"))
    for kind, counts in alex_hist.items():
        print(f"  {kind.split('Identifier')[1]:<26} days per level: {counts.tolist()}")

    print(line + "\nSTEP 1  SELF-ASSESSMENT  (computed from the person's own export only)")
    alex = encode_person(alex_hist, book)
    mu_alex = personal_mu(alex)
    print(f"  Alex's personal mu = {mu_alex:.2f}.  {in_plain_words(mu_alex)}")
    print(f"  The study's promise for the most exposed possible person: at most "
          f"{round(100 * best_guess_accuracy(worst_case_mu()))} in 100.")

    n_people, n_targets = 300, 150
    pool, pool_watch = make_population(2000, seed=11, book=book)   # "the population"
    watch = pool_watch[:n_targets]
    targets, others_pool = pool[:n_targets], pool[n_targets:]
    mus = np.array([personal_mu(x) for x in targets])
    bound = np.array([best_guess_accuracy(m) for m in mus])
    print(f"  Across {n_targets} people the personal bound ranges from "
          f"{100 * bound.min():.0f} to {100 * bound.max():.0f} in 100 "
          f"(median {100 * np.median(bound):.0f}).")

    print(line + "\nSTEP 2  TEST THE BOUND with two attackers")
    trials = 500
    strongest = np.array([attack_all_knowing(x, rng, trials) for x in targets])
    standard_error = np.sqrt(bound * (1 - bound) / (2 * trials))
    violations = int(((strongest - bound) / standard_error > 3).sum())
    print(f"  All-knowing attacker: right {100 * strongest.mean():.1f} in 100 on average; "
          f"the bound said {100 * bound.mean():.1f}.")
    print(f"  People for whom the attacker beat the bound (beyond chance, 3 SE): "
          f"{violations} of {n_targets}   <- the claim fails if this is not 0")
    realistic = attack_population(targets, others_pool, n_people, rng)
    print(f"  Realistic attacker (knows the population, not the other members): "
          f"right {100 * realistic.mean():.1f} in 100 on average, never above "
          f"{100 * realistic.max():.0f}.")
    print(f"  Does the bound rank people by real risk? Spearman correlation "
          f"bound vs realistic attack: {rank_correlation(mus, realistic):.2f}")
    print(f"  Caution: much of that is the watch. Within phone-only people the correlation is "
          f"{rank_correlation(mus[~watch], realistic[~watch]):.2f}, within watch wearers "
          f"{rank_correlation(mus[watch], realistic[watch]):.2f}.")
    print(f"  Bounds of phone-only people: {100 * bound[~watch].min():.0f} to {100 * bound[~watch].max():.0f} in 100; "
          f"of watch wearers: {100 * bound[watch].min():.0f} to {100 * bound[watch].max():.0f}.")

    print(line + "\nSTEP 3  WHAT PRIVACY COSTS IN USEFULNESS")
    cohort = pool[:n_people]
    steps = "HKQuantityTypeIdentifierStepCount"
    exact = decode_shares(cohort.sum(axis=0), book, steps)
    print("  noise sigma | typical person: right at most | error of step-level shares")
    for sigma in (0, 300, 900, 2700, 8100):
        errors = [np.abs(decode_shares(release(cohort, rng, sigma)[0], book, steps) - exact).mean()
                  for _ in range(20)]
        guess = 100 * best_guess_accuracy(float(np.median(mus)) * SIGMA / sigma) if sigma else 100
        print(f"  {sigma:>11} | {guess:>18.0f} in 100       | {100 * np.mean(errors):.2f} percentage points")

    print(line)


# ===========================================================================
# Part 7. Self-tests  (python healthtwin_self_assessment.py --test)
# ===========================================================================

def run_tests() -> None:
    rng = np.random.default_rng(0)
    book = build_codebook()

    # 1. XML round trip keeps the daily summaries.
    records = synthetic_person(np.random.default_rng(3))
    write_export_xml(records, "test_export.xml")
    original, reread = daily_levels(records), daily_levels(read_export_xml("test_export.xml"))
    assert original.keys() == reread.keys()
    assert all(np.array_equal(original[k], reread[k]) for k in original), "XML round trip"

    # 2. Nobody contributes more days than the window allows.
    assert all(h.sum() <= WINDOW_DAYS for h in original.values()), "window cap"

    # 3. The encoding is linear and decodable: unbinding recovers the day counts.
    x = encode_person(original, book)
    steps = "HKQuantityTypeIdentifierStepCount"
    decoded = [x @ book[(steps, lv)] / DIM for lv in range(N_LEVELS)]
    assert np.abs(np.array(decoded) - original[steps]).max() < 6, "decode"

    # 4. No person can exceed the study's worst-case promise.
    people, _ = make_population(200, seed=5, book=book)
    assert max(personal_mu(p) for p in people) <= worst_case_mu(), "worst case"

    # 5. Bound formulas.
    assert abs(best_guess_accuracy(0.0) - 0.5) < 1e-12
    assert abs(lifetime_mu([0.3, 0.4]) - 0.5) < 1e-12

    # 6. The strongest attacker matches the bound and does not beat it.
    mu = personal_mu(people[0])
    accuracy = attack_all_knowing(people[0], rng, trials=4000)
    slack = 4 * math.sqrt(0.25 / 8000)
    assert abs(accuracy - best_guess_accuracy(mu)) < slack, "bound is tight and holds"

    # 7. The published sum is still useful: decoded step shares stay close to the truth.
    exact = decode_shares(people.sum(axis=0), book, steps)
    noisy = decode_shares(release(people, rng)[0], book, steps)
    assert np.abs(exact - noisy).max() < 0.03, "utility"
    print("All 7 tests passed.")


def own_export(path: str) -> None:
    histograms = daily_levels(read_export_xml(path))
    if not histograms:
        sys.exit("No records of the study's data types were found in this export.")
    mu = personal_mu(encode_person(histograms, build_codebook()))
    print(f"Data types found: {len(histograms)}.  Your personal mu = {mu:.2f}.")
    print(in_plain_words(mu))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--test", action="store_true", help="run the self-tests")
    parser.add_argument("--export", metavar="XML", help="compute the bound for your own export.xml")
    args = parser.parse_args()
    if args.test:
        run_tests()
    elif args.export:
        own_export(args.export)
    else:
        demo()
