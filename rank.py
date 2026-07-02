import numpy as np
import argparse
import csv
import json
import math
import sys
import time
from datetime import date, datetime

W_SEMANTIC = 0.28
W_TRAJECTORY = 0.34
W_EXPERIENCE = 0.12
W_SKILL = 0.14
W_EDUCATION = 0.06
W_LOCATION = 0.06

assert abs((W_SEMANTIC + W_TRAJECTORY + W_EXPERIENCE + W_SKILL
            + W_EDUCATION + W_LOCATION) - 1.0) < 1e-9
assert W_TRAJECTORY >= 2 * W_SKILL
assert (W_EDUCATION + W_LOCATION) < W_TRAJECTORY
assert (W_EDUCATION + W_LOCATION) < W_SEMANTIC

EXP_BAND_LO, EXP_BAND_HI = 5.0, 9.0
PREFERRED_LOCATIONS = {"pune", "noida", "hyderabad", "mumbai", "delhi", "delhi ncr",
                       "gurgaon", "gurugram", "new delhi"}
SERVICES_FIRMS = {"tcs", "tata consultancy services", "infosys", "wipro", "accenture",
                  "cognizant", "capgemini", "hcl", "tech mahindra", "mindtree",
                  "ltimindtree", "mphasis"}

JD_CORE_TERMS = [
    "embedding", "embeddings", "retrieval", "ranking", "rank", "search",
    "recommendation", "recommender", "vector", "semantic search", "information retrieval",
    "nlp", "natural language", "machine learning", "deep learning", "ml", "llm",
    "fine-tuning", "fine tuning", "transformer", "bert", "sentence-transformers",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch", "opensearch",
    "learning to rank", "ndcg", "mrr", "relevance", "personalization", "matching",
    "bm25", "hybrid search", "rag", "knn", "ann", "feature", "model serving",
]
ENGINEER_DOMAIN_TERMS = [
    "engineer", "developer", "scientist", "ml", "machine learning", "ai", "data",
    "software", "backend", "search", "research engineer", "applied scientist",
    "platform", "infrastructure",
]
NON_ENGINEER_TITLES = [
    "marketing", "operations manager", "accountant", "hr ", "human resources",
    "sales", "customer support", "content writer", "graphic designer",
    "business analyst", "project manager", "civil engineer", "mechanical engineer",
    "operations",
]
SENIORITY_RANK = [
    ("intern", 0), ("junior", 1), ("associate", 1), ("", 2),
    ("senior", 3), ("staff", 4), ("lead", 4), ("principal", 5),
    ("head", 6), ("director", 6), ("vp", 7), ("chief", 8),
]
VISION_SPEECH_ROBOTICS = ["computer vision", "image classification", "object detection",
                          "speech recognition", "tts", "asr", "robotics", "slam"]
NLP_IR_TERMS = ["nlp", "natural language", "retrieval", "search", "ranking",
                "recommendation", "information retrieval", "embedding", "text"]

def parse_date(s):
    if not s or not isinstance(s, str):
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def months_between(d1, d2):
    if d1 is None or d2 is None:
        return None
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)

class Candidate:
    __slots__ = ("cid", "text", "yoe", "resolved_yoe", "titles", "current_title",
                 "skills", "skill_assess", "location", "willing_relocate",
                 "edu_fields", "edu_tiers", "career", "signals", "is_honeypot",
                 "honeypot_reasons")

    def __init__(self, rec):
        self.cid = rec.get("candidate_id", "")
        prof = rec.get("profile", {}) or {}
        self.current_title = (prof.get("current_title") or "").strip()
        self.location = (prof.get("location") or "").strip()
        try:
            self.yoe = float(prof.get("years_of_experience"))
        except (TypeError, ValueError):
            self.yoe = None

        career = rec.get("career_history", []) or []
        self.career = career
        self.titles = [(c.get("title") or "").strip() for c in career]
        if self.current_title:
            self.titles = [self.current_title] + self.titles

        parts = [prof.get("headline") or "", prof.get("summary") or ""]
        for c in career:
            parts.append(c.get("title") or "")
            parts.append(c.get("description") or "")
        self.text = " ".join(parts).strip()

        skills = rec.get("skills", []) or []
        self.skills = skills

        sig = rec.get("redrob_signals", {}) or {}
        self.signals = sig
        self.skill_assess = sig.get("skill_assessment_scores", {}) or {}
        self.willing_relocate = bool(sig.get("willing_to_relocate", False))

        edu = rec.get("education", []) or []
        self.edu_fields = [(e.get("field_of_study") or "").lower() for e in edu]
        self.edu_tiers = [(e.get("tier") or "unknown") for e in edu]

        summed = 0
        have_dur = False
        for c in career:
            dm = c.get("duration_months")
            if isinstance(dm, (int, float)):
                summed += dm
                have_dur = True
        summed_years = summed / 12.0 if have_dur else None
        if self.yoe is None:
            self.resolved_yoe = summed_years
        elif summed_years is not None and abs(self.yoe - summed_years) > 2.0:
            self.resolved_yoe = summed_years
        else:
            self.resolved_yoe = self.yoe

        self.is_honeypot = False
        self.honeypot_reasons = []

def detect_honeypot(c):
    reasons = []
    ryoe_months = (c.resolved_yoe or 0) * 12

    for s in c.skills:
        prof = (s.get("proficiency") or "").lower()
        dur = s.get("duration_months")
        if prof in ("advanced", "expert") and (dur is None or dur == 0):
            reasons.append(f"{s.get('name')}: {prof} with 0 months used")
        if isinstance(dur, (int, float)) and ryoe_months and dur > ryoe_months + 12:
            reasons.append(f"{s.get('name')}: used {dur}mo > career {ryoe_months:.0f}mo")

    for entry in c.career:
        sd = parse_date(entry.get("start_date"))
        ed = parse_date(entry.get("end_date")) or date(2025, 6, 1)
        dm = entry.get("duration_months")
        if sd and ed and sd > ed:
            reasons.append(f"{entry.get('company')}: start after end")
        span = months_between(sd, ed)
        if span is not None and isinstance(dm, (int, float)):
            if dm - span > 12:  
                reasons.append(f"{entry.get('company')}: {dm}mo claimed vs {span}mo span")

    if c.yoe is not None:
        summed = sum(e.get("duration_months", 0) for e in c.career
                     if isinstance(e.get("duration_months"), (int, float)))
        if summed and abs(summed / 12.0 - c.yoe) >= 1.0 and (summed / 12.0 - c.yoe) > 3.0:
            reasons.append(f"career sum {summed/12.0:.1f}y vs stated {c.yoe:.1f}y")

    if reasons:
        c.is_honeypot = True
        c.honeypot_reasons = reasons
    return c.is_honeypot

def title_domain_match(title):
    t = title.lower()
    if not t:
        return False
    for bad in NON_ENGINEER_TITLES:
        if bad in t:
            if any(g in t for g in ["ml engineer", "machine learning", "ai engineer",
                                    "data scientist", "data engineer", "software",
                                    "search", "nlp", "research engineer",
                                    "applied scientist", "backend"]):
                return True
            return False
    return any(g in t for g in ["engineer", "scientist", "developer", "ml", "ai",
                                "data", "software", "search", "researcher",
                                "architect", "analyst"])


def seniority_of(title):
    t = title.lower()
    best = 2
    for kw, rk in SENIORITY_RANK:
        if kw and kw in t:
            best = max(best, rk)
    return best


def score_trajectory(c):
    titles = [t for t in c.titles if t]
    if not titles:
        return 0.0
    matched = [t for t in titles if title_domain_match(t)]
    recent = titles[:max(1, min(3, len(titles)))]
    recent_offdomain = all(not title_domain_match(t) for t in recent)
    if recent_offdomain:
        return min(0.20, 0.05 * len(matched))
    frac = len(matched) / len(titles)
    base = 0.5 * frac + 0.5 * (1.0 if title_domain_match(titles[0]) else 0.0)
    dom_titles = [t for t in titles if title_domain_match(t)]
    progression = False
    if len(dom_titles) >= 2:
        sr = [seniority_of(t) for t in dom_titles]
        if sr[0] >= max(sr[1:]) and sr[0] > min(sr):
            progression = True
    score = base
    if progression:
        score = max(score, 0.70)
    return round(min(1.0, max(0.0, score)), 4)


def score_experience(c):
    y = c.resolved_yoe
    if y is None:
        return 0.0
    if EXP_BAND_LO <= y <= EXP_BAND_HI:
        return 1.0
    dist = (EXP_BAND_LO - y) if y < EXP_BAND_LO else (y - EXP_BAND_HI)
    if dist >= 10:
        return 0.0
    return round(max(0.0, 1.0 - dist / 10.0) * 0.999, 4)


def skill_is_relevant(name):
    n = name.lower()
    return any(term in n for term in [
        "ml", "machine learning", "deep learning", "nlp", "natural language",
        "retrieval", "ranking", "rank", "search", "recommendation", "embedding",
        "vector", "llm", "fine-tun", "transformer", "bert", "faiss", "pinecone",
        "weaviate", "qdrant", "milvus", "elasticsearch", "opensearch", "rag",
        "pytorch", "tensorflow", "python", "information retrieval", "relevance",
    ])


def score_skill_credibility(c):
    rel = [s for s in c.skills if skill_is_relevant(s.get("name", ""))]
    if not rel:
        return 0.0
    total = 0.0
    for s in rel:
        prof = (s.get("proficiency") or "").lower()
        dur = s.get("duration_months") or 0
        end = s.get("endorsements") or 0
        if prof in ("advanced", "expert") and dur == 0:
            continue
        assess = c.skill_assess.get(s.get("name"))
        cred = 0.0
        cred += min(1.0, dur / 36.0) * 0.4        
        cred += min(1.0, end / 30.0) * 0.3        
        if isinstance(assess, (int, float)):
            cred += (assess / 100.0) * 0.3
        else:
            cred += {"beginner": 0.05, "intermediate": 0.12,
                     "advanced": 0.2, "expert": 0.25}.get(prof, 0.05)
        total += cred
    
    return round(1.0 - math.exp(-total / 2.0), 4)


def score_education(c):
    if not c.edu_fields and not c.edu_tiers:
        return 0.0
    field_rel = 0.0
    for f in c.edu_fields:
        if any(k in f for k in ["computer", "data", "machine learning", "artificial",
                                "statistics", "mathematics", "electrical", "information"]):
            field_rel = max(field_rel, 1.0)
        elif f:
            field_rel = max(field_rel, 0.4)
    tier_val = 0.0
    tier_map = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.5, "tier_4": 0.3,
                "unknown": 0.4}
    for t in c.edu_tiers:
        tier_val = max(tier_val, tier_map.get(t, 0.4))
    return round(0.6 * field_rel + 0.4 * tier_val, 4)


def score_location(c):
    loc = c.location.lower()
    if loc and any(p in loc for p in PREFERRED_LOCATIONS):
        return 1.0
    if not loc:
        return 0.55 if c.willing_relocate else 0.15
    return 0.55 if c.willing_relocate else 0.15

def disqualifier_penalty(c, dataset_max_active):
    pen = 0.0
    reasons = []
    companies = [(e.get("company") or "").lower() for e in c.career]
    text_l = c.text.lower()

    if companies and all(any(f in comp for f in SERVICES_FIRMS) for comp in companies):
        pen += 0.25
        reasons.append("services-only career")

    has_vsr = any(term in text_l for term in VISION_SPEECH_ROBOTICS) or \
        any(any(term in (s.get("name", "").lower()) for term in VISION_SPEECH_ROBOTICS)
            for s in c.skills)
    has_nlp_ir = any(term in text_l for term in NLP_IR_TERMS) or \
        any(any(term in (s.get("name", "").lower()) for term in NLP_IR_TERMS)
            for s in c.skills)
    if has_vsr and not has_nlp_ir:
        pen += 0.20
        reasons.append("CV/speech/robotics without NLP/IR")

    durs = [e.get("duration_months") for e in c.career
            if isinstance(e.get("duration_months"), (int, float))]
    if len(durs) >= 4 and (sum(durs[:4]) / 4.0) < 18:
        pen += 0.15
        reasons.append("frequent job changes (<18mo avg)")

    if dataset_max_active is not None:
        has_current = any(e.get("is_current") for e in c.career)
        if not has_current:
            latest_end = None
            for e in c.career:
                ed = parse_date(e.get("end_date"))
                if ed and (latest_end is None or ed > latest_end):
                    latest_end = ed
            if latest_end is not None:
                gap = months_between(latest_end, dataset_max_active)
                if gap is not None and gap > 18:
                    pen += 0.15
                    reasons.append("no recent production activity")

    return min(pen, 1.0), reasons

def behavioral_multiplier(c, dataset_max_active):
    sig = c.signals
    comps = []

    rr = sig.get("recruiter_response_rate")
    if isinstance(rr, (int, float)) and rr >= 0:
        comps.append(max(0.0, min(1.0, rr)))

    ic = sig.get("interview_completion_rate")
    if isinstance(ic, (int, float)) and ic >= 0:
        comps.append(max(0.0, min(1.0, ic)))

    otw = sig.get("open_to_work_flag")
    if isinstance(otw, bool):
        comps.append(1.0 if otw else 0.3)

    la = parse_date(sig.get("last_active_date"))
    if la is not None and dataset_max_active is not None:
        gap_days = (dataset_max_active - la).days
        if gap_days <= 30:
            comps.append(1.0)
        elif gap_days >= 180:
            comps.append(0.0)
        else:
            comps.append(1.0 - (gap_days - 30) / 150.0)

    if not comps:
        return 1.0
    b = sum(comps) / len(comps)
    return 0.5 + 0.5 * b 

def make_reasoning(c, subs, pen_reasons):
    yoe = c.resolved_yoe
    title = c.current_title or (c.titles[0] if c.titles else "Unknown role")
    rel_skills = [s.get("name") for s in c.skills
                  if skill_is_relevant(s.get("name", ""))][:3]
    rr = c.signals.get("recruiter_response_rate")

    facts = []
    if yoe is not None:
        facts.append(f"{title} with {yoe:.1f} yrs")
    else:
        facts.append(f"{title}")
    if rel_skills:
        facts.append("relevant skills: " + ", ".join(rel_skills))
    if isinstance(rr, (int, float)) and rr >= 0:
        facts.append(f"recruiter response {rr:.0%}")

    if subs["trajectory"] >= 0.7:
        jd = "strong product-engineering/ML trajectory matching the AI-Engineer role"
    elif subs["semantic"] >= 0.4:
        jd = "profile describes retrieval/ranking/ML work the JD asks for"
    elif subs["trajectory"] <= 0.2:
        jd = "off-domain title relative to the AI-Engineer JD"
    else:
        jd = "partial alignment with the JD's retrieval/ranking focus"

    concerns = []
    if yoe is not None and not (EXP_BAND_LO <= yoe <= EXP_BAND_HI):
        concerns.append(f"experience {yoe:.1f}y outside the 5-9y band")
    if pen_reasons:
        concerns.append(pen_reasons[0])
    if c.is_honeypot:
        concerns.append("profile inconsistencies detected")

    s = "; ".join(facts) + ". " + jd.capitalize() + "."
    if concerns:
        s += " Concern: " + "; ".join(concerns[:2]) + "."
    return s[:300]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=100)
    args = ap.parse_args()

    t0 = time.time()

    cands = []
    excluded = 0
    n_lines = 0
    max_active = None
    try:
        f = open(args.candidates, "r", encoding="utf-8")
    except OSError as e:
        print(f"ERROR: cannot read candidates file: {e}", file=sys.stderr)
        sys.exit(2)
    with f:
        for line in f:
            if not line.strip():
                continue
            n_lines += 1
            try:
                rec = json.loads(line)
                c = Candidate(rec)
                if not c.cid:
                    excluded += 1
                    continue
                cands.append(c)
                la = parse_date(c.signals.get("last_active_date"))
                if la and (max_active is None or la > max_active):
                    max_active = la
            except Exception:
                excluded += 1
                continue

    print(f"[load] lines={n_lines} parsed={len(cands)} excluded={excluded} "
          f"({time.time()-t0:.1f}s)", file=sys.stderr)
    if len(cands) < args.top:
        print(f"ERROR: only {len(cands)} candidates; need {args.top}", file=sys.stderr)
        sys.exit(3)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    jd_text = (
        "Senior AI Engineer founding team. Production experience with embeddings "
        "based retrieval systems sentence-transformers BGE E5. Vector databases "
        "hybrid search FAISS Pinecone Weaviate Qdrant Milvus OpenSearch Elasticsearch. "
        "Strong Python. Evaluation frameworks for ranking systems NDCG MRR MAP "
        "offline online A/B testing. Applied machine learning at product companies. "
        "Built shipped end-to-end ranking search recommendation system at scale. "
        "NLP information retrieval relevance learning to rank LLM fine-tuning."
    )
    docs = [c.text if c.text else " " for c in cands]
    vec = TfidfVectorizer(max_features=40000, stop_words="english",
                          ngram_range=(1, 2), min_df=2, dtype=np.float32)
    X = vec.fit_transform(docs)
    jd_vec = vec.transform([jd_text])
    sims = linear_kernel(jd_vec, X).ravel()
    if sims.max() > 0:
        sem_scores = sims / sims.max()
    else:
        sem_scores = sims
    print(f"[tfidf] matrix={X.shape} ({time.time()-t0:.1f}s)", file=sys.stderr)

    results = []
    for i, c in enumerate(cands):
        detect_honeypot(c)
        subs = {
            "semantic": round(float(sem_scores[i]), 4) if c.text.strip() else 0.0,
            "trajectory": score_trajectory(c),
            "experience": score_experience(c),
            "skill": score_skill_credibility(c),
            "education": score_education(c),
            "location": score_location(c),
        }
        fit = (W_SEMANTIC * subs["semantic"] + W_TRAJECTORY * subs["trajectory"]
               + W_EXPERIENCE * subs["experience"] + W_SKILL * subs["skill"]
               + W_EDUCATION * subs["education"] + W_LOCATION * subs["location"])
        pen, pen_reasons = disqualifier_penalty(c, max_active)
        fit = max(0.0, fit - pen)
        mult = behavioral_multiplier(c, max_active)
        composite = max(0.0, min(1.0, fit * mult))
        if c.is_honeypot:
            composite = composite * 0.0005
        score4 = round(composite, 4)
        results.append((score4, c.cid, c, subs, pen_reasons))
    results.sort(key=lambda r: (-r[0], r[1]))
    top = results[:args.top]

    with open(args.out, "w", encoding="utf-8", newline="") as out:
        w = csv.writer(out)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (score4, cid, c, subs, pen_reasons) in enumerate(top, start=1):
            reasoning = make_reasoning(c, subs, pen_reasons)
            w.writerow([cid, rank, f"{score4:.4f}", reasoning])

    hp_in_top = sum(1 for r in top if r[2].is_honeypot)
    print(f"[done] wrote {len(top)} rows; honeypots_in_top={hp_in_top}; "
          f"total {time.time()-t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
