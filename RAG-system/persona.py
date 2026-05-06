"""
Persona Extraction (Part 2)

Extracts structured user persona from conversation signals using:
- Pattern matching (regex) for concrete facts
- Frequency analysis for habits and recurring themes
- Linguistic analysis for communication style
- Sentiment/tone analysis for personality traits

Output: JSON with habits, personal_facts, personality_traits, communication_style
"""

import json
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

DATA_DIR = Path(__file__).parent.parent / "data"
PERSONA_FILE = DATA_DIR / "persona.json"


# ── Signal patterns ───────────────────────────────────────────────────────────

HABIT_PATTERNS = [
    (r"\b(wake up|get up)\b.{0,40}\b(morning|early|late|[0-9]+\s*am|[0-9]+\s*pm)\b",
     "sleep_schedule"),
    (r"\b(go to bed|sleep|stay up)\b.{0,40}\b(late|night|early|[0-9]+\s*am|[0-9]+\s*pm)\b",
     "sleep_schedule"),
    (r"\b(every day|daily|every morning|every night|every week|always)\b.{0,60}",
     "routine"),
    (r"\b(coffee|tea|breakfast|lunch|dinner|eat|food|cook|meal)\b.{0,60}",
     "food_habits"),
    (r"\b(workout|exercise|gym|run|jog|yoga|walk|hike|swim)\b.{0,60}",
     "fitness_habits"),
    (r"\b(read|reading|books?)\b.{0,60}", "reading_habits"),
    (r"\b(watch|watching|tv|show|series|movie|netflix)\b.{0,60}", "media_habits"),
    (r"\b(smoke|smoking|drink|drinking|alcohol|vape)\b.{0,60}", "substance_habits"),
]

FACT_PATTERNS = [
    (r"\bi(?:'m| am) (?:a |an )?(\w[\w\s]{2,30})", "occupation_mentions"),
    (r"\bi work (?:as |at |for )?(.{5,40})", "work_mentions"),
    (r"\bi(?:'m| am) (?:from |based in |living in |in )([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
     "location_mentions"),
    (r"\bi(?:'m| am) (\d{1,2}) years? old", "age_mentions"),
    (r"\bmy (wife|husband|girlfriend|boyfriend|partner|spouse)\b", "relationship_status"),
    (r"\bmy (son|daughter|kid|child|baby)\b", "has_children"),
    (r"\bmy (dog|cat|pet|puppy|kitten)\b", "has_pets"),
    (r"\bi(?:'m| am) (?:married|single|divorced|widowed)\b", "relationship_status"),
    (r"\bi(?:'m| am) (?:a )?(?:student|studying|in school|in college|in university)\b",
     "education_status"),
    (r"\bi(?:'m| am) (?:from|born in|grew up in) (.{3,30})", "origin"),
    (r"\bmy (?:favorite|favourite) (?:\w+ )?is (.{3,40})", "preferences"),
    (r"\bi love (.{3,40})", "interests"),
    (r"\bi like (.{3,40})", "interests"),
    (r"\bi enjoy (.{3,40})", "interests"),
    (r"\bi hate (.{3,40})", "dislikes"),
    (r"\bi don't like (.{3,40})", "dislikes"),
]

PERSONALITY_SIGNALS = {
    "humor": [
        r"\blol\b", r"\bhaha\b", r"\blmao\b", r"\b😂\b", r"\b😄\b",
        r"\bjoke\b", r"\bfunny\b", r"\blaugh\b", r"\bhumor\b"
    ],
    "empathetic": [
        r"\bi(?:'m| am) sorry\b", r"\bthat(?:'s| is) (hard|tough|rough|sad)\b",
        r"\bi understand\b", r"\bi feel (you|that)\b", r"\bhugs?\b",
        r"\b❤️\b", r"\b🙏\b"
    ],
    "curious": [
        r"\b(really|that's)?\s*(interesting|cool|fascinating|awesome)\b",
        r"\bwonder\b", r"\bwhy\b.*\?", r"\bhow\b.*\?",
        r"\bwhat do you think\b", r"\bhave you (ever|tried)\b"
    ],
    "positive_outlook": [
        r"\blove (it|that|this)\b", r"\b(amazing|wonderful|great|excellent|fantastic)\b",
        r"\b(happy|excited|thrilled|joy|enjoy)\b", r"\bcan't wait\b", r"\blooking forward\b"
    ],
    "introverted": [
        r"\bstay (home|in|inside)\b", r"\balone\b", r"\bquiet\b",
        r"\bi prefer\b.{0,20}\balone\b", r"\bshy\b"
    ],
    "extroverted": [
        r"\bparty\b", r"\bpeople\b.{0,20}\benergy\b", r"\bsocial\b",
        r"\bmeet (new )?people\b", r"\bhanging out\b"
    ],
    "anxious_or_stressed": [
        r"\bstress(ed|ful)?\b", r"\banxi(ous|ety)\b", r"\bworr(y|ied)\b",
        r"\boverwhelm(ed|ing)\b", r"\bcan't sleep\b"
    ],
    "ambitious": [
        r"\bgoal\b", r"\bdream\b", r"\bpursuing\b", r"\bworking (hard|towards)\b",
        r"\bcareer\b", r"\bsucceed\b", r"\bachiev\b"
    ],
}

EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F9FF"
    "\u2600-\u27BF"
    "]+",
    flags=re.UNICODE
)


class PersonaExtractor:
    def __init__(self, messages):
        self.messages = messages

    def _get_user1_texts(self) -> List[str]:
        return [m.text for m in self.messages if m.speaker.strip() == "User 1"]

    def _extract_habits(self, texts: List[str]) -> Dict[str, List[str]]:
        habits: Dict[str, List[str]] = {}
        for text in texts:
            low = text.lower()
            for pattern, label in HABIT_PATTERNS:
                m = re.search(pattern, low)
                if m:
                    snippet = m.group(0).strip()
                    if label not in habits:
                        habits[label] = []
                    if len(habits[label]) < 5 and snippet not in habits[label]:
                        habits[label].append(snippet)
        return habits

    def _extract_facts(self, texts: List[str]) -> Dict[str, List[str]]:
        facts: Dict[str, List[str]] = {}
        for text in texts:
            low = text.lower()
            for pattern, label in FACT_PATTERNS:
                m = re.search(pattern, low)
                if m:
                    snippet = (m.group(1) if m.lastindex else m.group(0)).strip()
                    snippet = snippet.rstrip(".,!?").strip()
                    if len(snippet) > 2:
                        if label not in facts:
                            facts[label] = []
                        if len(facts[label]) < 8 and snippet not in facts[label]:
                            facts[label].append(snippet)
        return facts

    def _extract_personality(self, texts: List[str]) -> Dict[str, Any]:
        scores: Dict[str, int] = {trait: 0 for trait in PERSONALITY_SIGNALS}
        for text in texts:
            low = text.lower()
            for trait, patterns in PERSONALITY_SIGNALS.items():
                for p in patterns:
                    if re.search(p, low):
                        scores[trait] += 1

        total = max(sum(scores.values()), 1)
        # Normalize and keep traits with meaningful signal
        traits = {
            trait: round(count / total * 100, 1)
            for trait, count in scores.items()
            if count > 2
        }
        # Sort by strength
        traits = dict(sorted(traits.items(), key=lambda x: -x[1]))
        # Top-5 dominant traits
        top_traits = list(traits.keys())[:5]
        return {
            "dominant_traits": top_traits,
            "trait_scores_pct": {k: traits[k] for k in top_traits},
        }

    def _extract_communication_style(self, texts: List[str]) -> Dict[str, Any]:
        if not texts:
            return {}

        msg_lengths = [len(t.split()) for t in texts]
        avg_words = round(sum(msg_lengths) / len(msg_lengths), 1)

        emoji_count = sum(len(EMOJI_RE.findall(t)) for t in texts)
        emoji_rate = round(emoji_count / len(texts) * 100, 1)

        question_count = sum(1 for t in texts if "?" in t)
        question_rate = round(question_count / len(texts) * 100, 1)

        exclamation_count = sum(1 for t in texts if "!" in t)
        exclamation_rate = round(exclamation_count / len(texts) * 100, 1)

        # Capitalization style
        all_caps_msgs = sum(1 for t in texts if t.isupper() and len(t) > 3)
        all_lower_msgs = sum(1 for t in texts if t.islower() and len(t) > 3)

        length_style = (
            "very brief" if avg_words < 5
            else "short" if avg_words < 10
            else "medium" if avg_words < 20
            else "detailed"
        )

        # Most common openers
        openers = Counter()
        for t in texts:
            first_word = t.split()[0].lower().rstrip("!?,") if t.split() else ""
            if first_word:
                openers[first_word] += 1
        top_openers = [w for w, _ in openers.most_common(5)]

        return {
            "avg_message_length_words": avg_words,
            "message_length_style": length_style,
            "emoji_usage_pct": emoji_rate,
            "uses_emojis": emoji_rate > 5,
            "question_rate_pct": question_rate,
            "exclamation_rate_pct": exclamation_rate,
            "frequently_uses_caps": all_caps_msgs > len(texts) * 0.1,
            "common_openers": top_openers,
        }

    def _infer_interests(self, texts: List[str]) -> List[str]:
        """Frequency-based interest inference from content keywords."""
        interest_keywords = {
            "music": ["music", "song", "band", "guitar", "piano", "concert", "album", "singing"],
            "sports": ["sport", "soccer", "football", "basketball", "tennis", "gym", "athlete"],
            "cooking": ["cook", "recipe", "food", "meal", "kitchen", "bake", "chef"],
            "reading": ["book", "read", "novel", "author", "library", "fiction"],
            "travel": ["travel", "trip", "vacation", "visit", "country", "city", "explore"],
            "technology": ["tech", "computer", "code", "programming", "software", "app"],
            "nature/outdoors": ["hike", "outdoors", "nature", "mountain", "trail", "camping"],
            "animals/pets": ["dog", "cat", "pet", "animal", "fish", "bird"],
            "movies/tv": ["movie", "film", "show", "watch", "netflix", "series"],
            "art/creativity": ["art", "paint", "draw", "creative", "design", "craft"],
            "gaming": ["game", "gaming", "play", "console", "video game"],
            "fitness": ["workout", "exercise", "yoga", "run", "fitness", "gym"],
        }
        all_text = " ".join(texts).lower()
        interest_scores = {}
        for interest, kws in interest_keywords.items():
            score = sum(all_text.count(kw) for kw in kws)
            if score > 0:
                interest_scores[interest] = score
        top = sorted(interest_scores.items(), key=lambda x: -x[1])[:8]
        return [i for i, _ in top]

    def extract(self) -> Dict[str, Any]:
        print("[Persona] Extracting persona from User 1 messages…")
        texts = self._get_user1_texts()
        print(f"  User 1 messages: {len(texts):,}")

        habits = self._extract_habits(texts)
        facts = self._extract_facts(texts)
        personality = self._extract_personality(texts)
        comm_style = self._extract_communication_style(texts)
        interests = self._infer_interests(texts)

        persona = {
            "habits": habits,
            "personal_facts": facts,
            "personality_traits": personality,
            "communication_style": comm_style,
            "inferred_interests": interests,
            "data_source": f"{len(texts):,} messages from User 1",
        }

        with open(PERSONA_FILE, "w", encoding="utf-8") as f:
            json.dump(persona, f, indent=2)
        print(f"[Persona] Saved to {PERSONA_FILE}")
        return persona

    @staticmethod
    def load() -> Dict[str, Any]:
        with open(PERSONA_FILE, encoding="utf-8") as f:
            return json.load(f)
