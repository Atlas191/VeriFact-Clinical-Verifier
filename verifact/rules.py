import re
from typing import Optional, Tuple, Dict, Any, List


class DeterministicConsistencyChecker:
    """
    Deterministic Consistency Layer for Clinical Factuality Verification.
    Catches numerical, dosage, frequency, and clinical disposition mismatches that
    semantic NLI models frequently miss or softly entail.
    
    Uses character-level spatial proximity and clause-level scoping to bind medications
    to their exact adjacent dosages and frequencies, preventing false contradictions
    when multiple medications appear in the same sentence.
    """

    DOSAGE_PATTERN = re.compile(
        r'(\d+(?:\.\d+)?)\s*(mg|ml|mcg|g|kg|units|unit|meq|iu|tablets|capsules|tabs|puffs)',
        re.IGNORECASE
    )

    FREQUENCY_MAP = {
        "daily": "qd",
        "once daily": "qd",
        "qd": "qd",
        "q.d.": "qd",
        "q24h": "qd",
        "q24": "qd",
        "every 24 hours": "qd",
        "twice daily": "bid",
        "bid": "bid",
        "b.i.d.": "bid",
        "q12h": "bid",
        "q12": "bid",
        "every 12 hours": "bid",
        "three times daily": "tid",
        "tid": "tid",
        "t.i.d.": "tid",
        "q8h": "tid",
        "q8": "tid",
        "every 8 hours": "tid",
        "four times daily": "qid",
        "qid": "qid",
        "q.i.d.": "qid",
        "q6h": "qid",
        "q6": "qid",
        "every 6 hours": "qid",
        "at bedtime": "qhs",
        "qhs": "qhs",
        "bedtime": "qhs",
        "as needed": "prn",
        "prn": "prn",
        "q4h": "q4h",
        "every 4 hours": "q4h",
    }

    DISPOSITION_CONFLICTS = [
        {"terms": ["home", "residence"], "group": "home"},
        {"terms": ["skilled nursing facility", "snf", "rehab", "rehabilitation", "subacute"], "group": "facility"},
        {"terms": ["hospice", "palliative care"], "group": "hospice"},
        {"terms": ["expired", "deceased", "morgue"], "group": "deceased"},
    ]

    CLINICAL_STOPWORDS = {
        "daily", "twice", "three", "times", "once", "dose", "dosage", "prescribed",
        "patient", "taking", "initiated", "administered", "started", "discharged",
        "admitted", "history", "treatment", "therapy", "medication", "given", "with",
        "home", "tablets", "capsules", "units", "puffs", "orders", "regimen",
        "transitioned", "subcutaneous", "oral", "intravenous", "course", "schedule",
        "secondary", "prevention", "basal", "management", "hospital", "clinic", "initial"
    }

    def extract_dosages(self, text: str) -> List[Tuple[float, str]]:
        """Extracts list of (value, unit) tuples from text."""
        matches = self.DOSAGE_PATTERN.findall(text)
        results = []
        for val_str, unit in matches:
            try:
                results.append((float(val_str), unit.lower()))
            except ValueError:
                continue
        return results

    def extract_frequency_for_med(self, text: str, med_word: str) -> Optional[str]:
        """Extracts canonical frequency in the local clause immediately following med_word."""
        med_m = re.search(r'\b' + re.escape(med_word) + r'\b', text, re.I)
        if not med_m:
            return None
        sub = text[med_m.end(): med_m.end() + 60]
        # Restrict to clause before next medication or punctuation
        sub = re.split(r'[,;]|\band\b', sub)[0]
        # Check longer phrases first
        for phrase in sorted(self.FREQUENCY_MAP.keys(), key=len, reverse=True):
            if re.search(r'\b' + re.escape(phrase) + r'\b', sub, re.I):
                return self.FREQUENCY_MAP[phrase]
        return None

    def extract_claim_frequency(self, text: str) -> Optional[str]:
        """Extracts canonical clinical frequency from claim text."""
        text_lower = text.lower()
        for phrase in sorted(self.FREQUENCY_MAP.keys(), key=len, reverse=True):
            if re.search(r'\b' + re.escape(phrase) + r'\b', text_lower):
                return self.FREQUENCY_MAP[phrase]
        return None

    def check_claim(
        self,
        claim_text: str,
        premise_text: str,
        entity_value: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Runs deterministic consistency verification between claim and premise.
        Returns:
            (verdict, reasoning): ("CONTRADICTED", reason) if a deterministic violation is found,
                                 or (None, None) if consistent / no rule triggered.
        """
        claim_lower = claim_text.lower()
        premise_lower = premise_text.lower()

        # Extract candidate medication words in claim that are also present in premise
        words = [re.sub(r'[^a-z]', '', w) for w in claim_lower.split()]
        candidate_meds = [
            w for w in words
            if len(w) > 3 and w not in self.CLINICAL_STOPWORDS and re.search(r'\b' + re.escape(w) + r'\b', premise_lower)
        ]

        # -------------------------------------------------------------
        # Check 1: Dosage & Unit Discrepancy with Proximity Binding
        # -------------------------------------------------------------
        claim_dosages = self.extract_dosages(claim_text)
        if entity_value and not claim_dosages:
            claim_dosages = self.extract_dosages(entity_value)

        if claim_dosages and candidate_meds:
            # Find all premise dosages with positions
            all_premise_dosages = []
            for m in self.DOSAGE_PATTERN.finditer(premise_text):
                try:
                    all_premise_dosages.append((m.start(), m.end(), float(m.group(1)), m.group(2).lower()))
                except ValueError:
                    continue

            found_match = False
            mismatches = []

            for med in candidate_meds:
                med_matches = list(re.finditer(r'\b' + re.escape(med) + r'\b', premise_text, re.I))
                closest = None
                min_dist = float('inf')
                for med_m in med_matches:
                    for d in all_premise_dosages:
                        dist = min(abs(d[0] - med_m.end()), abs(med_m.start() - d[1]))
                        if dist < min_dist:
                            min_dist = dist
                            closest = (dist, d[2], d[3])

                if closest and closest[0] <= 60:
                    p_dist, p_val, p_unit = closest
                    for c_val, c_unit in claim_dosages:
                        c_u = "unit" if c_unit in ["unit", "units"] else c_unit
                        p_u = "unit" if p_unit in ["unit", "units"] else p_unit
                        if c_u == p_u:
                            if abs(c_val - p_val) < 1e-4:
                                found_match = True
                                break
                            else:
                                mismatches.append((med, c_val, c_unit, p_val, p_unit))

            if not found_match and mismatches:
                med, c_val, c_unit, p_val, p_unit = mismatches[0]
                reason = (
                    f"Dosage mismatch for '{med}': Claim specifies {c_val} {c_unit}, "
                    f"but source premise records {p_val} {p_unit}."
                )
                return "CONTRADICTED", reason

        # -------------------------------------------------------------
        # Check 2: Frequency Schedule Discrepancy (Med-Scoped)
        # -------------------------------------------------------------
        claim_freq = self.extract_claim_frequency(claim_lower)
        if claim_freq and candidate_meds:
            for med in candidate_meds:
                prem_freq = self.extract_frequency_for_med(premise_text, med)
                if prem_freq and claim_freq != prem_freq:
                    reason = (
                        f"Frequency schedule mismatch for '{med}': Claim specifies '{claim_freq}', "
                        f"whereas source premise records '{prem_freq}'."
                    )
                    return "CONTRADICTED", reason

        # -------------------------------------------------------------
        # Check 3: Discharge Disposition Conflict
        # -------------------------------------------------------------
        claim_disposition = None
        premise_disposition = None

        for item in self.DISPOSITION_CONFLICTS:
            for term in item["terms"]:
                if re.search(r'\b' + re.escape(term) + r'\b', claim_lower):
                    claim_disposition = item["group"]
                    break
            for term in item["terms"]:
                if re.search(r'\b' + re.escape(term) + r'\b', premise_lower):
                    premise_disposition = item["group"]
                    break

        if claim_disposition and premise_disposition and claim_disposition != premise_disposition:
            if "discharg" in claim_lower and "discharg" in premise_lower:
                reason = (
                    f"Discharge disposition contradiction: Claim indicates '{claim_disposition}', "
                    f"whereas premise confirms '{premise_disposition}'."
                )
                return "CONTRADICTED", reason

        return None, None

    # Common clinical drug name fragments for entity extraction
    DRUG_SUFFIXES = [
        "olol", "pril", "sartan", "statin", "mab", "nib", "pine", "azole",
        "mycin", "cillin", "cycline", "floxacin", "prazole", "dipine", "gliptin",
        "glutide", "tide", "parin", "farin", "vir", "navir",
    ]

    KNOWN_DRUG_NAMES = {
        "aspirin", "metoprolol", "lisinopril", "atorvastatin", "warfarin",
        "amiodarone", "clopidogrel", "metformin", "prednisone", "insulin",
        "glargine", "lispro", "heparin", "vancomycin", "ciprofloxacin",
        "ceftriaxone", "amoxicillin", "alteplase", "semaglutide", "ozempic",
        "furosemide", "amlodipine", "omeprazole", "pantoprazole", "levothyroxine",
        "hydrochlorothiazide", "losartan", "carvedilol", "digoxin", "gabapentin",
        "acetaminophen", "ibuprofen", "morphine", "fentanyl", "lorazepam",
        "enoxaparin", "apixaban", "rivaroxaban", "dabigatran",
    }

    def extract_drug_entities(self, text: str) -> List[str]:
        """
        Extracts likely drug/medication names from text using known drug names
        and common pharmaceutical suffixes.
        """
        words = re.findall(r'[a-z]+', text.lower())
        drugs = []
        for w in words:
            if len(w) < 4:
                continue
            if w in self.KNOWN_DRUG_NAMES:
                drugs.append(w)
            elif any(w.endswith(suffix) for suffix in self.DRUG_SUFFIXES):
                drugs.append(w)
        return list(set(drugs))

    def entity_absent_check(
        self,
        claim_text: str,
        premise_text: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Determines if a claim references a specific medication/entity that is
        completely absent from the premise text. If an entity is absent, the claim
        is UNSUPPORTED by definition (not CONTRADICTED).

        This catches the common LLM error of classifying fabricated drug mentions
        as CONTRADICTED rather than UNSUPPORTED.

        Returns:
            ("UNSUPPORTED", reason) if claim drug entity is absent from premise,
            or (None, None) if the entity exists in the premise.
        """
        claim_drugs = self.extract_drug_entities(claim_text)
        if not claim_drugs:
            return None, None

        premise_lower = premise_text.lower()

        # Check if ANY of the claim's drug entities appear in the premise
        for drug in claim_drugs:
            if re.search(r'\b' + re.escape(drug) + r'\b', premise_lower):
                return None, None  # At least one drug entity found in premise

        # None of the claim's drug entities appear in the premise
        reason = (
            f"Entity absence: Claim references {', '.join(claim_drugs)} which "
            f"{'is' if len(claim_drugs) == 1 else 'are'} not mentioned anywhere in the source premise."
        )
        return "UNSUPPORTED", reason
