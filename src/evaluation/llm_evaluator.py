"""
LLM-evaluator: Vurderer ÉT delkrav ad gangen mod policychunks + metadata.

Designvalg: ét LLM-kald pr. delkrav i stedet for ét stort kald for hele kravet.
Det koster flere kald, men giver:
- Fokuseret retrieval (chunks matcher præcis delkravets dækningstype)
- Konfidens og routing pr. delkrav i stedet for én udvandet samlet konfidens
- Begrundelser der kan citeres direkte i kundebeskeden
For krav under 5.000 DKK er den ekstra latenstid/pris acceptabel.
"""
import json
import re
from typing import Optional

try:  # mistralai >= 2.x
    from mistralai.client import Mistral
except ImportError:  # mistralai 1.x
    from mistralai import Mistral

from config.settings import MISTRAL_API_KEY, MISTRAL_MODEL_EVALUATOR
from src.models import DelAfgørelse, DelAfgørelseType, DelKrav, ForsikringsKrav

SYSTEM_PROMPT = """\
Du er en forsikringsskadebehandler for Nykredit Mastercard rejseforsikring.
Du skal være KRITISK og PRÆCIS — du beskytter forsikringsselskabets interesser,
men du skal også godkende berettigede krav uden unødig friktion.

Du vurderer ÉT DELKRAV ad gangen. Delkravet er en enkelt udgift udskilt fra
kundens samlede krav. Du modtager:
1. Konteksten for det samlede krav (hændelse, datoer, dokumentation)
2. Det specifikke delkrav du skal vurdere
3. Relevante uddrag fra forsikringsbetingelserne
4. Struktureret dækningsinformation inkl. udbetalingsregler for kortniveauet

Vurdér ALLE punkter for delkravet:
A) Er denne specifikke udgift dækket af den angivne dækningstype?
B) Er udbetalingsreglerne opfyldt? (Hvem har ret til udbetaling?)
C) Er beløbet korrekt og inden for maksimumgrænserne?
D) Er dokumentationskravene opfyldt for netop denne udgift?
E) Rammes delkravet af en undtagelse (specifik eller generel)?

Svar ALTID i dette JSON-format og intet andet:
{
  "afgørelse": "godkendt" | "afvist" | "manuelt_review",
  "begrundelse": "Kort, præcis begrundelse på dansk med reference til relevante betingelser",
  "konfidens": 0.0-1.0,
  "relevante_betingelser": ["betingelse 1", "betingelse 2"],
  "godkendt_beløb_dkk": null eller et tal i DKK,
  "forslag_til_anden_dækningstype": null eller en kanonisk dækningstype
}

VIGTIGT om fejlklassificering:
Delkravets dækningstype kan være FORKERT klassificeret (af kunden eller et
tidligere trin). Hvis din afvisning reelt betyder "denne udgift hører ikke
under DENNE dækningstype, men kunne høre under en anden" — fx nye flybilletter
efter en mistet forbindelse, som er undtaget under flyforsinkelse, men dækkes
af forsinket_fremmøde — så sæt "forslag_til_anden_dækningstype" til den type
du mener udgiften hører under. Sæt den KUN når undtagelsen/afvisningen handler
om forkert kategori, IKKE når udgiften slet ikke er forsikringsdækket
(fx almindelige ejendele der hører under kundens indboforsikring).

Regler:
- "godkendt" KUN hvis delkravet FULDT UD opfylder alle betingelser
- "afvist" hvis udgiften falder uden for dækningen eller rammes af en undtagelse
- "manuelt_review" ved tvivl, manglende beløb, manglende dokumentation, eller hvis
  beløbet kræver en beregning du ikke kan udføre sikkert
- Udbetalingsregler i den strukturerede dækningsinformation har HØJESTE prioritet
- Hvis delkravet mangler et specifikt beløb, kan du IKKE godkende — vælg manuelt_review
- Ved "godkendt": sæt godkendt_beløb_dkk til det berettigede beløb (kan være lavere end ansøgt)
- Konfidens: 0.9+ = meget sikker, 0.5-0.7 = usikker
"""


class LLMEvaluator:
    def __init__(self, api_key: str = MISTRAL_API_KEY, model: str = MISTRAL_MODEL_EVALUATOR, client: Optional[Mistral] = None):
        if client is not None:
            self._client = client
        else:
            if not api_key:
                raise ValueError("MISTRAL_API_KEY er ikke sat. Kopiér .env.example til .env og udfyld din nøgle.")
            self._client = Mistral(api_key=api_key)
        self._model = model

    def evaluate_delkrav(
        self,
        krav: ForsikringsKrav,
        delkrav: DelKrav,
        policy_chunks: list[dict],
        coverage_context: dict,
    ) -> DelAfgørelse:
        user_prompt = self._build_prompt(krav, delkrav, policy_chunks, coverage_context)

        response = self._client.chat.complete(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )

        raw_response = response.choices[0].message.content
        return self._parse_response(delkrav, raw_response)

    @staticmethod
    def _build_prompt(
        krav: ForsikringsKrav,
        delkrav: DelKrav,
        policy_chunks: list[dict],
        coverage_context: dict,
    ) -> str:
        krav_section = f"""## Samlet krav (kontekst)
- Krav ID: {krav.krav_id}
- Kortniveau: {krav.kortniveau.value}
- Samlet ansøgt beløb: {krav.beløb_dkk} DKK
- Hændelsesdato: {krav.hændelse_dato}
- Hændelsesbeskrivelse: {krav.hændelse_beskrivelse}"""
        if krav.rejse_startdato:
            krav_section += f"\n- Rejseperiode: {krav.rejse_startdato} til {krav.rejse_slutdato}"
        if krav.dokumentation:
            krav_section += f"\n- Dokumentation: {krav.dokumentation}"

        beløb_str = f"{delkrav.beløb_dkk} DKK" if delkrav.beløb_dkk is not None else "IKKE SPECIFICERET"
        delkrav_section = f"""## DELKRAV DER SKAL VURDERES
- Delkrav ID: {delkrav.delkrav_id}
- Beskrivelse: {delkrav.beskrivelse}
- Dækningstype: {delkrav.dækningstype}
- Beløb: {beløb_str}"""

        coverage_section = f"""## Dækningsinformation (struktureret)
```json
{json.dumps(coverage_context, ensure_ascii=False, indent=2)}
```"""

        chunks_section = "## Relevante uddrag fra forsikringsbetingelserne\n"
        if policy_chunks:
            for i, chunk in enumerate(policy_chunks, 1):
                meta = chunk.get("metadata", {})
                chunks_section += f"\n### Uddrag {i} (sektion: {meta.get('sektion', 'ukendt')})\n"
                chunks_section += chunk["text"] + "\n"
        else:
            chunks_section += "\nIngen specifikke policytekst-chunks fundet.\n"

        return f"{krav_section}\n\n{delkrav_section}\n\n{coverage_section}\n\n{chunks_section}"

    @staticmethod
    def _parse_response(delkrav: DelKrav, raw: str) -> DelAfgørelse:
        try:
            cleaned = re.sub(r"```json\s*|```\s*", "", raw).strip()
            data = json.loads(cleaned)

            afgørelse_map = {
                "godkendt": DelAfgørelseType.GODKENDT,
                "afvist": DelAfgørelseType.AFVIST,
                "manuelt_review": DelAfgørelseType.MANUELT_REVIEW,
            }
            afgørelse = afgørelse_map.get(data.get("afgørelse", ""), DelAfgørelseType.MANUELT_REVIEW)

            godkendt_beløb = data.get("godkendt_beløb_dkk")
            if afgørelse == DelAfgørelseType.AFVIST:
                godkendt_beløb = 0.0
            elif afgørelse == DelAfgørelseType.GODKENDT and godkendt_beløb is None:
                # Godkendt uden beløb giver ikke mening i STO — brug ansøgt beløb
                # hvis det findes, ellers eskalér
                if delkrav.beløb_dkk is not None:
                    godkendt_beløb = delkrav.beløb_dkk
                else:
                    afgørelse = DelAfgørelseType.MANUELT_REVIEW

            forslag = data.get("forslag_til_anden_dækningstype")
            if not isinstance(forslag, str) or forslag == delkrav.dækningstype:
                forslag = None

            return DelAfgørelse(
                delkrav_id=delkrav.delkrav_id,
                beskrivelse=delkrav.beskrivelse,
                dækningstype=delkrav.dækningstype,
                beløb_dkk=delkrav.beløb_dkk,
                afgørelse=afgørelse,
                begrundelse=data.get("begrundelse", "Ingen begrundelse givet"),
                konfidens=max(0.0, min(1.0, float(data.get("konfidens", 0.5)))),
                godkendt_beløb_dkk=float(godkendt_beløb) if godkendt_beløb is not None else None,
                relevante_betingelser=data.get("relevante_betingelser", []),
                forslag_til_anden_dækningstype=forslag,
                metadata_filtreret=False,
            )
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            return DelAfgørelse(
                delkrav_id=delkrav.delkrav_id,
                beskrivelse=delkrav.beskrivelse,
                dækningstype=delkrav.dækningstype,
                beløb_dkk=delkrav.beløb_dkk,
                afgørelse=DelAfgørelseType.MANUELT_REVIEW,
                begrundelse=f"Kunne ikke parse LLM-svar: {str(e)[:200]}. Rå response: {raw[:300]}",
                konfidens=0.0,
                metadata_filtreret=False,
            )
