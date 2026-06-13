"""
Krav-dekomponering/klassificering. To modes:

1. ITEMISERET (foretrukket): Kunden har angivet udgiftsposter med beskrivelse
   + beløb (+ evt. type-hint). LLM'ens ENESTE opgave er at klassificere hver
   post til en kanonisk dækningstype. Hintet er vejledende, ALDRIG bindende —
   kunder vælger ofte forkert type (fx flyforsinkelse vs forsinket_fremmøde).

2. FRITEKST (fallback): Kunden skrev én samlet beskrivelse. LLM'en opdeler
   den i delkrav med type + beløb, som før.

Fail-safes: kan svaret ikke parses, eller stemmer delbeløbene ikke med
totalen (kun fritekst-mode), falder vi tilbage til ét samlet delkrav —
pipelinen eskalerer så til manuelt review i stedet for at gætte.
"""
import json
import re
from typing import Optional

try:  # mistralai >= 2.x
    from mistralai.client import Mistral
except ImportError:  # mistralai 1.x
    from mistralai import Mistral

from config.settings import (
    BELØB_SUM_TOLERANCE_DKK,
    MISTRAL_API_KEY,
    MISTRAL_MODEL_DECOMPOSER,
)
from src.models import DelKrav, ForsikringsKrav

# Kanoniske dækningstyper på tværs af alle kortgrupper.
# Klassificeringen MÅ KUN vælge fra denne liste (eller 'ukendt').
KANONISKE_DÆKNINGSTYPER = [
    "sygdom_og_hjemtransport",
    "krisehjælp",
    "sygeledsagelse",
    "tilkaldelse",
    "hjemkaldelse",
    "privatansvarsforsikring",
    "retshjælp_og_sikkerhedsstillelse",
    "afbestillingsforsikring",
    "overfald",
    "rejseulykke",
    "bagageforsinkelse",
    "flyforsinkelse",
    "forsinket_fremmøde",
    "eftersøgning_og_redning",
    "evakuering_og_ufrivilligt_ophold",
    "bagagedækning",
    "feriekompensation",
    "feriekompensation_og_erstatningsrejse",
    "ferieboligsikring",
    "forsikring_ved_billeje",
]

# Disambigueringsregler — delt mellem begge modes. Skrevet pga. konkrete
# fejlklassificeringer (fly-clusteret er den hyppigste).
KLASSIFICERINGSREGLER = """\
Klassificeringsregler — læs dem GRUNDIGT, forskellene er afgørende:

FLY-CLUSTERET (hyppigste fejlkilde):
- "flyforsinkelse" dækker udgifter MENS man venter på et forsinket/aflyst fly:
  fortæring, overnatning, nødindkøb af tøj/toiletartikler. IKKE nye billetter.
- "forsinket_fremmøde" dækker udgifter til at INDHENTE REJSERUTEN, når man
  uforskyldt kommer for sent til et transportmiddel (fx fordi et tidligere,
  separat fly var forsinket): nye billetter, transport, kost og logi undervejs.
- Tommelfingerregel: venter man → flyforsinkelse; skal man KØBE SIG VIDERE
  på ruten efter en mistet forbindelse → forsinket_fremmøde.

BAGAGE-CLUSTERET:
- Indkøb fordi indskrevet bagage er FORSINKET → "bagageforsinkelse"
- Beskadigede, ødelagte eller stjålne ejendele → "bagagedækning"

SYGDOM:
- Transport til/fra læge/hospital og lægebehandling → "sygdom_og_hjemtransport"
- Ødelagte/mistede feriedage pga. sygdom/sengeleje → "feriekompensation"

GENERELT:
- Aflysning af rejsen FØR afrejse → "afbestillingsforsikring"
- Passer ingen type, brug "ukendt" — gæt ALDRIG på en tilnærmelsesvis type
- Et eventuelt kunde-hint er VEJLEDENDE, ikke bindende. Kunder kender ikke
  forskellen på beslægtede dækninger — klassificér efter hvad udgiften ER.

Eksempler:
- "Vores fly Billund-København var forsinket, så vi missede vores separate fly
  til Nairobi og måtte købe nye billetter" → forsinket_fremmøde
  (selv hvis kunden har valgt hintet "flyforsinkelse"!)
- "Flyet var aflyst, vi måtte overnatte på hotel og købe aftensmad i lufthavnen"
  → flyforsinkelse
- "Kufferten kom først 2 dage senere, vi købte undertøj og tandbørster"
  → bagageforsinkelse
- "Min kuffert blev flået op og min jakke ødelagt under flyvningen"
  → bagagedækning
"""

SYSTEM_PROMPT_KLASSIFICERING = f"""\
Du er en assistent for en dansk rejseforsikringsskadebehandler.
Kunden har angivet sit krav som en liste af udgiftsposter. Din ENESTE opgave
er at klassificere HVER post med præcis én dækningstype fra denne liste:
{json.dumps(KANONISKE_DÆKNINGSTYPER, ensure_ascii=False, indent=2)}

Du må IKKE vurdere om noget er dækket — det gør et senere trin.
Du må IKKE ændre beskrivelser eller beløb.

{KLASSIFICERINGSREGLER}

Svar ALTID med gyldig JSON i præcis dette format og intet andet:
{{
  "klassificeringer": [
    {{"post_nr": 1, "dækningstype": "en type fra listen eller 'ukendt'"}},
    {{"post_nr": 2, "dækningstype": "..."}}
  ]
}}
Der SKAL være præcis én klassificering pr. post, i samme rækkefølge.
"""

SYSTEM_PROMPT_FRITEKST = f"""\
Du er en assistent for en dansk rejseforsikringsskadebehandler.
Din ENESTE opgave er at opdele et forsikringskrav i delkrav og klassificere hvert delkrav.
Du må IKKE vurdere om noget er dækket — det gør et senere trin.

Et delkrav er en enkelt udgift eller et enkelt tab, kunden søger erstatning for.
Eksempel: "taxa til hospitalet", "lægebehandling", "en ødelagt feriedag", "et par ødelagte bukser"
er FIRE separate delkrav, selvom de stammer fra samme hændelse.

Hvert delkrav klassificeres med PRÆCIS én dækningstype fra denne liste:
{json.dumps(KANONISKE_DÆKNINGSTYPER, ensure_ascii=False, indent=2)}

{KLASSIFICERINGSREGLER}

Beløbsregler:
- Angiv beløb pr. delkrav KUN hvis det fremgår eksplicit eller kan udledes entydigt
- Hvis kunden kun angiver et totalbeløb uden opdeling: sæt "beløb_dkk" til null på alle delkrav
- Opfind ALDRIG en fordeling af beløb

Svar ALTID med gyldig JSON i præcis dette format og intet andet:
{{
  "delkrav": [
    {{
      "beskrivelse": "kort beskrivelse med kundens egne ord",
      "dækningstype": "en type fra listen eller 'ukendt'",
      "beløb_dkk": tal eller null
    }}
  ]
}}
"""


class ClaimDecomposer:
    def __init__(
        self,
        api_key: str = MISTRAL_API_KEY,
        model: str = MISTRAL_MODEL_DECOMPOSER,
        client: Optional[Mistral] = None,
    ):
        if client is not None:
            self._client = client
        else:
            if not api_key:
                raise ValueError("MISTRAL_API_KEY er ikke sat. Kopiér .env.example til .env og udfyld din nøgle.")
            self._client = Mistral(api_key=api_key)
        self._model = model

    # ── Offentlig indgang ──────────────────────────────────────────────

    def decompose(self, krav: ForsikringsKrav) -> list[DelKrav]:
        """Vælg mode: itemiserede poster → klassificering, ellers fritekst."""
        if krav.poster:
            return self._klassificer_poster(krav)
        return self._decompose_fritekst(krav)

    # ── Mode 1: klassificering af itemiserede poster ───────────────────

    def _klassificer_poster(self, krav: ForsikringsKrav) -> list[DelKrav]:
        """
        Ét LLM-kald klassificerer alle poster. Fail-safe pr. post:
        ugyldig/manglende klassificering → brug kundens hint hvis gyldigt,
        ellers 'ukendt' (→ manuelt review for den post).
        """
        try:
            response = self._client.chat.complete(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_KLASSIFICERING},
                    {"role": "user", "content": self._build_poster_prompt(krav)},
                ],
                temperature=0.0,
            )
            raw = response.choices[0].message.content
            typer = self._parse_klassificeringer(raw, antal=len(krav.poster))
        except Exception as e:
            print(f"  [DECOMP] Klassificeringsfejl ({type(e).__name__}): {e} — bruger hints/ukendt")
            typer = [None] * len(krav.poster)

        delkrav = []
        for i, (post, dtype) in enumerate(zip(krav.poster, typer), 1):
            if dtype is None:
                dtype = post.dækningstype_hint if post.dækningstype_hint in KANONISKE_DÆKNINGSTYPER else "ukendt"
            delkrav.append(
                DelKrav(
                    delkrav_id=f"{krav.krav_id}-D{i}",
                    beskrivelse=post.beskrivelse,
                    dækningstype=dtype,
                    beløb_dkk=post.beløb_dkk,
                )
            )
        return delkrav

    @staticmethod
    def _build_poster_prompt(krav: ForsikringsKrav) -> str:
        prompt = f"## Hændelse (kontekst)\n{krav.hændelse_beskrivelse}\n\n## Udgiftsposter\n"
        for i, post in enumerate(krav.poster, 1):
            prompt += f"{i}. {post.beskrivelse} — {post.beløb_dkk} DKK"
            if post.dækningstype_hint:
                prompt += f" (kundens hint: {post.dækningstype_hint})"
            prompt += "\n"
        return prompt

    @staticmethod
    def _parse_klassificeringer(raw: str, antal: int) -> list[Optional[str]]:
        """Returnerer en type pr. post (None hvis ugyldig/mangler)."""
        cleaned = re.sub(r"```json\s*|```\s*", "", raw).strip()
        data = json.loads(cleaned)
        items = data.get("klassificeringer", [])

        typer: list[Optional[str]] = [None] * antal
        for item in items:
            nr = item.get("post_nr")
            dtype = item.get("dækningstype")
            if isinstance(nr, int) and 1 <= nr <= antal and dtype in KANONISKE_DÆKNINGSTYPER:
                typer[nr - 1] = dtype
        return typer

    # ── Mode 2: fritekst-dekomponering (uændret adfærd) ────────────────

    def _decompose_fritekst(self, krav: ForsikringsKrav) -> list[DelKrav]:
        try:
            response = self._client.chat.complete(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_FRITEKST},
                    {"role": "user", "content": self._build_fritekst_prompt(krav)},
                ],
                temperature=0.0,
            )
            raw = response.choices[0].message.content
            delkrav = self._parse_fritekst(krav, raw)
        except Exception as e:
            print(f"  [DECOMP] Fejl under dekomponering ({type(e).__name__}): {e}")
            return [self._fallback_delkrav(krav)]

        if not self._beløb_er_konsistente(krav, delkrav):
            print("  [DECOMP] Delbeløb stemmer ikke med totalbeløb — fallback til samlet delkrav")
            return [self._fallback_delkrav(krav)]

        return delkrav

    @staticmethod
    def _build_fritekst_prompt(krav: ForsikringsKrav) -> str:
        prompt = f"""## Forsikringskrav
- Samlet ansøgt beløb: {krav.beløb_dkk} DKK
- Kundens beskrivelse: {krav.hændelse_beskrivelse}"""
        if krav.dækningstype:
            prompt += f"\n- Kundens angivne dækningstype (kun et hint): {krav.dækningstype}"
        if krav.dokumentation:
            prompt += f"\n- Vedlagt dokumentation: {krav.dokumentation}"
        return prompt

    def _parse_fritekst(self, krav: ForsikringsKrav, raw: str) -> list[DelKrav]:
        cleaned = re.sub(r"```json\s*|```\s*", "", raw).strip()
        data = json.loads(cleaned)
        items = data.get("delkrav", [])
        if not items:
            return [self._fallback_delkrav(krav)]

        delkrav = []
        for i, item in enumerate(items, 1):
            dtype = item.get("dækningstype", "ukendt")
            if dtype not in KANONISKE_DÆKNINGSTYPER:
                dtype = "ukendt"
            beløb = item.get("beløb_dkk")
            delkrav.append(
                DelKrav(
                    delkrav_id=f"{krav.krav_id}-D{i}",
                    beskrivelse=str(item.get("beskrivelse", "")).strip() or "Uspecificeret delkrav",
                    dækningstype=dtype,
                    beløb_dkk=float(beløb) if beløb is not None else None,
                )
            )
        return delkrav

    @staticmethod
    def _beløb_er_konsistente(krav: ForsikringsKrav, delkrav: list[DelKrav]) -> bool:
        beløb = [d.beløb_dkk for d in delkrav]
        if any(b is None for b in beløb):
            return True
        return abs(sum(beløb) - krav.beløb_dkk) <= BELØB_SUM_TOLERANCE_DKK

    @staticmethod
    def _fallback_delkrav(krav: ForsikringsKrav) -> DelKrav:
        return DelKrav(
            delkrav_id=f"{krav.krav_id}-D1",
            beskrivelse=krav.hændelse_beskrivelse,
            dækningstype=krav.dækningstype or "ukendt",
            beløb_dkk=krav.beløb_dkk,
        )
