"""
Aggregering: Kombinerer delafgørelser til én samlet KravAfgørelse
og bygger en deterministisk, kundevendt besked.

Aggregeringsregler (i prioriteret rækkefølge):
1. Mindst ét delkrav i manuelt_review → HELE kravet i manuelt_review.
2. Alle godkendt → godkendt
3. Alle afvist → afvist
4. Blandet godkendt/afvist → delvist_godkendt

Kundebeskeden bygges af en TEMPLATE, ikke en LLM. I produktion må den
kundevendte tekst ikke kunne hallucinere — begrundelserne fra
evalueringen citeres, men strukturen er deterministisk.
"""
from src.models import (
    Afgørelse,
    DelAfgørelse,
    DelAfgørelseType,
    ForsikringsKrav,
    KravAfgørelse,
)

# Mapping fra enum-værdier til læsbare kortnavne
_KORTNIVEAU_LABELS = {
    "mastercard_gold": "Nykredit Mastercard Gold",
    "mastercard_platinum": "Nykredit Mastercard Platinum",
    "mastercard_world_elite": "Nykredit Mastercard World Elite",
    "mastercard_gold_business": "Nykredit Mastercard Gold Business",
    "mastercard_platinum_business": "Nykredit Mastercard Platinum Business",
    "mastercard_world_elite_business": "Nykredit Mastercard World Elite Business",
    "mastercard_gold_family": "Nykredit Mastercard Gold (Familiedækning)",
    "mastercard_platinum_family": "Nykredit Mastercard Platinum (Familiedækning)",
    "mastercard_world_elite_family": "Nykredit Mastercard World Elite (Familiedækning)",
}

_DÆKNINGSTYPE_LABELS = {
    "sygdom_og_hjemtransport": "sygdom/tilskadekomst",
    "bagageforsinkelse": "bagageforsinkelse",
    "bagagedækning": "bagage",
    "feriekompensation": "feriekompensation",
    "afbestillingsforsikring": "afbestilling",
    "flyforsinkelse": "flyforsinkelse",
    "forsinket_fremmøde": "forsinket fremmøde",
    "privatansvar": "privatansvar",
    "retshjælp": "retshjælp",
    "selvrisikodækning": "selvrisikodækning",
    "rejsedokumenter": "rejsedokumenter",
    "evakuering_og_eftersøgning": "evakuering og eftersøgning",
}


def _kortniveau_label(kortniveau) -> str:
    raw = kortniveau.value if hasattr(kortniveau, "value") else str(kortniveau)
    return _KORTNIVEAU_LABELS.get(raw, raw)


def _dækningstype_label(dækningstype: str) -> str:
    return _DÆKNINGSTYPE_LABELS.get(dækningstype, dækningstype)


def _find_primary_dækningstype(delafgørelser: list[DelAfgørelse]) -> str:
    if not delafgørelser:
        return "rejseforsikring"
    typer = {}
    for d in delafgørelser:
        label = _dækningstype_label(d.dækningstype)
        typer[label] = typer.get(label, 0.0) + (d.beløb_dkk or 0.0)
    if typer:
        primary = max(typer, key=typer.get)
        return f"{primary} m.fl." if len(typer) > 1 else primary
    return "rejseforsikring"


def aggreger(krav: ForsikringsKrav, delafgørelser: list[DelAfgørelse]) -> KravAfgørelse:
    godkendte = [d for d in delafgørelser if d.afgørelse == DelAfgørelseType.GODKENDT]
    afviste = [d for d in delafgørelser if d.afgørelse == DelAfgørelseType.AFVIST]
    reviews = [d for d in delafgørelser if d.afgørelse == DelAfgørelseType.MANUELT_REVIEW]

    godkendt_beløb = sum(d.godkendt_beløb_dkk or 0.0 for d in godkendte)

    llm_konfidenser = [d.konfidens for d in delafgørelser if not d.metadata_filtreret]
    samlet_konfidens = min(llm_konfidenser) if llm_konfidenser else 1.0

    if reviews:
        afgørelse = Afgørelse.MANUELT_REVIEW
        begrundelse = (
            f"{len(reviews)} af {len(delafgørelser)} delkrav kræver manuelt review: "
            + "; ".join(f"[{d.delkrav_id}] {d.begrundelse}" for d in reviews)
        )
        godkendt_beløb = 0.0
    elif godkendte and not afviste:
        afgørelse = Afgørelse.GODKENDT
        begrundelse = f"Alle {len(delafgørelser)} delkrav godkendt"
    elif afviste and not godkendte:
        afgørelse = Afgørelse.AFVIST
        begrundelse = (
            f"Alle {len(delafgørelser)} delkrav afvist: "
            + "; ".join(f"[{d.delkrav_id}] {d.begrundelse}" for d in afviste)
        )
    else:
        afgørelse = Afgørelse.DELVIST_GODKENDT
        begrundelse = (
            f"{len(godkendte)} delkrav godkendt ({godkendt_beløb:.2f} DKK), "
            f"{len(afviste)} delkrav afvist"
        )

    return KravAfgørelse(
        krav_id=krav.krav_id,
        afgørelse=afgørelse,
        begrundelse=begrundelse,
        kundebesked=byg_kundebesked(krav, afgørelse, delafgørelser, godkendt_beløb),
        konfidens=samlet_konfidens,
        delafgørelser=delafgørelser,
        ansøgt_beløb_dkk=krav.beløb_dkk,
        godkendt_beløb_dkk=godkendt_beløb,
        metadata_filtreret=all(d.metadata_filtreret for d in delafgørelser),
    )


def byg_kundebesked(
    krav: ForsikringsKrav,
    afgørelse: Afgørelse,
    delafgørelser: list[DelAfgørelse],
    godkendt_beløb: float,
) -> str:
    """Deterministisk kundevendt besked. Ingen LLM — ingen hallucination."""
    kort_label = _kortniveau_label(krav.kortniveau)
    emne = _find_primary_dækningstype(delafgørelser)
    linjer = []

    # ── Åbning ──────────────────────────────────────────────
    linjer.append(
        f"Tak for din henvendelse vedrørende {emne} under din rejse."
    )
    linjer.append(
        f"Vi har gennemgået dit krav ({krav.krav_id}) i henhold til "
        f"forsikringsbetingelserne for {kort_label}."
    )
    linjer.append("")

    # ── Manuelt review ──────────────────────────────────────
    if afgørelse == Afgørelse.MANUELT_REVIEW:
        linjer.append(
            "Dit krav kræver en nærmere vurdering af en sagsbehandler. "
            "Du vil høre fra os hurtigst muligt med en afgørelse."
        )
        linjer.append("")
        linjer.append(
            "Såfremt du har spørgsmål i mellemtiden, "
            "er du velkommen til at kontakte os."
        )
        linjer.append("")
        linjer.append("Du ønskes en god dag.")
        return "\n".join(linjer)

    godkendte = [d for d in delafgørelser if d.afgørelse == DelAfgørelseType.GODKENDT]
    afviste = [d for d in delafgørelser if d.afgørelse == DelAfgørelseType.AFVIST]

    # Split godkendte into normal and goodwill
    normal_godkendte = [d for d in godkendte if not getattr(d, "goodwill", False)]
    goodwill_godkendte = [d for d in godkendte if getattr(d, "goodwill", False)]

    normal_beløb = sum(d.godkendt_beløb_dkk or 0.0 for d in normal_godkendte)
    goodwill_beløb = sum(d.godkendt_beløb_dkk or 0.0 for d in goodwill_godkendte)

    # ── Godkendelser (altid først — good news first) ────────
    if godkendte:
        linjer.append("Afgørelse")
        if afgørelse == Afgørelse.GODKENDT and not goodwill_godkendte:
            linjer.append(
                f"Vi kan godkende dit krav og udbetaler {godkendt_beløb:.2f} DKK "
                f"til din NemKonto."
            )
        elif afgørelse == Afgørelse.GODKENDT and goodwill_godkendte:
            linjer.append(
                f"Vi udbetaler {godkendt_beløb:.2f} DKK til din NemKonto."
            )
        else:
            linjer.append(
                f"Vi kan godkende en del af dit krav og udbetaler "
                f"{godkendt_beløb:.2f} DKK til din NemKonto."
            )
        linjer.append("")

        # Normal approvals — itemized
        if normal_godkendte:
            linjer.append("Erstatningsopgørelse:")
            for d in normal_godkendte:
                beløb_str = f"{d.godkendt_beløb_dkk:.2f} DKK" if d.godkendt_beløb_dkk else "—"
                linjer.append(f"  {d.beskrivelse}: {beløb_str}")
            if not goodwill_godkendte:
                linjer.append(f"  I alt: {normal_beløb:.2f} DKK")
            linjer.append("")

        # Goodwill approvals — clearly framed as a one-time exception
        if goodwill_godkendte:
            linjer.append(
                "Derudover har vi valgt undtagelsesvist at imødekomme følgende "
                "udgifter. Vi gør opmærksom på, at disse udgifter normalt ikke "
                "er dækket i henhold til forsikringsbetingelserne, men vi har "
                "valgt at dække dem i denne sag:"
            )
            linjer.append("")
            for d in goodwill_godkendte:
                beløb_str = f"{d.godkendt_beløb_dkk:.2f} DKK" if d.godkendt_beløb_dkk else "—"
                linjer.append(f"  {d.beskrivelse}: {beløb_str}")
            linjer.append("")
            linjer.append(f"  Samlet udbetaling: {godkendt_beløb:.2f} DKK")
            linjer.append("")

    # ── Afvisninger ─────────────────────────────────────────
    if afviste:
        if godkendte:
            linjer.append(
                "Vi har desuden gennemgået følgende dele af dit krav, "
                "som desværre ikke kan imødekommes:"
            )
        else:
            linjer.append(
                "Vi har gennemgået dit krav, men kan desværre ikke "
                "imødekomme det ud fra de gældende forsikringsbetingelser."
            )
        linjer.append("")

        for d in afviste:
            beløb_note = f" ({d.beløb_dkk:.2f} DKK)" if d.beløb_dkk else ""
            linjer.append(f"{d.beskrivelse}{beløb_note}")
            linjer.append(f"{d.begrundelse}")
            linjer.append("")

    # ── Afslutning ──────────────────────────────────────────
    linjer.append(
        "Du kan finde dine forsikringsbetingelser på nykredit.dk "
        "under dit kort, hvis du ønsker at læse bestemmelserne i deres helhed."
    )
    linjer.append("")
    linjer.append(
        "Såfremt du har yderligere spørgsmål, er du velkommen til at kontakte os."
    )
    linjer.append("")
    linjer.append("Du ønskes en god dag.")

    return "\n".join(linjer)
