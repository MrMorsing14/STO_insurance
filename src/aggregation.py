"""
Aggregering: Kombinerer delafgørelser til én samlet KravAfgørelse
og bygger en deterministisk, kundevendt besked.

Aggregeringsregler (i prioriteret rækkefølge):
1. Mindst ét delkrav i manuelt_review → HELE kravet i manuelt_review.
   Rationale: En sagsbehandler skal alligevel røre sagen, og en delvis
   auto-udbetaling efterfulgt af en manuel afgørelse på resten giver
   kunden to forvirrende svar på samme krav. Den itemiserede analyse
   følger med, så sagsbehandleren har et forspring.
2. Alle godkendt → godkendt
3. Alle afvist → afvist
4. Blandet godkendt/afvist → delvist_godkendt (det er "bukse-casen")

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
        godkendt_beløb = 0.0  # ingen auto-udbetaling før den manuelle vurdering
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
    linjer = [f"Vedr. dit krav {krav.krav_id}:", ""]

    if afgørelse == Afgørelse.MANUELT_REVIEW:
        linjer.append(
            "Dit krav kræver en manuel vurdering af en sagsbehandler. "
            "Du hører fra os hurtigst muligt."
        )
        return "\n".join(linjer)

    godkendte = [d for d in delafgørelser if d.afgørelse == DelAfgørelseType.GODKENDT]
    afviste = [d for d in delafgørelser if d.afgørelse == DelAfgørelseType.AFVIST]

    if godkendte:
        linjer.append("Vi kan godkende følgende:")
        for d in godkendte:
            beløb = f" ({d.godkendt_beløb_dkk:.2f} DKK)" if d.godkendt_beløb_dkk else ""
            linjer.append(f"  • {d.beskrivelse}{beløb}")
        linjer.append("")

    if afviste:
        linjer.append("Vi kan desværre ikke godkende følgende:")
        for d in afviste:
            linjer.append(f"  • {d.beskrivelse} — {d.begrundelse}")
        linjer.append("")

    if afgørelse == Afgørelse.AFVIST:
        linjer.append("Dit krav er derfor afvist i sin helhed.")
    else:
        linjer.append(f"Samlet godkendt beløb: {godkendt_beløb:.2f} DKK af ansøgte {krav.beløb_dkk:.2f} DKK.")

    return "\n".join(linjer)
