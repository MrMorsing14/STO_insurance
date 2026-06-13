/* STO Prototype — frontend logik.
   Kalder POST /api/claims og renderer delkrav-opdelingen. */

const kortLabels = {
  mastercard_gold: 'Gold', mastercard_gold_family: 'Gold Family',
  mastercard_platinum: 'Platinum', mastercard_platinum_family: 'Platinum Family',
  world_elite: 'World Elite', world_elite_family: 'World Elite Family',
  mastercard_business: 'Business', mastercard_business_platinum: 'Business Platinum'
};

const statusConfig = {
  godkendt:          { icon: 'ti-circle-check',  title: 'Kravet er godkendt' },
  delvist_godkendt:  { icon: 'ti-circle-half-2', title: 'Kravet er delvist godkendt' },
  afvist:            { icon: 'ti-circle-x',      title: 'Kravet er afvist' },
  manuelt_review:    { icon: 'ti-clock',         title: 'Sendt til manuel behandling' }
};

const delStatusIcon = {
  godkendt: 'ti-check',
  afvist: 'ti-x',
  manuelt_review: 'ti-clock'
};


const typeOptions = [
  ['', 'Lad systemet vurdere'],
  ['sygdom_og_hjemtransport', 'Sygdom og hjemtransport'],
  ['afbestillingsforsikring', 'Afbestilling'],
  ['bagageforsinkelse', 'Bagage forsinket'],
  ['bagagedækning', 'Bagage stjålet/beskadiget'],
  ['flyforsinkelse', 'Flyforsinkelse (ventetid)'],
  ['forsinket_fremmøde', 'Mistet forbindelse / forsinket fremmøde'],
  ['feriekompensation', 'Ødelagte feriedage'],
  ['feriekompensation_og_erstatningsrejse', 'Ødelagte feriedage / erstatningsrejse'],
  ['rejseulykke', 'Rejseulykke'],
  ['overfald', 'Overfald'],
  ['privatansvarsforsikring', 'Privatansvar'],
  ['forsikring_ved_billeje', 'Billeje'],
  ['sygeledsagelse', 'Sygeledsagelse'],
  ['tilkaldelse', 'Tilkaldelse'],
  ['hjemkaldelse', 'Hjemkaldelse'],
  ['retshjælp_og_sikkerhedsstillelse', 'Retshjælp'],
  ['eftersøgning_og_redning', 'Eftersøgning og redning'],
  ['evakuering_og_ufrivilligt_ophold', 'Evakuering'],
  ['krisehjælp', 'Krisehjælp']
];

function addPost(desc = '', amount = '', hint = '') {
  const list = document.getElementById('postList');
  const row = document.createElement('div');
  row.className = 'post-row';

  const descInput = document.createElement('input');
  descInput.type = 'text';
  descInput.className = 'post-desc';
  descInput.placeholder = 'Hvad er udgiften? Fx "nye flybilletter København-Nairobi"';
  descInput.value = desc;

  const amountWrap = document.createElement('div');
  amountWrap.className = 'input-with-suffix post-amount-wrap';
  const amountInput = document.createElement('input');
  amountInput.type = 'number';
  amountInput.className = 'post-amount';
  amountInput.min = '0';
  amountInput.step = '50';
  amountInput.placeholder = '0';
  amountInput.value = amount;
  amountInput.addEventListener('input', updateSum);
  const suffix = document.createElement('span');
  suffix.className = 'suffix';
  suffix.textContent = 'DKK';
  amountWrap.appendChild(amountInput);
  amountWrap.appendChild(suffix);

  const typeSelect = document.createElement('select');
  typeSelect.className = 'post-type';
  for (const [val, label] of typeOptions) {
    const opt = document.createElement('option');
    opt.value = val;
    opt.textContent = label;
    typeSelect.appendChild(opt);
  }
  typeSelect.value = hint;

  const removeBtn = document.createElement('button');
  removeBtn.type = 'button';
  removeBtn.className = 'post-remove';
  removeBtn.title = 'Fjern post';
  removeBtn.innerHTML = '<i class="ti ti-trash" aria-hidden="true"></i>';
  removeBtn.addEventListener('click', () => { row.remove(); updateSum(); });

  row.appendChild(descInput);
  row.appendChild(amountWrap);
  row.appendChild(typeSelect);
  row.appendChild(removeBtn);
  list.appendChild(row);
  updateSum();
}

function readPoster() {
  return [...document.querySelectorAll('.post-row')].map(row => ({
    beskrivelse: row.querySelector('.post-desc').value.trim(),
    beløb_dkk: parseFloat(row.querySelector('.post-amount').value) || 0,
    dækningstype_hint: row.querySelector('.post-type').value || null
  }));
}

function updateSum() {
  const sum = readPoster().reduce((a, p) => a + p.beløb_dkk, 0);
  document.getElementById('postSum').textContent =
    sum.toLocaleString('da-DK') + ' DKK';
}

// Start med én demo-post (din kammerats case)
document.addEventListener('DOMContentLoaded', () => {
  addPost('Nye flybilletter København til Nairobi efter mistet forbindelse', 4000, '');
});

// ── Klient-side rate limit (kosmetisk — hvert krav koster flere LLM-kald) ──
const RATE_LIMIT = 5;
const RATE_WINDOW_MS = 2 * 60 * 1000;
let submitTimestamps = JSON.parse(sessionStorage.getItem('sto_timestamps') || '[]');

function isRateLimited() {
  const now = Date.now();
  submitTimestamps = submitTimestamps.filter(t => now - t < RATE_WINDOW_MS);
  return submitTimestamps.length >= RATE_LIMIT;
}

function recordSubmit() {
  submitTimestamps.push(Date.now());
  sessionStorage.setItem('sto_timestamps', JSON.stringify(submitTimestamps));
}

// ── Submit ──────────────────────────────────────────────────────────────

const LOADING_STEPS = [
  'Opdeler kravet i delkrav…',
  'Slår dækning op…',
  'Vurderer hvert delkrav mod betingelserne…',
  'Samler afgørelsen…'
];
let loadingInterval = null;

async function submitClaim() {
  hideError();

  if (isRateLimited()) {
    showError('For mange forsøg — vent et øjeblik før du indsender igen.');
    return;
  }

  const poster = readPoster();
  const sum = poster.reduce((a, p) => a + p.beløb_dkk, 0);

  const body = {
    kortniveau: document.getElementById('kortniveau').value,
    beløb_dkk: sum,
    hændelse_beskrivelse: document.getElementById('beskrivelse').value.trim(),
    poster: poster,
    hændelse_dato: document.getElementById('haendelse_dato').value,
    rejse_startdato: document.getElementById('rejse_start').value || null,
    rejse_slutdato: document.getElementById('rejse_slut').value || null,
    dokumentation: document.getElementById('dokumentation').value.trim() || null
  };

  if (!body.kortniveau || body.hændelse_beskrivelse.length < 10 || !body.hændelse_dato) {
    showError('Udfyld kortniveau, hændelsesdato og en beskrivelse på mindst 10 tegn.');
    return;
  }
  if (poster.length === 0) {
    showError('Tilføj mindst én udgiftspost.');
    return;
  }
  if (poster.some(p => p.beskrivelse.length < 3 || p.beløb_dkk <= 0)) {
    showError('Hver udgiftspost skal have en beskrivelse og et beløb over 0.');
    return;
  }

  setLoading(true);
  recordSubmit();

  try {
    const resp = await fetch('/api/claims', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `Serverfejl (${resp.status})`);
    }

    const result = await resp.json();
    showResult(body.kortniveau, result);
  } catch (e) {
    if (e instanceof TypeError) {
      showError('Kan ikke nå serveren. Kører backenden? (uvicorn app.api:app)');
    } else {
      showError(e.message);
    }
  } finally {
    setLoading(false);
  }
}

function setLoading(on) {
  const btn = document.getElementById('submitBtn');
  btn.querySelector('.btn-text').style.display = on ? 'none' : 'inline-flex';
  btn.querySelector('.btn-loading').style.display = on ? 'inline-flex' : 'none';
  btn.disabled = on;

  const loadingText = document.getElementById('loadingText');
  clearInterval(loadingInterval);
  if (on) {
    let step = 0;
    loadingText.textContent = LOADING_STEPS[0];
    loadingInterval = setInterval(() => {
      step = Math.min(step + 1, LOADING_STEPS.length - 1);
      loadingText.textContent = LOADING_STEPS[step];
    }, 4000);
  }
}

function showError(msg) {
  const el = document.getElementById('formError');
  el.textContent = msg;
  el.style.display = 'block';
}

function hideError() {
  document.getElementById('formError').style.display = 'none';
}

// ── Result rendering ────────────────────────────────────────────────────

function dkk(n) {
  return (n ?? 0).toLocaleString('da-DK', { minimumFractionDigits: 0, maximumFractionDigits: 2 }) + ' DKK';
}

function showResult(kort, r) {
  document.getElementById('formView').style.display = 'none';
  document.getElementById('resultView').style.display = 'block';

  const cfg = statusConfig[r.afgørelse] || statusConfig.manuelt_review;

  const card = document.getElementById('resultCard');
  card.setAttribute('data-status', r.afgørelse);

  const iconWrap = document.getElementById('resultIconWrap');
  iconWrap.setAttribute('data-status', r.afgørelse);
  document.getElementById('resultIcon').className = 'ti ' + cfg.icon;
  document.getElementById('resultStatus').textContent = cfg.title;
  document.getElementById('resultId').textContent =
    `Krav ${r.krav_id}` + (r.behandlingstid_sek ? ` · behandlet på ${r.behandlingstid_sek} sek.` : '');

  // Delkrav-liste
  const wrap = document.getElementById('delkravWrap');
  const list = document.getElementById('delkravList');
  list.textContent = '';
  if (r.delafgørelser && r.delafgørelser.length > 0) {
    wrap.style.display = 'block';
    for (const d of r.delafgørelser) {
      list.appendChild(renderDelkrav(d));
    }
  } else {
    wrap.style.display = 'none';
  }

  // Kundebesked (preformatteret tekst fra backend — vis linjeskift)
  document.getElementById('resultKundebesked').textContent = r.kundebesked || r.begrundelse;

  document.getElementById('resultApproved').textContent = dkk(r.godkendt_beløb_dkk);
  document.getElementById('resultAmount').textContent = dkk(r.ansøgt_beløb_dkk);
  document.getElementById('resultConf').textContent = Math.round((r.konfidens ?? 0) * 100) + '%';
  document.getElementById('resultType').textContent =
    r.afgørelse === 'manuelt_review' ? 'Manuel' : 'Automatisk (STO)';

  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function renderDelkrav(d) {
  // Bygges med DOM-API'er (ikke innerHTML) — beskrivelser/begrundelser
  // indeholder tekst der oprindeligt kommer fra brugerinput og LLM
  const row = document.createElement('div');
  row.className = 'delkrav-item';
  row.setAttribute('data-status', d.afgørelse);

  const icon = document.createElement('div');
  icon.className = 'delkrav-icon';
  icon.innerHTML = `<i class="ti ${delStatusIcon[d.afgørelse] || 'ti-clock'}" aria-hidden="true"></i>`;

  const main = document.createElement('div');
  main.className = 'delkrav-main';

  const top = document.createElement('div');
  top.className = 'delkrav-top';

  const desc = document.createElement('span');
  desc.className = 'delkrav-desc';
  desc.textContent = d.beskrivelse;

  const amount = document.createElement('span');
  amount.className = 'delkrav-amount';
  if (d.afgørelse === 'godkendt' && d.godkendt_beløb_dkk != null) {
    amount.textContent = dkk(d.godkendt_beløb_dkk);
  } else if (d.beløb_dkk != null) {
    amount.textContent = dkk(d.beløb_dkk);
  }

  top.appendChild(desc);
  top.appendChild(amount);

  const reason = document.createElement('div');
  reason.className = 'delkrav-reason';
  reason.textContent = d.begrundelse;

  const meta = document.createElement('div');
  meta.className = 'delkrav-meta';
  const metaParts = [d.dækningstype.replace(/_/g, ' ')];
  if (d.metadata_filtreret) metaParts.push('afgjort uden AI-vurdering');
  meta.textContent = metaParts.join(' · ');

  main.appendChild(top);
  main.appendChild(reason);
  main.appendChild(meta);

  if (d.omklassificeret_fra) {
    const badge = document.createElement('span');
    badge.className = 'delkrav-reclass';
    badge.innerHTML = '<i class="ti ti-arrows-exchange" aria-hidden="true"></i>';
    badge.appendChild(document.createTextNode(
      ' Omklassificeret fra ' + d.omklassificeret_fra.replace(/_/g, ' ')
    ));
    main.appendChild(badge);
  }

  row.appendChild(icon);
  row.appendChild(main);
  return row;
}

function goBack() {
  document.getElementById('resultView').style.display = 'none';
  document.getElementById('formView').style.display = 'block';
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
