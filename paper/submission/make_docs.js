const {
  Document, Packer, Paragraph, TextRun, AlignmentType, HeadingLevel,
} = require('docx');
const fs = require('fs');
const path = require('path');

// -----------------------------------------------------------
// Shared style: 11pt Times New Roman, single spacing.
// docx-js uses half-points; 22 = 11pt.
// Paragraph spacing: 240 twips before = ~12pt gap.
// -----------------------------------------------------------
const defaultRun = { font: 'Times New Roman', size: 22 };
const pageSetup = {
  size: { width: 12240, height: 15840 },
  margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
};

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after ?? 200 },
    alignment: opts.alignment ?? AlignmentType.LEFT,
    children: [new TextRun({ text, bold: opts.bold ?? false, italics: opts.italics ?? false, font: 'Times New Roman', size: 22 })],
  });
}

function heading(text) {
  return new Paragraph({
    spacing: { before: 240, after: 200 },
    children: [new TextRun({ text, bold: true, font: 'Times New Roman', size: 24 })],
  });
}

const TITLE = "An interpretable, trials-corrected machine-learning search for Penrose Hawking points in the Planck 2018 cosmic microwave background";

const ABSTRACT_TEXT = "Roger Penrose's conformal cyclic cosmology (CCC) predicts that supermassive black holes evaporating during a previous cosmic aeon leave small, quasi-circular temperature imprints on the cosmic microwave background (CMB) of the present aeon — “Hawking points” — of order one degree in angular size and tens of microkelvin in central amplitude. Existing searches assumed a specific radial signal template or used a supervised classifier whose decisions could not be inspected physically, and none folded in the look-elsewhere effect from the outset; Jow and Scott subsequently showed that the earlier significances collapse into the Gaussian ΛCDM null under a proper trials-factor correction. Here we remove these three limitations simultaneously. For every candidate sky patch we compute a 40-dimensional vector of physically-named features (a ring-variance profile, spherical wavelet coefficients, Minkowski excursion-set functionals, and local moments), train a whitened multivariate Gaussian density model on 1000 Gaussian ΛCDM null simulations, and reduce the learned anomaly score to a closed-form linear combination of the input features by ridge regression. Statistical significance is calibrated against the empirical distribution of the sky-wide maximum of the compact statistic across the null ensemble, absorbing the look-elsewhere effect by construction; the trials-corrected p-values are uniform on 200 held-out control simulations at Kolmogorov–Smirnov p_KS = 0.48. The compact statistic independently recovers the classical Gurzadyan–Penrose ring-variance ratio as one of its dominant terms, and identifies the Minkowski V1 perimeter of ±1σ excursion sets as a new co-dominant channel. Applied to the four Planck 2018 component-separation CMB maps under the official Planck common confidence mask, no candidate exceeds the trials-corrected α = 0.01 threshold in more than one of the four maps except a single cluster at galactic coordinates (l, b) ≈ (133°, −24°). Its asymmetric morphology does not match the AMNP-2018 template, and a Planck 353 GHz thermal-dust cross-check finds the candidate direction is 0.57× the median dust brightness and 0.49× the median dust variability of same-galactic-latitude controls, ruling out a shared foreground residual. We report a calibrated non-detection of AMNP-2018-shape Hawking points above ∼80 μK at ∼1° angular scale in Planck PR3 data; the physical origin of the (l, b) ≈ (133°, −24°) cluster remains open between a rare Gaussian ΛCDM null-tail configuration and a genuinely anomalous CMB feature.";

const KEYWORDS = "cosmic microwave background; cosmology: observations; cosmology: theory; conformal cyclic cosmology; Hawking points; machine learning; methods: statistical; methods: data analysis";

// =============================================================
// JCAP cover letter
// =============================================================
const jcapCoverBody = [
  p('Ram Chand', { alignment: AlignmentType.RIGHT, after: 0 }),
  p('Department of Natural Sciences', { alignment: AlignmentType.RIGHT, after: 0 }),
  p('The Begum Nusrat Bhutto Women University', { alignment: AlignmentType.RIGHT, after: 0 }),
  p('Sukkur, Sindh, Pakistan', { alignment: AlignmentType.RIGHT, after: 0 }),
  p('ram.chand2k11@yahoo.com', { alignment: AlignmentType.RIGHT, after: 240 }),

  p(new Date().toISOString().slice(0, 10), { after: 240 }),

  p('The Editors', { after: 0 }),
  p('Journal of Cosmology and Astroparticle Physics (JCAP)', { after: 240 }),

  p('Dear Editors,', { after: 240 }),

  p(`I am submitting for your consideration the manuscript "${TITLE}" for publication in JCAP.`),

  p('The paper addresses a specific gap in the CMB anomaly-search literature. Roger Penrose\'s conformal cyclic cosmology predicts that supermassive black holes evaporating during the previous cosmic aeon should leave small circular temperature imprints on the CMB of the current aeon, called Hawking points. The prior searches for these features — the ring-variance search of Gurzadyan and Penrose (2010), the Gaussian-template search of An, Meissner, Nurowski and Penrose (2020), and the supervised deep-learning search HawkingNet (Bodnia et al., 2024) — each suffer from at least one of three co-occurring methodological limitations: they assume a specific radial signal template, they deliver decisions that cannot be inspected physically, or they do not calibrate against the look-elsewhere effect properly. The Bayesian re-analysis of Jow and Scott (2020) showed that once the trials factor is folded in, the earlier claims dissolve into the Gaussian ΛCDM null.'),

  p('The pipeline I present is designed to remove all three limitations at once. It is template-free: I learn the distribution of physically-named CMB patch features (a ring-variance profile, spherical wavelet coefficients, Minkowski excursion-set functionals, local moments) under a Gaussian ΛCDM null from 1000 simulated skies, with no assumption about the anomaly shape. It is interpretable: the learned anomaly score is reduced to a compact linear combination of the input features by ridge regression, so the machine\'s decision is directly readable and can be compared to the classical Gurzadyan-Penrose ring-variance statistic. And it is trials-corrected from the start: the calibration null is the empirical distribution of the sky-wide maximum of the compact statistic across the null ensemble, and the trials-corrected p-values pass a Kolmogorov-Smirnov uniformity check on 200 independent Gaussian control simulations (p_KS = 0.48).'),

  p('Three findings are, I believe, of direct interest to the JCAP readership. First, the compact anomaly statistic independently recovers the classical Gurzadyan-Penrose ring-variance ratio as one of its two dominant terms — a direct internal validation of the pipeline against the field\'s existing physical intuition. Second, the pipeline promotes the Minkowski V1 perimeter of the plus and minus one sigma excursion sets as a new co-dominant channel, not used in the prior CCC anomaly literature. Third, applied to the four Planck 2018 component-separation maps under the official Planck common confidence mask, the pipeline finds one cross-pipeline consistent above-threshold cluster at galactic coordinates (l, b) close to (133 degrees, minus 24 degrees). This cluster\'s morphology is asymmetric and does not match the AMNP-2018 template; a Planck 353 GHz thermal-dust cross-check finds the candidate direction is 0.57 times the median dust brightness and 0.49 times the median dust variability of same-galactic-latitude controls, ruling out a shared foreground residual. The cluster is therefore neither a Hawking-point detection nor a foreground residual; its physical origin remains open between a rare Gaussian ΛCDM null-tail configuration and a genuinely anomalous CMB feature, requiring higher-N calibration to resolve.'),

  p('I report the pipeline\'s overall result as a calibrated non-detection of AMNP-2018-shape Hawking points above approximately 80 microkelvin at approximately one-degree angular scale in Planck PR3 data.'),

  p('The manuscript is prepared with the official jcappub JCAP LaTeX class, is approximately 6000 words, and contains three figures and five tables. All code, configuration files, and analysis outputs that underpin the paper are publicly available at https://github.com/ramc77/CCC-HawkingPoints, so that any reader can reproduce the calibration certification, the sensitivity curve, the distilled statistic, and the cross-pipeline candidate matrix.'),

  p('I confirm that this manuscript has not been published elsewhere and is not under consideration by any other journal. I have no competing interests to declare, and no external funding sources to acknowledge beyond the institutional support of The Begum Nusrat Bhutto Women University.'),

  p('Thank you for your time and consideration.', { after: 320 }),

  p('Sincerely,', { after: 320 }),

  p('Ram Chand', { after: 0 }),
  p('Department of Natural Sciences', { after: 0 }),
  p('The Begum Nusrat Bhutto Women University', { after: 0 }),
  p('Sukkur, Sindh, Pakistan', { after: 0 }),
  p('ram.chand2k11@yahoo.com'),
];

const jcapCoverDoc = new Document({
  styles: { default: { document: { run: defaultRun } } },
  sections: [{ properties: { page: pageSetup }, children: jcapCoverBody }],
});

// =============================================================
// PoTDU cover letter
// =============================================================
const potduCoverBody = [
  p('Ram Chand', { alignment: AlignmentType.RIGHT, after: 0 }),
  p('Department of Natural Sciences', { alignment: AlignmentType.RIGHT, after: 0 }),
  p('The Begum Nusrat Bhutto Women University', { alignment: AlignmentType.RIGHT, after: 0 }),
  p('Sukkur, Sindh, Pakistan', { alignment: AlignmentType.RIGHT, after: 0 }),
  p('ram.chand2k11@yahoo.com', { alignment: AlignmentType.RIGHT, after: 240 }),

  p(new Date().toISOString().slice(0, 10), { after: 240 }),

  p('The Editors', { after: 0 }),
  p('Physics of the Dark Universe', { after: 240 }),

  p('Dear Editors,', { after: 240 }),

  p(`I am submitting for your consideration the manuscript "${TITLE}" for publication in Physics of the Dark Universe.`),

  p('The paper is a search for an observational signature of an alternative cosmological model. Roger Penrose\'s conformal cyclic cosmology (CCC) proposes that our universe is one aeon in an infinite cyclic sequence, and predicts that supermassive black holes which evaporated near the end of the previous aeon should leave small circular temperature imprints — "Hawking points" — on the cosmic microwave background of the present aeon. This is a genuinely falsifiable, quantitative prediction of an alternative to standard inflationary cosmology, and is therefore a natural fit for Physics of the Dark Universe\'s scope in cyclic and non-standard cosmological models.'),

  p('The existing searches for this signature — the ring-variance search of Gurzadyan and Penrose (2010), the Gaussian-template search of An, Meissner, Nurowski and Penrose (2020), and the supervised deep-learning classifier HawkingNet (Bodnia et al., 2024) — each assumed a specific signal shape, or delivered decisions that could not be inspected physically, or did not properly calibrate against the statistical look-elsewhere effect of scanning the whole sky. Jow and Scott (2020) showed that once the trials factor is correctly included, the earlier claimed detections are statistically indistinguishable from the standard Gaussian ΛCDM sky.'),

  p('I built a machine-learning pipeline that removes all three of these limitations at once. It learns the statistical distribution of a Gaussian ΛCDM sky directly from 1000 simulated skies, with no assumption about what a Hawking point should look like; it then compresses the learned anomaly score into a short, human-readable formula built from physically-named quantities (a generalised ring-variance statistic, spherical-wavelet coefficients, and Minkowski functionals of the local excursion set), so the result can be checked against physical intuition rather than trusted as a black box; and its statistical significance is calibrated from the outset against the empirical distribution of the sky-wide maximum score across the null simulation ensemble, which is the only calibration that correctly absorbs the look-elsewhere effect. The trials-corrected p-values pass a Kolmogorov-Smirnov uniformity test on 200 independent control simulations (p = 0.48).'),

  p('Two results are, I believe, of particular interest to the Physics of the Dark Universe readership. First, the distilled formula independently rediscovers the classical Gurzadyan-Penrose ring-variance ratio as one of its two dominant terms, without ever being told about it — a strong internal validation — and identifies a previously-unused statistic, the Minkowski V1 perimeter of the plus/minus one sigma excursion set, as a co-dominant channel. Second, applied to all four Planck 2018 component-separation maps, the pipeline finds one cross-pipeline consistent candidate at galactic coordinates (l, b) approximately (133 degrees, minus 24 degrees) whose morphology does not match the predicted Hawking-point shape, and which a Planck 353 GHz thermal-dust cross-check shows is not a foreground residual either (0.57 times and 0.49 times the median dust brightness and variability of same-latitude control directions). The physical origin of this candidate is left as an open question requiring a larger calibration ensemble.'),

  p('The overall result is a calibrated non-detection of Hawking points above approximately 80 microkelvin at the predicted one-degree scale in Planck PR3 data — a template-free, trials-corrected constraint on this specific CCC prediction.'),

  p('The manuscript is prepared with the Elsevier elsarticle class in the numbered-reference format, is approximately 6000 words, and contains three figures and five tables; a Highlights summary is included as the first page of the manuscript file, consistent with journal submission requirements. All code, configuration files, and analysis outputs that underpin the paper are publicly available at https://github.com/ramc77/CCC-HawkingPoints, so that any reader can reproduce the calibration certification, the sensitivity curve, the distilled statistic, and the cross-pipeline candidate matrix.'),

  p('I confirm that this manuscript has not been published elsewhere and is not under consideration by any other journal. I have no competing interests to declare, and no external funding sources to acknowledge beyond the institutional support of The Begum Nusrat Bhutto Women University.'),

  p('Thank you for your time and consideration.', { after: 320 }),

  p('Sincerely,', { after: 320 }),

  p('Ram Chand', { after: 0 }),
  p('Department of Natural Sciences', { after: 0 }),
  p('The Begum Nusrat Bhutto Women University', { after: 0 }),
  p('Sukkur, Sindh, Pakistan', { after: 0 }),
  p('ram.chand2k11@yahoo.com'),
];

const potduCoverDoc = new Document({
  styles: { default: { document: { run: defaultRun } } },
  sections: [{ properties: { page: pageSetup }, children: potduCoverBody }],
});

// =============================================================
// Portal abstracts — single flowing paragraph, matching the
// paper's actual abstract style (not the old structured-block
// Context/Aims/Methods/Results/Conclusions format, which is an
// A&A-specific convention this paper no longer uses).
// =============================================================
function makeAbstractDoc(journalName) {
  const body = [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 240 },
      children: [new TextRun({ text: TITLE, bold: true, font: 'Times New Roman', size: 24 })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 120 },
      children: [new TextRun({
        text: 'Ram Chand, Department of Natural Sciences, The Begum Nusrat Bhutto Women University, Sukkur, Sindh, Pakistan',
        italics: true, font: 'Times New Roman', size: 22,
      })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 400 },
      children: [new TextRun({
        text: `Prepared for submission to ${journalName}`,
        italics: true, font: 'Times New Roman', size: 20,
      })],
    }),
    heading('Abstract'),
    new Paragraph({
      spacing: { after: 200 },
      children: [new TextRun({ text: ABSTRACT_TEXT, font: 'Times New Roman', size: 22 })],
    }),
    heading('Keywords'),
    p(KEYWORDS + '.'),
  ];
  return new Document({
    styles: { default: { document: { run: defaultRun } } },
    sections: [{ properties: { page: pageSetup }, children: body }],
  });
}

const jcapAbstractDoc = makeAbstractDoc('JCAP');
const potduAbstractDoc = makeAbstractDoc('Physics of the Dark Universe');

// -----------------------------------------------------------
// Serialise
// -----------------------------------------------------------
const outDir = path.join(__dirname);
async function main() {
  const writes = [
    ['JCAP_cover_letter.docx', jcapCoverDoc],
    ['JCAP_portal_abstract.docx', jcapAbstractDoc],
    ['POTDU_cover_letter.docx', potduCoverDoc],
    ['POTDU_portal_abstract.docx', potduAbstractDoc],
  ];
  console.log('Wrote:');
  for (const [name, doc] of writes) {
    const buf = await Packer.toBuffer(doc);
    const outPath = path.join(outDir, name);
    fs.writeFileSync(outPath, buf);
    console.log(' - ' + outPath);
  }
}
main();
