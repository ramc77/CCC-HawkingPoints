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

// -----------------------------------------------------------
// COVER LETTER
// -----------------------------------------------------------
const coverBody = [
  p('Ram Chand', { alignment: AlignmentType.RIGHT, after: 0 }),
  p('Department of Natural Sciences', { alignment: AlignmentType.RIGHT, after: 0 }),
  p('The Begum Nusrat Bhutto Women University', { alignment: AlignmentType.RIGHT, after: 0 }),
  p('Sukkur, Sindh, Pakistan', { alignment: AlignmentType.RIGHT, after: 0 }),
  p('ram.chand2k11@yahoo.com', { alignment: AlignmentType.RIGHT, after: 240 }),

  p(new Date().toISOString().slice(0, 10), { after: 240 }),

  p('The Editors', { after: 0 }),
  p('Journal of Cosmology and Astroparticle Physics (JCAP)', { after: 240 }),

  p('Dear Editors,', { after: 240 }),

  p('I am submitting for your consideration the manuscript "An interpretable, trials-corrected machine-learning search for Penrose Hawking points in the Planck 2018 cosmic microwave background" for publication in JCAP.'),

  p('The paper addresses a specific gap in the CMB anomaly-search literature. Roger Penrose\'s conformal cyclic cosmology predicts that supermassive black holes evaporating during the previous cosmic aeon should leave small circular temperature imprints on the CMB of the current aeon, called Hawking points. The prior searches for these features — the ring-variance search of Gurzadyan and Penrose (2010), the Gaussian-template search of An, Meissner, Nurowski and Penrose (2020), and the supervised deep-learning search HawkingNet (Bodnia et al., 2024) — each suffer from at least one of three co-occurring methodological limitations: they assume a specific radial signal template, they deliver decisions that cannot be inspected physically, or they do not calibrate against the look-elsewhere effect properly. The Bayesian re-analysis of Jow and Scott (2020) showed that once the trials factor is folded in, the earlier claims dissolve into the Gaussian LambdaCDM null.'),

  p('The pipeline I present in this paper is designed to remove all three limitations at once. It is template-free: I learn the distribution of physically-named CMB patch features (a ring-variance profile, spherical wavelet coefficients, Minkowski excursion-set functionals, local moments) under a Gaussian LambdaCDM null from 1000 simulated skies, with no assumption about the anomaly shape. It is interpretable: the learned anomaly score is reduced to a compact linear combination of the input features by ridge regression, so the machine\'s decision is directly readable and can be compared to the classical Gurzadyan-Penrose ring-variance statistic. And it is trials-corrected from the start: the calibration null is the empirical distribution of the sky-wide maximum of the compact statistic across the null ensemble, and the trials-corrected p-values pass a Kolmogorov-Smirnov uniformity check on 200 independent Gaussian control simulations (p_KS = 0.48).'),

  p('Three findings from the paper are, I believe, of direct interest to the JCAP readership. First, the compact anomaly statistic independently recovers the classical Gurzadyan-Penrose ring-variance ratio as one of its two dominant terms — a direct internal validation of the pipeline against the field\'s existing physical intuition. Second, the pipeline promotes the Minkowski V1 perimeter of the plus and minus one sigma excursion sets as a new co-dominant channel, not used in the prior CCC anomaly literature; I propose it as a physically-motivated statistic for future template-free CMB anomaly searches. Third, applied to the four Planck 2018 component-separation maps under the official Planck common confidence mask, the pipeline finds one cross-pipeline consistent above-threshold cluster at galactic coordinates (l, b) close to (133 degrees, minus 24 degrees). This cluster\'s morphology is asymmetric and does not match the AMNP-2018 template; a Planck 353 GHz thermal-dust cross-check finds the candidate direction is 0.57 times the median dust brightness and 0.49 times the median dust variability of same-galactic-latitude controls, ruling out a shared foreground residual. The cluster is therefore neither a Hawking-point detection nor a foreground residual; its physical origin remains open between a rare Gaussian LambdaCDM null-tail configuration and a genuinely anomalous CMB feature, requiring higher-N calibration to resolve.'),

  p('I report the pipeline\'s overall result as a calibrated non-detection of AMNP-2018-shape Hawking points above approximately 80 microkelvin at approximately one-degree angular scale in Planck PR3 data. The paper is written to state this clearly and to state honestly the sensitivity floor of the current analysis.'),

  p('The manuscript is approximately 5500 words and contains three figures and six tables. All code, configuration files, and analysis outputs that underpin the paper are publicly available at https://github.com/ramc77/CCC-HawkingPoints, so that any reader can reproduce the calibration certification, the sensitivity curve, the distilled statistic, and the cross-pipeline candidate matrix.'),

  p('I confirm that this manuscript has not been published elsewhere and is not under consideration by any other journal. I have no competing interests to declare, and no external funding sources to acknowledge beyond the institutional support of The Begum Nusrat Bhutto Women University.'),

  p('Thank you for your time and consideration.', { after: 320 }),

  p('Sincerely,', { after: 320 }),

  p('Ram Chand', { after: 0 }),
  p('Department of Natural Sciences', { after: 0 }),
  p('The Begum Nusrat Bhutto Women University', { after: 0 }),
  p('Sukkur, Sindh, Pakistan', { after: 0 }),
  p('ram.chand2k11@yahoo.com'),
];

const coverDoc = new Document({
  styles: { default: { document: { run: defaultRun } } },
  sections: [{ properties: { page: pageSetup }, children: coverBody }],
});

// -----------------------------------------------------------
// PORTAL ABSTRACT (structured, matches paper abstract)
// -----------------------------------------------------------
const abstractBody = [
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 240 },
    children: [new TextRun({
      text: 'An interpretable, trials-corrected machine-learning search for Penrose Hawking points in the Planck 2018 cosmic microwave background',
      bold: true, font: 'Times New Roman', size: 24,
    })],
  }),

  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 400 },
    children: [new TextRun({
      text: 'Ram Chand, Department of Natural Sciences, The Begum Nusrat Bhutto Women University, Sukkur, Sindh, Pakistan',
      italics: true, font: 'Times New Roman', size: 22,
    })],
  }),

  heading('Abstract'),

  new Paragraph({
    spacing: { after: 200 },
    children: [
      new TextRun({ text: 'Context. ', bold: true, font: 'Times New Roman', size: 22 }),
      new TextRun({ text: "Roger Penrose's conformal cyclic cosmology (CCC) predicts that supermassive black holes evaporating during the previous cosmic aeon leave small, quasi-circular temperature imprints on the CMB of the current aeon — Hawking points — of order one degree in angular size and tens of microkelvin in central amplitude. The original ring-variance search of Gurzadyan and Penrose and the Gaussian-template search of An, Meissner, Nurowski and Penrose reported candidate detections, but Jow and Scott showed the significances dissolve into the Gaussian LambdaCDM null under a proper look-elsewhere correction.", font: 'Times New Roman', size: 22 }),
    ],
  }),

  new Paragraph({
    spacing: { after: 200 },
    children: [
      new TextRun({ text: 'Aims. ', bold: true, font: 'Times New Roman', size: 22 }),
      new TextRun({ text: 'We identify three co-occurring limitations of the existing Hawking-point search literature — template dependence, the absence of an inspectable decision rule, and inadequate trials-factor calibration — and build a search pipeline that addresses all three simultaneously.', font: 'Times New Roman', size: 22 }),
    ],
  }),

  new Paragraph({
    spacing: { after: 200 },
    children: [
      new TextRun({ text: 'Methods. ', bold: true, font: 'Times New Roman', size: 22 }),
      new TextRun({ text: 'For each candidate sky patch we compute a 40-dimensional vector of physically-named features: a ring-variance profile, spherical wavelet coefficients, Minkowski functionals of local excursion sets, and local moments. A whitened multivariate Gaussian density model trained on 1000 Gaussian LambdaCDM null simulations assigns an anomaly score. We reduce this score to a compact linear combination of the input features by ridge regression (symbolic distillation), and calibrate its significance against the empirical distribution of the sky-wide maximum of the compact statistic across the null ensemble, absorbing the look-elsewhere effect by construction. Candidates are required to appear consistently in all four Planck component-separation maps and to pass a robust extreme-pixel scrub.', font: 'Times New Roman', size: 22 }),
    ],
  }),

  new Paragraph({
    spacing: { after: 200 },
    children: [
      new TextRun({ text: 'Results. ', bold: true, font: 'Times New Roman', size: 22 }),
      new TextRun({ text: 'The trials-corrected p-values are uniform on 200 held-out Gaussian control simulations (Kolmogorov-Smirnov p = 0.48). The compact statistic recovers the classical Gurzadyan-Penrose ring-variance ratio as one of its dominant terms, and identifies the Minkowski perimeter V1 of plus and minus one sigma excursion sets as a new co-dominant channel. Under the official Planck common confidence mask, no candidate exceeds the trials-corrected alpha = 0.01 threshold in more than one of the four maps except a single cluster at galactic coordinates (l, b) approximately (133 degrees, minus 24 degrees). Its morphology is asymmetric and does not match the AMNP-2018 template. A Planck 353 GHz thermal-dust cross-check finds the candidate direction is 0.57 times the median dust brightness and 0.49 times the median dust variability of same-galactic-latitude controls, ruling out a shared foreground residual.', font: 'Times New Roman', size: 22 }),
    ],
  }),

  new Paragraph({
    spacing: { after: 200 },
    children: [
      new TextRun({ text: 'Conclusions. ', bold: true, font: 'Times New Roman', size: 22 }),
      new TextRun({ text: 'We report a calibrated non-detection of AMNP-2018-shape Hawking points above about 80 microkelvin at approximately one-degree angular scale in Planck PR3 data. The physical origin of the cluster at (l, b) approximately (133 degrees, minus 24 degrees) is unresolved between a rare Gaussian LambdaCDM null-tail configuration and a genuinely anomalous CMB feature; higher-N calibration is required to distinguish these.', font: 'Times New Roman', size: 22 }),
    ],
  }),

  heading('Keywords'),
  p('cosmic microwave background; cosmology: observations; cosmology: theory; methods: statistical; methods: data analysis; conformal cyclic cosmology; Hawking points; look-elsewhere effect; symbolic distillation.'),
];

const abstractDoc = new Document({
  styles: { default: { document: { run: defaultRun } } },
  sections: [{ properties: { page: pageSetup }, children: abstractBody }],
});

// -----------------------------------------------------------
// Serialise
// -----------------------------------------------------------
const outDir = path.join(__dirname);
async function main() {
  const coverBuf = await Packer.toBuffer(coverDoc);
  fs.writeFileSync(path.join(outDir, 'JCAP_cover_letter.docx'), coverBuf);
  const abstractBuf = await Packer.toBuffer(abstractDoc);
  fs.writeFileSync(path.join(outDir, 'JCAP_portal_abstract.docx'), abstractBuf);
  console.log('Wrote:');
  console.log(' - ' + path.join(outDir, 'JCAP_cover_letter.docx'));
  console.log(' - ' + path.join(outDir, 'JCAP_portal_abstract.docx'));
}
main();
