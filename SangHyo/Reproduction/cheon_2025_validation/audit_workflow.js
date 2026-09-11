export const meta = {
  name: 'cheon-existing-code-audit',
  description: 'Read each existing Cheon-2025 implementation, extract its protocol with evidence, then adversarially verify every claimed deviation',
  phases: [{ title: 'Read', detail: 'one reader per existing implementation' }, { title: 'Verify', detail: 'one skeptic per implementation re-checks each claim against the code' }],
}
const ROOT = '/Users/pig30nidae/Pig30nidaE/AJOU/26-1-summer/Google-AJOU-AI-Capstone'
const REF = ROOT + '/SangHyo/Reproduction/cheon_2025_validation'
const IMPLS = [
  { key: 'taehyun_paper_reproduction', path: ROOT + '/Taehyun/Paper_Reproduction', hint: 'Cheon_LGBM.py and README.md (ignore Choi_Ensemble.py / KimPark_LR.py which reproduce other papers)' },
  { key: 'taehyun_paper_reproduction_nested', path: ROOT + '/Taehyun/Paper_Reproduction_Nested', hint: 'Cheon_LGBM.py and README.md (ignore the Choi / KimPark scripts)' },
  { key: 'sanghyo_xai_paper_reproduction', path: ROOT + '/SangHyo/Reproduction/XAI_Paper_Reproduction', hint: 'src/xai_paper_reproduction.py, scripts/0*.py, XAI_Paper_Reproduction_Colab.py, PAPER_IMPLEMENTATION_NOTES.md, README.md, REPRODUCTION_DETAILS_KO.md, outputs/reproduction_report.md, outputs/reproduction_audit.json' },
  { key: 'sanghyo_xai_paper_reproduction2', path: ROOT + '/SangHyo/Reproduction/XAI_Paper_Reproduction2', hint: 'Training/XAI_Paper_Reproduction2_PaperExact_Colab.py (the .ipynb is the same code), README_KO.md, RESULTS_REVIEW_KO.md' },
  { key: 'sanghyo_binary_paperlgbm_nommse', path: ROOT + '/SangHyo/Binary/Binary_PaperLGBM_NoMMSE', hint: 'paperlib.py and every other .py plus README_KO.md; this is a subject-level (leakage-free) re-run of the paper features, so classify it as such' },
  { key: 'taehyun_xai_dirs', path: ROOT + '/Taehyun/xai and ' + ROOT + '/Taehyun/previous/xai', hint: 'first list the files; decide whether these implement Cheon et al. 2025 (SHAP/LightGBM dementia risk) at all; if not, say so and keep the rest short' },
]
const READ_SCHEMA = {
  type: 'object',
  properties: {
    implementation: { type: 'string' },
    is_cheon_reproduction: { type: 'boolean' },
    entry_files: { type: 'array', items: { type: 'string' } },
    cohort: { type: 'string' },
    record_count: { type: 'string' },
    label: { type: 'string' },
    record_construction: { type: 'string' },
    missing_values: { type: 'string' },
    features: { type: 'string' },
    feature_selection: { type: 'string' },
    split: { type: 'string' },
    model_and_hyperparameters: { type: 'string' },
    statistics: { type: 'string' },
    reported_results: { type: 'string' },
    deviations_from_paper: { type: 'array', items: { type: 'object', properties: {
      item: { type: 'string' }, existing_choice: { type: 'string' }, paper_or_protocol_says: { type: 'string' },
      evidence: { type: 'string' }, severity: { type: 'string', enum: ['high', 'medium', 'low'] } },
      required: ['item', 'existing_choice', 'paper_or_protocol_says', 'evidence', 'severity'] } },
    potential_errors: { type: 'array', items: { type: 'object', properties: {
      description: { type: 'string' }, evidence: { type: 'string' }, confidence: { type: 'string', enum: ['high', 'medium', 'low'] } },
      required: ['description', 'evidence', 'confidence'] } },
    differences_vs_new_implementation: { type: 'array', items: { type: 'string' } },
  },
  required: ['implementation', 'is_cheon_reproduction', 'entry_files', 'cohort', 'record_count', 'label', 'record_construction',
    'missing_values', 'features', 'feature_selection', 'split', 'model_and_hyperparameters', 'statistics', 'reported_results',
    'deviations_from_paper', 'potential_errors', 'differences_vs_new_implementation'],
}
const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    verdicts: { type: 'array', items: { type: 'object', properties: {
      claim: { type: 'string' }, verdict: { type: 'string', enum: ['confirmed', 'refuted', 'unclear'] },
      reason: { type: 'string' }, evidence: { type: 'string' } },
      required: ['claim', 'verdict', 'reason', 'evidence'] } },
    missed_items: { type: 'array', items: { type: 'string' } },
  },
  required: ['verdicts', 'missed_items'],
}
const readerPrompt = (impl) => `You are auditing an EXISTING implementation related to Cheon et al. (2025), "A Study on Dementia Risk Assessment Using Lifelog Data with Explainable AI" (JKIIE 51(2):161-170), inside a research repository. STRICT RULES: read-only; never modify any file; never run scripts, notebooks or Python; never open anything under ${ROOT}/Data (raw subject data); never print e-mail addresses or subject identifiers you may encounter in code comments or outputs.

Reference (paper-derived, written independently of any existing code):
- ${REF}/PAPER_PROTOCOL.md  (what the paper states, with page refs; the 72-feature list; Table 1/2 values; 5-fold CV; SHAP forward selection; tuned AUC 0.9492)
- ${REF}/ASSUMPTIONS.md     (choices the new implementation made where the paper is silent, IDs A01-A26)
Read both first.

Target implementation: ${impl.key}
Location: ${impl.path}
Files to read: ${impl.hint}. Read the code completely, not just headers. Use Read/Grep; use Bash only for ls/find/wc/grep.

Extract, each with file:line evidence, how the existing implementation handles:
1. cohort: which AI-Hub partitions (Training/Validation) it uses, number of subjects, and whether MMSE/cognitive files are read;
2. record_count: how many daily records it ends up with and how that number arises (merge/dedup/filters);
3. label: class mapping (CN vs MCI+Dem? other?);
4. record_construction: how activity and sleep rows are paired (row-wise, date join, other) and duplicate handling;
5. missing_values: imputation / dropping / NaN passthrough, and where it is fit (whole dataset vs training fold);
6. features: how many features, which of the paper's 72 are present/absent, extra features not in the paper, and the exact definitions of sleep_time, bedtime clock conversion, the 1-min MET statistics (ddof, kurtosis type, autocorrelation lag, quantiles), 5-min class counts and hypnogram counts;
7. feature_selection: SHAP explainer type, importance aggregation, forward-selection procedure, on which data the ranking is computed (all data / training fold / inner fold), k chosen;
8. split: record-level vs subject-level, KFold/StratifiedKFold/GroupKFold/StratifiedGroupKFold, shuffle, seeds, repeats, any hold-out (e.g. the 33-subject Validation), nested or not;
9. model_and_hyperparameters: model(s), library defaults vs tuned values, whether the paper's Table 2 values (min_data_in_leaf 41, num_leaves 330, n_estimators 1000, learning_rate 0.08) are used, any grid search;
10. statistics: how fold/repeat scores are aggregated (pooled folds vs repeat means), CI method, threshold;
11. reported_results: the numbers the implementation reports (README/report/outputs), e.g. ROC-AUC values, and which stage they correspond to.
Then list deviations_from_paper (things that contradict PAPER_PROTOCOL.md or silently add assumptions not in the paper), potential_errors (bugs or leakage: e.g. feature selection or imputation fit on data that includes evaluation rows, subject overlap across folds, label-derived inputs, wrong class mapping), and differences_vs_new_implementation (vs the choices in ASSUMPTIONS.md). Be concrete and cite file:line for every claim. If the target is not a Cheon reproduction at all, set is_cheon_reproduction=false, explain briefly, and keep other fields short.`
const verifierPrompt = (impl, r) => `You are a skeptical second reviewer. RULES: read-only; never modify files; never run code; never open ${ROOT}/Data; never print e-mail addresses or subject identifiers.
A first reader audited the existing implementation "${impl.key}" at ${impl.path} (files: ${impl.hint}) against the paper-derived reference ${REF}/PAPER_PROTOCOL.md and ${REF}/ASSUMPTIONS.md.
Below are the reader's claims. For EACH claim in deviations_from_paper and potential_errors, re-open the cited file/lines yourself and decide: confirmed (code really does this), refuted (code does not do this, or the reader misread), or unclear (cannot be determined from the code). Try hard to refute; default to 'unclear' rather than 'confirmed' if the evidence is not in the code. Quote the decisive line(s) as evidence. Also list missed_items: important protocol facts (cohort size, record count, split type, seeds, hyperparameters, feature count, SHAP scope, statistics) that the reader did not report or reported without evidence, after checking the code yourself.
Reader's structured report:
${JSON.stringify(r, null, 2)}`
phase('Read')
const results = await pipeline(
  IMPLS,
  impl => agent(readerPrompt(impl), { label: `read:${impl.key}`, phase: 'Read', schema: READ_SCHEMA }),
  (r, impl) => r ? agent(verifierPrompt(impl, r), { label: `verify:${impl.key}`, phase: 'Verify', schema: VERIFY_SCHEMA }).then(v => ({ impl: impl.key, path: impl.path, reader: r, verify: v })) : null,
)
const out = results.filter(Boolean)
log(`audited ${out.length}/${IMPLS.length} implementations`)
return out