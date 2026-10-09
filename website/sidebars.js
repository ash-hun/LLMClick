/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
module.exports = {
  docs: [
    {type: 'category', label: 'Getting started', collapsed: false, items: ['index', 'quick-start', 'project-structure']},
    {type: 'category', label: 'Concepts', items: ['concepts/how-a-run-works', 'concepts/stages', 'concepts/idempotency']},
    {type: 'category', label: 'Modeling', items: ['modeling/recipes', 'modeling/decision-recipes', 'modeling/extending']},
    {
      type: 'category',
      label: 'Evaluation',
      items: [
        'evaluation/overview', 'evaluation/custom', 'evaluation/benchmark', 'evaluation/compare', 'evaluation/decision',
        'evaluation/embedding', 'evaluation/contamination', 'evaluation/design', 'evaluation/survey',
      ],
    },
    {
      type: 'category',
      label: 'Data',
      items: ['data/overview', 'data/synthetic', 'data/practices', 'data/lineages', 'data/design', 'data/survey'],
    },
    {
      type: 'category',
      label: 'Reference',
      items: ['reference/recipes', 'reference/config', 'reference/api', 'reference/cli', 'reference/settings'],
    },
    {type: 'category', label: 'Operations', items: ['operations/jobs', 'operations/security']},
    {type: 'category', label: 'Contributing', items: ['contributing/development']},
  ],
};
