/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
// Each channel opens with its overview (what it covers and how it flows), then goes down to one page per recipe.
const family = (label, overview, items) => ({type: 'category', label, link: {type: 'doc', id: overview}, items});

module.exports = {
  docs: [
    {type: 'category', label: 'Getting started', collapsed: false, items: ['index', 'quick-start', 'project-structure']},
    {type: 'category', label: 'Concepts', items: ['concepts/how-a-run-works', 'concepts/stages', 'concepts/idempotency']},
    {
      type: 'category',
      label: 'Modeling',
      items: [
        'modeling/overview',
        family('LLM recipes', 'modeling/llm/overview', ['modeling/llm/sft', 'modeling/llm/instruction', 'modeling/llm/dpo', 'modeling/llm/grpo']),
        family('Decision recipes', 'modeling/decision/overview', ['modeling/decision/sft', 'modeling/decision/cispo']),
        'modeling/embedding/contrastive',
        family('Speech recipes', 'modeling/speech/overview', ['modeling/speech/tts-piper', 'modeling/speech/tts-qwen', 'modeling/speech/wakeword']),
        'modeling/extending',
      ],
    },
    {
      type: 'category',
      label: 'Evaluation',
      items: [
        'evaluation/overview', 'evaluation/benchmark', 'evaluation/llm-performance', 'evaluation/custom', 'evaluation/compare', 'evaluation/decision',
        'evaluation/embedding', 'evaluation/wakeword', 'evaluation/contamination', 'evaluation/design', 'evaluation/survey',
      ],
    },
    {
      type: 'category',
      label: 'Data',
      items: ['data/overview', 'data/synthetic', 'data/speech', 'data/practices', 'data/lineages', 'data/design', 'data/survey'],
    },
    {
      type: 'category',
      label: 'Reference',
      items: ['reference/recipes', 'reference/config', 'reference/api', 'reference/cli', 'reference/settings'],
    },
    {
      type: 'category',
      label: 'Experiments',
      items: [
        'experiments/overview',
        family('Decoder LLM', 'experiments/decoder-llm/overview', ['experiments/decoder-llm/jarvis-router']),
        family('Hybrid LLM (Gated DeltaNet)', 'experiments/hybrid-llm/overview', []),
        family('Speech models', 'experiments/speech/overview', ['experiments/speech/jarvis-agent']),
      ],
    },
    {type: 'category', label: 'Operations', items: ['operations/jobs', 'operations/security']},
    {type: 'category', label: 'Contributing', items: ['contributing/development']},
  ],
};
