# server/data/

This folder is reserved for optional static JSON datasets.

If you want to add hand-crafted issues beyond what issue_generator.py produces,
place them here as JSON files and load them in issue_generator.py.

Format for each file (e.g. extra_bugs.json):

[
  {
    "title": "...",
    "body": "...",
    "correct_label": "label_bug",
    "correct_severity": "P1",
    "has_stack_trace": false,
    "components": ["auth"]
  }
]

This folder is empty in Phase 1. It is optional for the final submission.
