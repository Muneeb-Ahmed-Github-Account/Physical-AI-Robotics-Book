# Spec-Kit Plus History Tracking

This document explains how to properly track changes to the Humanoid Robotics & Physical AI Course Book using Spec-Kit Plus conventions.

## Overview

All changes to the course book content must be tracked using Spec-Kit Plus history tracking. This ensures proper version control, change documentation, and maintainability of the educational content.

## Change Documentation Process

### 1. Prompt History Records (PHRs)

For significant changes, create Prompt History Records in the `history/prompts/` directory:

- **Constitution changes**: `history/prompts/constitution/`
- **Feature-specific changes**: `history/prompts/<feature-name>/`
- **General changes**: `history/prompts/general/`

### 2. PHR Format

Each PHR should follow this structure:

```markdown
---
id: [Incremental number]
title: [3-7 word descriptive title]
stage: [spec|plan|tasks|red|green|refactor|explainer|misc|general]
date: YYYY-MM-DD
surface: agent
model: [Model used]
feature: [Feature name or "none"]
branch: [Git branch]
user: [User identifier]
command: [Command used]
labels: ["topic1","topic2",...]
links:
  spec: [URL or null]
  ticket: [URL or null]
  adr: [URL or null]
  pr: [URL or null]
files:
 - [file paths changed]
tests:
 - [test names]
---

## Prompt

[Full user prompt text]

## Response snapshot

[Key assistant output]

## Outcome

[Impact, tests, files, next steps, reflection]

## Evaluation notes

[Failure modes, results, etc.]
```

### 3. Git Commit Guidelines

When committing changes to the course book:

1. **Descriptive commit messages** that explain the educational value of the change
2. **Reference the appropriate spec/plan/task** in the commit message
3. **Include validation results** if applicable

Example commit message:
```
Add ROS 2 fundamentals chapter with complete overview-theory-implementation-example-application structure

- Implements T020-T033 from tasks.md
- Follows required chapter structure
- Includes validated code examples with proper citations
- Ready for peer review per review-workflow.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

### 4. Change Tracking Best Practices

1. **Document significant changes** that affect course structure or content
2. **Track educational improvements** that enhance student learning
3. **Record technical updates** when robotics frameworks or tools change
4. **Maintain backward compatibility** where possible
5. **Update dependencies** and cross-references as needed

### 5. Version Management

- Major changes that affect course structure should increment major version
- Minor content additions or improvements increment minor version
- Bug fixes and corrections increment patch version
- All changes should be documented in change logs

### 6. Integration with Course Development

- Link PHRs to relevant tasks in `tasks.md`
- Reference ADRs when architectural decisions are made
- Update planning documents when course structure changes
- Maintain consistency with original specification goals

## Validation

Before committing changes, ensure:

1. All validation scripts pass (`scripts/citation-validator.js`, `scripts/cross-ref-validator.js`, etc.)
2. Content follows required structure and standards
3. Citations are complete and accurate
4. Cross-references are valid
5. Educational objectives are met