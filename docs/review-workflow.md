# Content Review and Approval Workflow

This document outlines the workflow for reviewing and approving content in the Humanoid Robotics & Physical AI Course Book.

## Review Process Overview

All content must go through a structured review process before being approved for inclusion in the course book. This ensures technical accuracy, educational effectiveness, and adherence to the required standards.

## Review Stages

### Stage 1: Self-Review
Before submitting content for formal review, authors must complete the following self-review checklist:

- [ ] Chapter follows required structure: overview → theory → implementation → examples → applications
- [ ] All technical claims have numbered citations to authoritative sources
- [ ] Code examples are accurate and functional
- [ ] Content is appropriate for intermediate-advanced engineering audience
- [ ] Exercises are meaningful and achievable
- [ ] All validation scripts pass (citation-validator.js, cross-ref-validator.js)
- [ ] No plagiarism - all sourced content properly attributed and rewritten

### Stage 2: Peer Review
Once self-review is complete, the content moves to peer review:

1. **Technical Accuracy Review**
   - Domain expert reviews all technical claims and concepts
   - Validates code examples in appropriate environments
   - Ensures hardware recommendations are current and accurate

2. **Educational Effectiveness Review**
   - Education specialist assesses learning objectives and progression
   - Validates exercises and practical applications
   - Ensures content maintains appropriate audience level

3. **Citation and Reference Review**
   - Verifies all citations point to authoritative sources
   - Checks citation format follows simple numbered reference style
   - Confirms proper attribution and no plagiarism

4. **Docusaurus Compatibility Review**
   - Validates Markdown formatting
   - Checks navigation structure and cross-references
   - Tests build and deployment process

### Stage 3: Final Approval
After successful peer review, content receives final approval:

- Lead editor reviews all feedback and ensures all issues are resolved
- Final quality check against course objectives
- Approval for inclusion in the course book

## Review Tools

The following tools should be used during the review process:

- **Citation Validator**: `node scripts/citation-validator.js`
- **Cross-Reference Validator**: `node scripts/cross-ref-validator.js`
- **Plagiarism Detector**: `node scripts/plagiarism-detector.js`
- **Quality Checklist**: Use the template in `docs/checklist-template.md`

## Review Responsibilities

- **Authors**: Responsible for initial content creation and self-review
- **Technical Reviewers**: Domain experts who validate technical accuracy
- **Education Reviewers**: Specialists who assess educational effectiveness
- **Editor**: Final approval authority and quality gatekeeper

## Resolution Process

If issues are identified during review:

1. Reviewer documents specific issues with clear recommendations
2. Author addresses all issues and resubmits for review
3. Only critical issues require full re-review; minor issues can be approved after correction
4. All changes must pass validation scripts before final approval

## Timeline

- Self-review: Author completes as part of writing process
- Peer review: 3-5 business days per review cycle
- Final approval: 1-2 business days after all issues are resolved

## Quality Gates

Content must meet all of the following requirements to be approved:

- Technical accuracy verified by domain expert
- Educational effectiveness confirmed by specialist
- All validation scripts pass without errors
- Proper citations for all technical claims
- Appropriate audience level maintained
- Required chapter structure followed
- No plagiarism detected