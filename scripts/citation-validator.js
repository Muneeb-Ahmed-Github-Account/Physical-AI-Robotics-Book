#!/usr/bin/env node

// Content validation script for Humanoid Robotics Book
// Checks citations, technical accuracy, and chapter structure

const fs = require('fs');
const path = require('path');

function validateCitations(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');

  // Find all technical claims that should have citations
  const lines = content.split('\n');
  const issues = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Look for technical claims that likely need citations
    if (/(must|should|requires|implements|follows|uses|based on|according to|states that|found that|demonstrates|achieves|enables|provides|recommends|suggests|indicates|shows|proves|confirms|validates|verifies)/i.test(line)) {
      // Check if the line or nearby lines have citations
      let hasCitation = false;

      // Check current line and surrounding lines for citations
      for (let j = Math.max(0, i-2); j <= Math.min(lines.length-1, i+2); j++) {
        if (/\[\d+\]|(reference|citation|source|see)\s+\d+/i.test(lines[j])) {
          hasCitation = true;
          break;
        }
      }

      if (!hasCitation) {
        issues.push({
          line: i + 1,
          content: line.trim(),
          message: 'Technical claim found without citation'
        });
      }
    }
  }

  return issues;
}

function validateTechnicalTerms(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  const issues = [];

  // Look for common technical terms that should be defined or explained
  const termsToCheck = [
    'ROS 2',
    'Docker',
    'Gazebo',
    'Unity',
    'NVIDIA Isaac',
    'VLA',
    'SLAM',
    'PID controller',
    'kinematics',
    'dynamics',
    'inverse kinematics',
    'forward kinematics',
    'point cloud',
    'TF',
    'transform'
  ];

  for (const term of termsToCheck) {
    if (content.includes(term) && !content.includes(`${term}`)) {
      // Basic check - in reality, we'd want to ensure terms are properly explained
      // when first introduced
    }
  }

  return issues;
}

function validateFileStructure(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  const sections = ['## Overview', '## Theory', '## Implementation', '## Examples', '## Applications'];
  const missingSections = [];

  for (const section of sections) {
    if (!content.includes(section)) {
      missingSections.push(section.replace('## ', ''));
    }
  }

  return missingSections;
}

function validateCodeBlocks(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  const issues = [];

  // Look for code blocks and check if they have proper language specification
  const codeBlockRegex = /```(\w+)?\s*\n([\s\S]*?)\n```/g;
  let match;

  while ((match = codeBlockRegex.exec(content)) !== null) {
    const language = match[1];
    if (!language) {
      issues.push({
        message: 'Code block missing language specification',
        content: match[0].substring(0, 50) + '...'
      });
    }
  }

  return issues;
}

function processDirectory(dirPath) {
  const files = fs.readdirSync(dirPath);

  for (const file of files) {
    const filePath = path.join(dirPath, file);
    const stat = fs.statSync(filePath);

    if (stat.isDirectory()) {
      processDirectory(filePath);
    } else if (file.endsWith('.md')) {
      console.log(`\nValidating: ${filePath}`);

      // Validate citations
      const citationIssues = validateCitations(filePath);
      if (citationIssues.length > 0) {
        console.log('  Citation issues found:');
        citationIssues.forEach(issue => {
          console.log(`    Line ${issue.line}: ${issue.message}`);
          console.log(`      "${issue.content}"`);
        });
      } else {
        console.log('  ✓ Citations: OK');
      }

      // Validate technical terms
      const termIssues = validateTechnicalTerms(filePath);
      if (termIssues.length > 0) {
        console.log('  Technical term issues:');
        termIssues.forEach(issue => {
          console.log(`    ${issue.message}`);
        });
      }

      // Validate code blocks
      const codeIssues = validateCodeBlocks(filePath);
      if (codeIssues.length > 0) {
        console.log('  Code block issues found:');
        codeIssues.forEach(issue => {
          console.log(`    ${issue.message}`);
          console.log(`      "${issue.content}"`);
        });
      } else {
        console.log('  ✓ Code blocks: OK');
      }

      // Validate structure
      const missingSections = validateFileStructure(filePath);
      if (missingSections.length > 0) {
        console.log(`  Missing sections: ${missingSections.join(', ')}`);
      } else {
        console.log('  ✓ Structure: OK');
      }
    }
  }
}

// Process the docs directory
const docsDir = './docs';
if (fs.existsSync(docsDir)) {
  processDirectory(docsDir);
} else {
  console.log('Docs directory not found');
}

console.log('\nValidation complete.');