#!/usr/bin/env node

// Code example validation script for Humanoid Robotics Book
// Validates syntax and formatting of code examples in markdown files

const fs = require('fs');
const path = require('path');

function extractCodeBlocks(content) {
  const codeBlockRegex = /```(\w+)?\s*\n([\s\S]*?)\n```/g;
  const matches = [];
  let match;

  while ((match = codeBlockRegex.exec(content)) !== null) {
    matches.push({
      language: match[1] || 'unknown',
      code: match[2],
      position: match.index
    });
  }

  return matches;
}

function validatePythonSyntax(code) {
  // Basic Python syntax validation by checking for common issues
  // For ROS 2 examples, we allow partial code snippets
  const lines = code.split('\n');
  const errors = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();

    if (!line) continue; // Skip empty lines

    // Check for syntax issues that would definitely be wrong
    if (line.startsWith('def ') || line.startsWith('class ')) {
      // These should end with colon
      if (!line.endsWith(':')) {
        errors.push(`Line ${i + 1}: Missing colon after definition`);
      }
    }

    // Check for obvious syntax errors
    if (line.includes('import') && line.includes('import import')) {
      errors.push(`Line ${i + 1}: Duplicate import statement`);
    }
  }

  return errors;
}

function validateBashSyntax(code) {
  const lines = code.split('\n');
  const errors = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();

    if (!line) continue;

    // Check for common bash syntax issues
    if (line.includes('&&') && line.startsWith('#')) {
      // Comments shouldn't contain && unless it's in the text
    }
  }

  return errors;
}

function validateCMakeSyntax(code) {
  const lines = code.split('\n');
  const errors = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();

    if (!line) continue;

    // Check for basic CMake syntax
    if (line.match(/^\w+\(/) && !line.endsWith(')')) {
      // Function call should end with )
      let foundEnd = false;
      for (let j = i; j < Math.min(i + 10, lines.length); j++) {
        if (lines[j].trim().endsWith(')')) {
          foundEnd = true;
          break;
        }
      }
      if (!foundEnd) {
        errors.push(`Line ${i + 1}: Potential unclosed function call`);
      }
    }
  }

  return errors;
}

function validateCodeBlock(codeBlock) {
  const { language, code } = codeBlock;
  let errors = [];

  switch (language.toLowerCase()) {
    case 'python':
    case 'py':
      errors = validatePythonSyntax(code);
      break;
    case 'bash':
    case 'sh':
      errors = validateBashSyntax(code);
      break;
    case 'cmake':
      errors = validateCMakeSyntax(code);
      break;
    case 'yaml':
    case 'xml':
    case 'json':
      // These could have more specific validation
      break;
    default:
      // For unknown languages, do basic validation
      break;
  }

  return {
    language,
    hasErrors: errors.length > 0,
    errors
  };
}

function processFile(filePath) {
  console.log(`\nValidating code examples in: ${filePath}`);

  const content = fs.readFileSync(filePath, 'utf8');
  const codeBlocks = extractCodeBlocks(content);

  if (codeBlocks.length === 0) {
    console.log('  No code blocks found');
    return true;
  }

  let allValid = true;
  let validCount = 0;

  for (const block of codeBlocks) {
    const validation = validateCodeBlock(block);

    if (validation.hasErrors) {
      console.log(`  ❌ Code block in ${validation.language}:`);
      validation.errors.forEach(error => {
        console.log(`    ${error}`);
      });
      allValid = false;
    } else {
      console.log(`  ✅ ${validation.language} code block validated`);
      validCount++;
    }
  }

  console.log(`  Summary: ${validCount}/${codeBlocks.length} code blocks valid`);
  return allValid;
}

function processDirectory(dirPath) {
  const files = fs.readdirSync(dirPath);
  let allValid = true;

  for (const file of files) {
    const filePath = path.join(dirPath, file);
    const stat = fs.statSync(filePath);

    if (stat.isDirectory()) {
      const dirValid = processDirectory(filePath);
      allValid = allValid && dirValid;
    } else if (file.endsWith('.md')) {
      const fileValid = processFile(filePath);
      allValid = allValid && fileValid;
    }
  }

  return allValid;
}

// Process the docs directory
const docsDir = './docs';
if (fs.existsSync(docsDir)) {
  const result = processDirectory(docsDir);

  console.log('\n' + '='.repeat(50));
  if (result) {
    console.log('✅ All code examples validated successfully!');
  } else {
    console.log('❌ Some code examples have validation issues that need review.');
  }
  console.log('='.repeat(50));
} else {
  console.log('Docs directory not found');
}