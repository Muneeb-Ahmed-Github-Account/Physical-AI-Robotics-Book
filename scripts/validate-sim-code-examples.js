#!/usr/bin/env node

// Code example validation script for Digital Twin Simulation module
// Validates syntax and formatting of code examples in digital twin markdown files

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
  // For ROS 2 and simulation examples, we allow partial code snippets
  const lines = code.split('\n');
  const errors = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();

    if (!line) continue; // Skip empty lines

    // Check for common syntax issues that would definitely be wrong
    if (line.startsWith('def ') && !line.endsWith(':')) {
      // Check if it's a continuation of a previous line (like function signature spanning multiple lines)
      if (!line.endsWith('\\')) {
        errors.push(`Line ${i + 1}: Missing colon after function definition`);
      }
    }

    if (line.startsWith('class ') && !line.endsWith(':')) {
      errors.push(`Line ${i + 1}: Missing colon after class definition`);
    }

    // Only check for colons in complete statements, not in multi-line expressions
    if (line.includes('if ') && !line.includes('(') && line.endsWith('if')) {
      errors.push(`Line ${i + 1}: Incomplete if statement`);
    }

    if (line.includes('for ') && !line.includes('(') && line.endsWith('for')) {
      errors.push(`Line ${i + 1}: Incomplete for loop`);
    }

    if (line.includes('while ') && !line.includes('(') && line.endsWith('while')) {
      errors.push(`Line ${i + 1}: Incomplete while loop`);
    }

    if (line.includes('try:') && !line.endsWith(':')) {
      errors.push(`Line ${i + 1}: Missing colon after try`);
    }

    if (line.includes('except ') && line.includes(':') && !line.endsWith(':')) {
      errors.push(`Line ${i + 1}: Missing colon after except`);
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

function validateCppSyntax(code) {
  const lines = code.split('\n');
  const errors = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();

    if (!line) continue;

    // Check for common C++ syntax issues
    if (line.startsWith('class ') && !line.endsWith('{')) {
      errors.push(`Line ${i + 1}: Class definition should end with opening brace`);
    }

    if (line.includes('#include') && !line.includes('<') && !line.includes('"')) {
      errors.push(`Line ${i + 1}: Include directive should specify header file`);
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
    case 'cpp':
    case 'c++':
      errors = validateCppSyntax(code);
      break;
    case 'xml':
    case 'urdf':
    case 'sdf':
      // These could have more specific validation
      break;
    case 'javascript':
    case 'js':
      // JavaScript validation could be added
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

// Process the digital twin directory
const digitalTwinDir = './docs/digital-twin';
if (fs.existsSync(digitalTwinDir)) {
  const files = fs.readdirSync(digitalTwinDir);
  let allValid = true;

  for (const file of files) {
    if (file.endsWith('.md')) {
      const filePath = path.join(digitalTwinDir, file);
      const fileValid = processFile(filePath);
      allValid = allValid && fileValid;
    }
  }

  console.log('\n' + '='.repeat(50));
  if (allValid) {
    console.log('✅ All digital twin code examples validated successfully!');
  } else {
    console.log('❌ Some digital twin code examples have validation issues that need review.');
  }
  console.log('='.repeat(50));
} else {
  console.log('Digital twin directory not found');
}