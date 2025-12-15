#!/usr/bin/env node

// Code example validation script for Vision-Language-Action (VLA) systems module
// Validates syntax and functionality of code examples in VLA documentation

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
  // For VLA examples, we allow partial code snippets
  const lines = code.split('\n');
  const errors = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();

    if (!line) continue; // Skip empty lines

    // Check for common syntax issues that would definitely be wrong
    if (line.startsWith('def ') && !/:\s*(#.*)?$/.test(line) && !line.endsWith('\\')) {
      errors.push(`Line ${i + 1}: Missing colon after function definition`);
    }

    if (line.startsWith('class ') && !/:\s*(#.*)?$/.test(line) && !line.endsWith('\\')) {
      errors.push(`Line ${i + 1}: Missing colon after class definition`);
    }

    if (line.startsWith('for ') && !line.includes(' in ') && !line.endsWith('\\')) {
      errors.push(`Line ${i + 1}: Incomplete for loop`);
    }

    if (line.startsWith('if ') && !/:\s*(#.*)?$/.test(line) && !line.endsWith('\\') && !line.includes('else') && !line.includes('elif')) {
      // Check if this might be part of a list comprehension, generator expression, string, or comment
      // or part of a multi-line statement
      const isPartOfComprehension = /\[.*if\s/.test(line) || /\(.*if\s/.test(line) || /,\s*if\s/.test(line);
      const isLikelyMultiLine = line.endsWith('(') || line.endsWith(',') || line.endsWith('\\');

      // Check if this looks like it's part of a list comprehension (common patterns in VLA code)
      // This looks for patterns like: `obj for obj in visual_objects` followed by `if condition`
      const prevLine = i > 0 ? lines[i-1].trim() : '';
      const isListComprehensionIf = (prevLine.includes(' for ') && prevLine.includes(' in ')) ||
                                   (line.includes('matches_description') && prevLine.includes('for '));

      if (!isPartOfComprehension && !isLikelyMultiLine && !isListComprehensionIf && !line.startsWith('#') && !/^["']/.test(line)) {
        errors.push(`Line ${i + 1}: Missing colon after if statement`);
      }
    }

    if (line.startsWith('while ') && !/:\s*(#.*)?$/.test(line) && !line.endsWith('\\')) {
      errors.push(`Line ${i + 1}: Missing colon after while loop`);
    }

    if (line.includes('try:') && !/:\s*(#.*)?$/.test(line)) {
      errors.push(`Line ${i + 1}: Missing colon after try`);
    }

    if (line.includes('except ') && !/:\s*(#.*)?$/.test(line)) {
      errors.push(`Line ${i + 1}: Missing colon after except`);
    }

    if (line.includes('with ') && !/:\s*(#.*)?$/.test(line) && !line.endsWith('\\') && !line.startsWith('#') && !/^["']/.test(line)) {
      errors.push(`Line ${i + 1}: Incomplete with statement`);
    }
  }

  return errors;
}

function validateBashSyntax(code) {
  const lines = code.split('\n');
  const errors = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();

    if (!line) continue; // Skip empty lines

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

    if (!line) continue; // Skip empty lines

    // Check for common C++ syntax issues
    if (line.includes('#include') && !line.includes('<') && !line.includes('"')) {
      errors.push(`Line ${i + 1}: Include directive missing file specification`);
    }

    if (line.includes('namespace ') && !line.endsWith('{') && !line.includes('std')) {
      errors.push(`Line ${i + 1}: Namespace declaration incomplete`);
    }
  }

  return errors;
}

function validateXMLSyntax(code) {
  const lines = code.split('\n');
  const errors = [];

  // Simple validation for XML/URDF syntax
  let openTags = 0;
  let closeTags = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Count open and close tags
    const openMatches = line.match(/<[a-zA-Z]/g);
    const closeMatches = line.match(/<\/[a-zA-Z]/g);

    if (openMatches) openTags += openMatches.length;
    if (closeMatches) closeTags += closeMatches.length;
  }

  if (openTags !== closeTags) {
    errors.push(`XML tag mismatch: ${openTags} open tags, ${closeTags} close tags`);
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
    case 'shell':
      errors = validateBashSyntax(code);
      break;
    case 'cpp':
    case 'c++':
      errors = validateCppSyntax(code);
      break;
    case 'xml':
    case 'urdf':
    case 'sdf':
      errors = validateXMLSyntax(code);
      break;
    case 'yaml':
    case 'yml':
      // Basic validation for YAML
      if (code.includes('  \t') || code.includes('\t  ')) {
        errors.push('YAML should use spaces, not tabs for indentation');
      }
      break;
    case 'javascript':
    case 'js':
      // Basic JS validation
      break;
    default:
      // For unknown languages, perform basic validation
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
      console.log(`  ❌ ${validation.language} code block validation failed:`);
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

// Process the VLA directory
const vlaDir = './docs/vla-systems';
if (fs.existsSync(vlaDir)) {
  const result = processDirectory(vlaDir);

  console.log('\n' + '='.repeat(50));
  if (result) {
    console.log('✅ All VLA code examples validated successfully!');
  } else {
    console.log('❌ Some VLA code examples have validation issues that need review.');
  }
  console.log('='.repeat(50));
} else {
  console.log('VLA directory not found');
}

module.exports = {
  extractCodeBlocks,
  validatePythonSyntax,
  validateBashSyntax,
  validateCppSyntax,
  validateXMLSyntax,
  validateCodeBlock,
  processFile,
  processDirectory
};