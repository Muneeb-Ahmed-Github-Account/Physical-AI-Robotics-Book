#!/usr/bin/env node

// Plagiarism detection script for Humanoid Robotics Book
// Checks for potential plagiarism by identifying long sequences of unoriginal content

const fs = require('fs');
const path = require('path');

function calculateSimilarity(str1, str2) {
  // Simple similarity calculation using character n-grams
  const n = 3; // trigrams
  const normalize = s => s.toLowerCase().replace(/\s+/g, '');

  const s1 = normalize(str1);
  const s2 = normalize(str2);

  if (s1.length < n || s2.length < n) {
    return 0;
  }

  // Generate n-grams
  const getNGrams = str => {
    const ngrams = [];
    for (let i = 0; i <= str.length - n; i++) {
      ngrams.push(str.substring(i, i + n));
    }
    return ngrams;
  };

  const ngrams1 = getNGrams(s1);
  const ngrams2 = getNGrams(s2);

  // Calculate Jaccard similarity
  const set1 = new Set(ngrams1);
  const set2 = new Set(ngrams2);
  const intersection = new Set([...set1].filter(x => set2.has(x)));
  const union = new Set([...set1, ...set2]);

  return intersection.size / union.size;
}

function detectPotentialPlagiarism(filePath, threshold = 0.8) {
  const content = fs.readFileSync(filePath, 'utf8');
  const lines = content.split('\n');
  const issues = [];

  // Look for long sequences of text that might be copied
  const textBlocks = [];
  let currentBlock = [];

  for (const line of lines) {
    // Skip markdown formatting lines
    if (!line.trim() || line.startsWith('#') || line.startsWith('```') ||
        line.startsWith('- ') || line.startsWith('* ') || line.startsWith('1.')) {
      if (currentBlock.length > 0) {
        textBlocks.push(currentBlock.join(' '));
        currentBlock = [];
      }
      continue;
    }

    // Add line to current block if it's regular text
    if (line.trim() && !line.startsWith('[') && !line.startsWith('|')) {
      currentBlock.push(line.trim());
    } else {
      if (currentBlock.length > 0) {
        textBlocks.push(currentBlock.join(' '));
        currentBlock = [];
      }
    }
  }

  if (currentBlock.length > 0) {
    textBlocks.push(currentBlock.join(' '));
  }

  // Check for similar blocks within the same file (self-plagiarism)
  for (let i = 0; i < textBlocks.length; i++) {
    for (let j = i + 1; j < textBlocks.length; j++) {
      const similarity = calculateSimilarity(textBlocks[i], textBlocks[j]);
      if (similarity > threshold) {
        issues.push({
          type: 'self_similarity',
          similarity: similarity.toFixed(2),
          block1: textBlocks[i].substring(0, 100) + '...',
          block2: textBlocks[j].substring(0, 100) + '...',
          message: `High similarity detected within file (${(similarity * 100).toFixed(0)}%)`
        });
      }
    }
  }

  // Check for very long sentences that might indicate direct copying
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.length > 200) { // Arbitrarily long line
      issues.push({
        type: 'long_sentence',
        line: i + 1,
        content: line.substring(0, 100) + '...',
        message: 'Long sentence detected - possible direct copy'
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
      console.log(`\nChecking for potential plagiarism: ${filePath}`);

      const issues = detectPotentialPlagiarism(filePath);

      if (issues.length > 0) {
        console.log('  Potential issues found:');
        issues.forEach(issue => {
          console.log(`    ${issue.message}`);
          if (issue.line) {
            console.log(`      Line: ${issue.line}`);
          }
          if (issue.similarity) {
            console.log(`      Similarity: ${issue.similarity}`);
          }
          console.log(`      Content: ${issue.content || issue.block1}`);
        });
      } else {
        console.log('  ✓ No obvious plagiarism issues detected');
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

console.log('\nPlagiarism detection complete. Remember to verify all content is properly attributed and rewritten from original sources.');