#!/usr/bin/env node

// Cross-reference validation script for Humanoid Robotics Book
// Checks that cross-references between chapters are valid and functional

const fs = require('fs');
const path = require('path');

function findMarkdownFiles(dirPath) {
  let results = [];

  const files = fs.readdirSync(dirPath);

  for (const file of files) {
    const filePath = path.join(dirPath, file);
    const stat = fs.statSync(filePath);

    if (stat.isDirectory()) {
      results = results.concat(findMarkdownFiles(filePath));
    } else if (file.endsWith('.md')) {
      results.push(filePath);
    }
  }

  return results;
}

function extractCrossReferences(content) {
  // Find different types of cross-references in the content
  const references = {
    internalLinks: [],  // [text](./path.md) or [text](/docs/path)
    anchorLinks: [],    // [text](#section-title)
    docReferences: []   // [text](@docusaurus)
  };

  // Match internal links like [text](./path.md) or [text](/docs/path)
  const internalLinkRegex = /\[([^\]]+)\]\(([^)]*\.(md|html)|\/docs\/[^\)]+)\)/g;
  let match;
  while ((match = internalLinkRegex.exec(content)) !== null) {
    references.internalLinks.push({
      text: match[1],
      link: match[2],
      fullMatch: match[0]
    });
  }

  // Match anchor links like [text](#section-title)
  const anchorLinkRegex = /\[([^\]]+)\]\(#([^\)]+)\)/g;
  while ((match = anchorLinkRegex.exec(content)) !== null) {
    references.anchorLinks.push({
      text: match[1],
      anchor: match[2],
      fullMatch: match[0]
    });
  }

  // Match Docusaurus doc references like [text](doc:filename)
  const docRefRegex = /\[([^\]]+)\]\(doc:([^\)]+)\)/g;
  while ((match = docRefRegex.exec(content)) !== null) {
    references.docReferences.push({
      text: match[1],
      doc: match[2],
      fullMatch: match[0]
    });
  }

  return references;
}

function validateCrossReferences(filePath, allFiles) {
  const content = fs.readFileSync(filePath, 'utf8');
  const references = extractCrossReferences(content);
  const issues = [];

  // Check internal links
  for (const ref of references.internalLinks) {
    let targetPath = ref.link;

    // Handle relative paths
    if (targetPath.startsWith('./') || targetPath.startsWith('../')) {
      const dir = path.dirname(filePath);
      targetPath = path.resolve(dir, targetPath);
    } else if (targetPath.startsWith('/docs/')) {
      // Convert /docs/path to docs/path
      targetPath = targetPath.substring(1);
    }

    // If it doesn't have an extension, try adding .md
    if (!path.extname(targetPath)) {
      targetPath += '.md';
    }

    if (!fs.existsSync(targetPath)) {
      issues.push({
        type: 'broken_link',
        reference: ref.fullMatch,
        target: targetPath,
        message: `Broken internal link: ${ref.link}`
      });
    }
  }

  // Check anchor links (basic validation - we'd need to parse the target file to fully validate)
  for (const ref of references.anchorLinks) {
    // For now, just log anchor references for manual review
    issues.push({
      type: 'anchor_reference',
      reference: ref.fullMatch,
      target: ref.anchor,
      message: `Anchor reference found: ${ref.anchor} (requires manual validation)`
    });
  }

  // Check doc references
  for (const ref of references.docReferences) {
    // Look for a file that might match the doc reference
    const docName = ref.doc;
    let found = false;

    for (const file of allFiles) {
      const fileName = path.basename(file, '.md');
      if (fileName === docName || fileName === docName.replace('/', '')) {
        found = true;
        break;
      }
    }

    if (!found) {
      issues.push({
        type: 'broken_doc_ref',
        reference: ref.fullMatch,
        target: ref.doc,
        message: `Unresolved doc reference: ${ref.doc}`
      });
    }
  }

  return issues;
}

function main() {
  const docsDir = './docs';
  if (!fs.existsSync(docsDir)) {
    console.log('Docs directory not found');
    return;
  }

  const allFiles = findMarkdownFiles(docsDir);

  console.log(`Found ${allFiles.length} markdown files in docs directory\n`);

  for (const file of allFiles) {
    console.log(`Validating cross-references in: ${file}`);

    const issues = validateCrossReferences(file, allFiles);

    if (issues.length > 0) {
      console.log('  Issues found:');
      issues.forEach(issue => {
        console.log(`    ${issue.message}`);
        console.log(`      Reference: ${issue.reference}`);
      });
    } else {
      console.log('  ✓ Cross-references: OK');
    }

    console.log('');
  }

  console.log('Cross-reference validation complete.');
}

main();