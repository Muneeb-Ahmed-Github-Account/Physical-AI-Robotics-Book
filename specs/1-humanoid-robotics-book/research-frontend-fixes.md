# Research: Docusaurus Frontend Fixes for Humanoid Robotics Book

**Feature**: 1-humanoid-robotics-book | **Date**: 2025-12-14

## Research Summary

This document addresses the three main frontend issues identified in the Docusaurus site analysis:
1. Missing static assets causing navbar logo not to appear
2. "Page Not Found" error on homepage due to missing root route
3. Missing static assets directory structure

## Issue 1: Missing Static Assets Directory and Files

### Decision: Create Static Assets Directory
**Rationale**: Docusaurus follows a convention where static assets (images, favicons, etc.) are stored in a `static/` directory at the project root. These files are copied to the build output root, making them accessible via the configured `baseUrl`.

**Implementation approach**:
- Create `static/` directory at project root
- Create `static/img/` subdirectory for images
- Add required image files:
  - `favicon.ico` - Site favicon
  - `logo.svg` - Navbar logo
  - `docusaurus-social-card.jpg` - Social sharing image

### Alternatives considered:
1. Use public/ directory - Docusaurus also supports this but static/ is the conventional location
2. Inline SVG for logo - Would increase bundle size and complicate maintenance
3. External image hosting - Would create dependency on external resources

## Issue 2: Homepage "Page Not Found" Error

### Decision: Configure docs/intro.md as homepage
**Rationale**: The most straightforward approach that maintains content modularity and leverages existing documentation. By adding a `slug: /` frontmatter to the intro.md file, it will serve as the homepage while remaining part of the docs structure.

**Implementation approach**:
- Add frontmatter to `docs/intro.md`:
```markdown
---
slug: /
title: Humanoid Robotics & Physical AI Course Book
---
```

### Alternatives considered:
1. Create custom homepage component - Would require more complex React development
2. Create separate index page - Would duplicate content already in intro.md
3. Redirect root to /docs/ - Would provide poor user experience

## Issue 3: Base URL and Routing Configuration

### Decision: Maintain existing baseUrl configuration
**Rationale**: The current `baseUrl: '/humanoid-robotics-book/'` is correctly configured for GitHub Pages deployment. The routing issues stem from missing assets and homepage, not from the baseUrl setting.

**Implementation approach**:
- Keep existing baseUrl configuration in `docusaurus.config.js`
- Ensure static assets and homepage are properly configured to work with this baseUrl

### Alternatives considered:
1. Change baseUrl to '/' - Would break GitHub Pages deployment
2. Use different routeBasePath for docs - Would complicate navigation

## Technical Implementation Details

### Static Assets Best Practices
- Image formats: SVG for logos (scalable), ICO for favicon (standard format), JPG for social cards (good compression for photos)
- Image optimization: All images should be optimized for web delivery
- Directory structure: Follow Docusaurus convention of `static/img/`

### Homepage Configuration
- The slug approach maintains the documentation structure while providing a proper homepage
- No need to remove existing `src/pages/` directory if it doesn't exist (which it doesn't in this project)

## Validation Approach

Each fix will be validated by:
1. Running local Docusaurus server (`npm run start`)
2. Testing asset loading (navbar logo, favicon)
3. Verifying homepage displays correctly at root URL
4. Running build process (`npm run build`) to ensure no errors
5. Serving built site locally (`npm run serve`) to verify production build