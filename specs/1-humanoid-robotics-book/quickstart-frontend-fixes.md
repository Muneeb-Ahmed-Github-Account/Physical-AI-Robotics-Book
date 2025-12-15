# Quickstart Guide: Docusaurus Frontend Fixes

**Feature**: 1-humanoid-robotics-book | **Date**: 2025-12-14

## Overview

This guide provides step-by-step instructions to implement the fixes for Docusaurus frontend issues: missing static assets, navbar logo not appearing, and homepage "Page Not Found" error.

## Prerequisites

- Node.js and npm installed
- Access to the project repository
- Basic understanding of Docusaurus structure

## Step-by-Step Implementation

### Step 1: Create Static Assets Directory

1. Create the static directory at the project root:
```bash
mkdir static
```

2. Create the img subdirectory:
```bash
mkdir static/img
```

### Step 2: Add Required Static Assets

1. Create or obtain the following image files:
   - `favicon.ico` - 16x16 or 32x32 pixel ICO file
   - `logo.svg` - SVG logo file (suggested size: 150x40 pixels)
   - `docusaurus-social-card.jpg` - JPG image (suggested size: 1200x630 pixels)

2. Place these files in the `static/img/` directory:
```bash
# After creating or downloading the images
cp favicon.ico static/img/
cp logo.svg static/img/
cp docusaurus-social-card.jpg static/img/
```

### Step 3: Configure Homepage

1. Open `docs/intro.md` in your editor

2. Add the following frontmatter at the top of the file:
```markdown
---
slug: /
title: Humanoid Robotics & Physical AI Course Book
---
```

3. The complete intro.md file should start like this:
```markdown
---
slug: /
title: Humanoid Robotics & Physical AI Course Book
sidebar_position: 1
---

# Introduction

Welcome to the Humanoid Robotics & Physical AI Course Book...
```

### Step 4: Verify Configuration

1. Ensure your `docusaurus.config.js` has the correct asset references:
   - `favicon: 'img/favicon.ico'`
   - `logo: { src: 'img/logo.svg' }`
   - `image: 'img/docusaurus-social-card.jpg'`

2. These should match the file paths created in Step 2.

### Step 5: Test the Implementation

1. Start the local development server:
```bash
npm run start
```

2. Visit `http://localhost:3000/humanoid-robotics-book/` (or your configured baseUrl)

3. Verify:
   - Navbar logo appears correctly
   - Favicon loads in the browser tab
   - Homepage displays the intro content
   - No "Page Not Found" errors

### Step 6: Build and Validate

1. Run a production build to ensure everything works:
```bash
npm run build
```

2. Serve the build locally to test:
```bash
npm run serve
```

3. Visit the served site and verify all functionality works as expected.

## Troubleshooting

### Logo still not appearing
- Verify the `logo.svg` file exists in `static/img/`
- Check that the file is not corrupted
- Ensure the file size is reasonable (< 100KB recommended)

### Homepage still showing "Page Not Found"
- Verify the `slug: /` frontmatter is correctly added to `docs/intro.md`
- Ensure no other homepage files conflict (like `src/pages/index.js` if it existed)
- Restart the development server after making changes

### Build errors
- Check for typos in file paths
- Verify all referenced assets exist
- Ensure proper file permissions

## Next Steps

1. Deploy the updated site to your hosting platform
2. Test on different browsers and devices
3. Verify social sharing functionality with the new social card image