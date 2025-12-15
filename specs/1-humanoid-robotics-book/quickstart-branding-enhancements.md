# Quickstart Guide: Homepage Branding & Visual Design Enhancements

**Feature**: 1-humanoid-robotics-book | **Date**: 2025-12-14

## Overview

This guide provides step-by-step instructions to implement the branding and visual design enhancements for the Humanoid Robotics Book website: optimizing the navbar logo with dark mode support, enhancing the homepage hero section with a professional background, and ensuring accessibility and performance standards.

## Prerequisites

- Node.js and npm installed
- Access to the project repository
- Basic understanding of Docusaurus structure
- Image editing software (for creating optimized logos and backgrounds)

## Step-by-Step Implementation

### Step 1: Optimize and Create Logo Assets

1. Optimize the existing logo.svg file:
   - Remove unnecessary metadata and comments
   - Minify the SVG code
   - Ensure dimensions are appropriate (150-200px width, under 50px height)

2. Create a dark mode logo variant:
   - Create `logo-dark.svg` in the `static/img/` directory
   - Adjust colors to be visible against dark backgrounds
   - Maintain the same dimensions as the light mode logo

3. Place both files in the static/img directory:
```bash
# After creating or optimizing the logos
cp logo.svg static/img/
cp logo-dark.svg static/img/
```

### Step 2: Create Homepage Background Image

1. Design a professional background image:
   - Subtle tech-inspired pattern or abstract geometric design
   - Reflects robotics/physical AI theme without being distracting
   - Optimized for web delivery (JPG format recommended)

2. Place the background image in the static/img directory:
```bash
cp hero-background.jpg static/img/
```

### Step 3: Update Docusaurus Configuration

1. Open `docusaurus.config.js` in your editor

2. Update the navbar logo configuration to include dark mode support:
```javascript
logo: {
  alt: 'Humanoid Robotics Logo',
  src: 'img/logo.svg',
  srcDark: 'img/logo-dark.svg',  // Add this line
  href: '/humanoid-robotics-book/docs/intro',
  target: '_self',
  width: 180,  // Optional: specify width
  height: 40,  // Optional: specify height
},
```

### Step 4: Enhance Homepage Component

1. Open `src/pages/index.js` in your editor

2. Update the component to include background styling and improved layout:
```javascript
import React from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Welcome to ${siteConfig.title}`}
      description="Overview of the Humanoid Robotics & Physical AI Course Book">
      <main>
        <section
          className="hero hero--primary"
          style={{
            backgroundImage: 'url(/img/hero-background.jpg)',
            backgroundSize: 'cover',
            backgroundPosition: 'center',
            padding: '80px 20px',
            textAlign: 'center',
            minHeight: '500px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          <div
            style={{
              backgroundColor: 'rgba(0, 0, 0, 0.7)',
              padding: '40px',
              borderRadius: '8px',
              maxWidth: '800px'
            }}
          >
            <h1
              className="hero__title"
              style={{
                fontSize: '2.5rem',
                marginBottom: '20px',
                color: 'white'
              }}
            >
              {siteConfig.title}
            </h1>
            <p
              className="hero__subtitle"
              style={{
                fontSize: '1.2rem',
                marginBottom: '30px',
                color: '#ddd'
              }}
            >
              {siteConfig.tagline}
            </p>
            <div>
              <Link
                className="button button--primary button--lg"
                to="/docs/intro">
                Read the Book - Start Learning
              </Link>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
```

### Step 5: Add Custom CSS (Optional Enhancement)

1. Open or create `src/css/custom.css`

2. Add custom styles for the enhanced homepage:
```css
/* Enhanced homepage hero styles */
.hero {
  position: relative;
}

.hero::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(135deg, rgba(22, 22, 22, 0.9) 0%, rgba(44, 44, 44, 0.8) 100%);
  z-index: 1;
}

.hero > .container {
  position: relative;
  z-index: 2;
}

/* Dark mode specific styles */
html[data-theme='dark'] .hero {
  background-color: #1a1a1a;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .hero__title {
    font-size: 2rem !important;
  }

  .hero__subtitle {
    font-size: 1rem !important;
  }

  .hero-section {
    padding: 60px 20px !important;
    min-height: 400px;
  }
}
```

### Step 6: Test the Implementation

1. Start the local development server:
```bash
npm run start
```

2. Visit `http://localhost:3000/humanoid-robotics-book/` (or your configured baseUrl)

3. Verify:
   - Navbar logo appears correctly in both light and dark modes
   - Homepage has the enhanced background
   - Text is readable with good contrast
   - Layout is responsive on different screen sizes
   - Call-to-action button is visible and functional

### Step 7: Validate Accessibility and Performance

1. Run accessibility checks:
   - Use browser dev tools or extensions like axe or WAVE
   - Verify contrast ratios meet WCAG 2.1 AA standards (4.5:1 minimum)

2. Check performance:
   - Run Lighthouse audit in browser dev tools
   - Verify page loads quickly with optimized images
   - Confirm no console errors

### Step 8: Build and Final Validation

1. Run a production build to ensure everything works:
```bash
npm run build
```

2. Serve the build locally to test:
```bash
npm run serve
```

3. Visit the served site and verify all enhancements work as expected in production mode.

## Troubleshooting

### Dark mode logo not appearing
- Verify the `logo-dark.svg` file exists in `static/img/`
- Check that the `srcDark` property is correctly added to the navbar configuration
- Ensure the file is not corrupted

### Background image not loading
- Verify the background image file exists in `static/img/`
- Check that the path in the component is correct
- Ensure the image is properly optimized for web delivery

### Layout issues on mobile
- Verify responsive CSS is properly implemented
- Check that the background image scales appropriately
- Ensure text remains readable at smaller sizes

## Next Steps

1. Deploy the updated site to your hosting platform
2. Monitor performance metrics after deployment
3. Gather user feedback on the new design
4. Iterate based on usage analytics and feedback